"""Interfacing to DAQ via MongoDB

The DAQ uses MongoDB for input and output.  The classes defined hear allow the
user to read data from the DAQ and also inject raw occurences into the DAQ.

"""
from datetime import datetime

import numpy as np

import time
import pymongo
import snappy
from bson.binary import Binary
from pax.datastructure import Event, Occurrence
from pax import plugin, units


START_TIME_KEY = 'time_min'


class MongoDBInput(plugin.InputPlugin):

    """Read data from DAQ database

    This assumes that an event builder has run.
    """

    def startup(self):
        self.log.debug("Connecting to %s" % self.config['address'])
        try:
            self.client = pymongo.MongoClient(self.config['address'])
            self.database = self.client[self.config['database']]
            self.collection = self.database[self.config['collection']]
        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise

        # TODO (tunnell): Sort by event number
        self.cursor = self.collection.find()
        self.number_of_events = self.cursor.count()

        self.mongo_time_unit = self.config.get('mongo_time_unit',
                                               10 * units.ns)

        if self.number_of_events == 0:
            raise RuntimeError("No events found... did you run the event"
                               "builder?")

    def number_events(self):
        return self.number_of_events

    def get_events(self):
        """Generator of events from Mongo
        """
        for i, doc_event in enumerate(self.collection.find()):
            self.log.debug("Fetching document %s" % repr(doc_event['_id']))

            # Store channel waveform-occurrences by iterating over all
            # occurrences.
            # This involves parsing MongoDB documents using WAX output format
            assert isinstance(doc_event['range'][0], int)
            assert isinstance(doc_event['range'][1], int)

            # Convert from Mongo's time unit to pax units
            event = Event(n_channels=self.config['n_channels'],
                          start_time=int(doc_event['range'][0]) * self.mongo_time_unit,
                          sample_duration=self.config['sample_duration'],
                          stop_time=int(doc_event['range'][1]) * self.mongo_time_unit)

            event.event_number = i  # TODO: should come from Mongo

            assert isinstance(event.start_time, int)
            assert isinstance(event.stop_time, int)

            for doc_occurrence in doc_event['docs']:

                assert isinstance(doc_occurrence['time'], int)
                assert isinstance(doc_event['range'][0], int)

                data = snappy.compress(doc_occurrence['data'])

                event.occurrences.append(Occurrence(
                    left=doc_occurrence['time'] - doc_event['range'][0],
                    raw_data=np.fromstring(data, dtype="<i2"),
                    channel=doc_occurrence['channel']
                ))

            if event.length() == 0:
                raise RuntimeWarning("Empty event")

            yield event


class MongoDBFakeDAQOutput(plugin.OutputPlugin):

    """Inject PMT pulses into DAQ to test trigger.

    This plugin aims to emulate the DAQReader by creating run control documents
    and feeding raw data into the DAQ's MongoDB format.  Consult here for more
    on formats:

    https://docs.google.com/drawings/d/1dytKBmMARsZtuyUmLbzm9IbXM1hre
    -knkEIU4X3Ot8U/edit

    Note: write run document after output collection created.
    """

    def startup(self):
        """Setup"""

        # Collect all events in a buffer, then inject them at the end.
        self.collect_then_dump = self.config['collect_then_dump']
        self.repeater = int(self.config['repeater'])  # Hz repeater
        self.runtime = int(self.config['runtime'])  # How long run repeater

        # Schema for input collection
        self.start_time_key = START_TIME_KEY
        self.stop_time_key = 'time_max'
        self.bulk_key = 'bulk'

        self.connections = {}

        try:
            self.client = pymongo.MongoClient(self.config['run_address'])

            # Used for coordinating which runs to analyze
            self.log.debug("Connecting to %s" % self.config['run_address'])
            self.run_client = self.get_connection(self.config['run_address'])

            # Used for storing the binary output from digitizers
            self.log.debug("Connecting to %s" % self.config['raw_address'])
            self.raw_client = self.get_connection(self.config['raw_address'])

        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise

        self.run_database = self.run_client[self.config['run_database']]
        self.raw_database = self.raw_client[self.config['raw_database']]

        self.run_collection = self.run_database[self.config['run_collection']]
        self.raw_collection = self.raw_database[self.config['raw_collection']]

        self.raw_collection.ensure_index([(self.start_time_key, -1)])
        self.raw_collection.ensure_index([(self.start_time_key, 1)])
        self.raw_collection.ensure_index([(self.stop_time_key, -1)])
        self.raw_collection.ensure_index([(self.stop_time_key, 1)])

        self.raw_collection.ensure_index([('_id', pymongo.HASHED)])

        # self.log.info("Sharding %s" % str(c))
        # self.raw_client.admin.command('shardCollection',
        #                              '%s.%s' % (self.config['raw_database'], self.config['raw_collection']),
        #                              key = {'_id': pymongo.HASHED})

        # Send run doc
        self.query = {"name": self.config['name'],
                      "starttimestamp": str(datetime.now()),
                      "runmode": "calibration",
                      "reader": {
                          "compressed": True,
                          "starttimestamp": 0,
                          "data_taking_ended": False,
                          "options": {},
                          "storage_buffer": {
                              "dbaddr": self.config['raw_address'],
                              "dbname": self.config['raw_database'],
                              "dbcollection": self.config['raw_collection'],
                          },
        },
            "trigger": {
                          "mode": "calibration",
                          "status": "waiting_to_be_processed",
        },
            "processor": {"mode": "something"},
            "comments": [],
        }

        self.log.info("Injecting run control document")
        self.run_collection.insert(self.query)

        # Used for computing offsets so reader starts from zero time
        self.starttime = None

        self.occurences = []

    def get_connection(self, hostname):
        if hostname not in self.connections:
            try:
                self.connections[hostname] = pymongo.Connection(hostname)

            except pymongo.errors.ConnectionFailure:
                self.log.fatal("Cannot connect to mongo at %s" % hostname)

        return self.connections[hostname]

    def shutdown(self):
        """Notify run database that datataking stopped
        """
        self.handle_occurences()  # write remaining data

        # Update runs DB
        self.query['reader']['stoptimestamp'] = str(datetime.now())
        self.query['reader']['data_taking_ended'] = True
        self.run_collection.save(self.query)

    @staticmethod
    def chunks(l, n):
        """ Yield successive n-sized chunks from l.
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def write_event(self, event):
        self.log.debug('Writing event')

        # We have to divide by the sample duration because the DAQ expects units
        # of 10 ns.  However, note that the division is done with a // operator.
        # This is an integer divide, thus gives an integer back.  If you do not
        # do this, you will store the time as a float, which will lead to
        # problems with the time precision (and weird errors in the DSP).  See
        # issue #35 for more info.
        time = event.start_time // event.sample_duration

        assert isinstance(time, int)

        if self.starttime is None:
            self.starttime = time
        elif time < self.starttime:
            error = "Found events before start of run"
            self.log.fatal(error)
            raise RuntimeError(error)

        for oc in event.occurrences:
            pmt_num = oc.channel
            sample_position = oc.left
            samples_occurrence = oc.raw_data

            assert isinstance(sample_position, int)

            occurence_doc = {}

            # occurence_doc['_id'] = uuid.uuid4()
            occurence_doc['module'] = pmt_num  # TODO: fix wax
            occurence_doc['channel'] = pmt_num

            occurence_doc['time'] = time + sample_position - self.starttime

            data = np.array(samples_occurrence, dtype=np.int16).tostring()
            if self.query['reader']['compressed']:
                data = snappy.compress(data)

            # Convert raw samples into BSON format
            occurence_doc['data'] = Binary(data, 0)

            self.occurences.append(occurence_doc)

        if not self.collect_then_dump:
            self.handle_occurences()

    def handle_occurences(self):
        docs = self.occurences  # []
        # for occurences in list(self.chunks(self.occurences,
        # 1000)):

        # docs.append({'test' : 0,
        #                     'docs' : occurences})

        i = 0
        t0 = time.time()  # start time
        t1 = time.time()  # last time

        if self.repeater > 0:
            while (t1 - t0) < self.runtime:
                this_time = time.time()
                n = int((this_time - t1) * self.repeater)
                if n == 0:
                    continue

                t1 = this_time
                self.log.fatal('How many events to inject %d', n)

                modified_docs = []
                min_time = None
                max_time = None

                for _ in range(n):
                    i += 1
                    for doc in self.occurences:
                        # doc['_id'] = uuid.uuid4()
                        doc['time'] += i * (t1 - t0) / self.repeater

                        if min_time is None or doc['time'] < min_time:
                            min_time = doc['time']
                        if max_time is None or doc['time'] > max_time:
                            max_time = doc['time']

                        modified_docs.append(doc.copy())

                        if len(modified_docs) > 1000:
                            self.raw_collection.insert({self.start_time_key: min_time,
                                                        self.stop_time_key: max_time,
                                                        self.bulk_key: modified_docs},
                                                       w=0)
                            modified_docs = []
                            min_time = None
                            max_time = None

        elif len(docs) > 0:
            times = [doc['time'] for doc in docs]
            min_time = min(times)
            max_time = max(times)

            t0 = time.time()

            self.raw_collection.insert({self.start_time_key: min_time,
                                        self.stop_time_key: max_time,
                                        self.bulk_key: docs},
                                       w=0)

            t1 = time.time()

        self.occurences = []


class MongoDBInputTriggered(plugin.InputPlugin):

    """Read triggered data produced by kodiaq with MongoDB output
    """

    def startup(self):
        self.log.debug("Connecting to %s, database %s, collection %s" % (
            self.config['address'], self.config['database'], self.config['collection']))
        try:
            self.client = pymongo.MongoClient(self.config['address'])
            self.database = self.client[self.config['database']]
            self.collection = self.database[self.config['collection']]
        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise

        # This wont work once you put in an extra detector
        self.n_real_channels = len(self.config['channels_top'] + self.config['channels_bottom'])

        self.last_pulse_time = None
        self.current_event = None
        self.event_number = -1

        self.mongo_time_unit = int(self.config['mongo_time_unit'])

        # Continuous acquisition options
        self.continuous_acquisition = self.config.get('continuous_acquisition', False)
        self.sleep_seconds_if_too_few_events = self.config.get('sleep_seconds_if_too_few_events', 2)
        self.min_number_of_events = self.config.get('min_number_of_events', 30)

    def total_number_events(self):
        return self.number_of_events

    def get_events(self):
        # In continuous acquisition mode, do an infinite loop. Else, do once.
        while self.continuous_acquisition or self.last_pulse_time is None:

            # Ensure we can get sorted results:
            self.collection.ensure_index([("time", pymongo.ASCENDING)])

            # What query should we do?
            if self.last_pulse_time is not None:
                self.log.debug("Last pulse time %s, in mongo units %s" % (
                   self.last_pulse_time, self.last_pulse_time / self.mongo_time_unit))
                # We've processed pulses before, only get pulses we haven't seen yet now
                query = {"time": {"$gt": self.last_pulse_time / self.mongo_time_unit}}
            else:
                # Get all pulses in the collection
                query = {}

            # Get cursor to the desired pulses in ascending time order
            self.cursor = self.collection.find(query).sort('time', pymongo.ASCENDING)

            # Assuming each channel has data always (ie no ZLE), we know the number of events
            self.number_of_events = self.cursor.count() / self.n_real_channels

            if int(self.number_of_events) != self.number_of_events:
                raise RuntimeError(self.cursor.count(), self.n_real_channels)

            # If we're in continuous acquisition mode, and there's only a few events,
            # we can take a brief break to allow the DAQ to catch up.
            if self.continuous_acquisition:
                if self.number_of_events < self.min_number_of_events:
                    self.log.warning("Only %d events found: waiting %d sec for more data" % (
                        self.number_of_events, self.sleep_seconds_if_too_few_events))
                    time.sleep(self.sleep_seconds_if_too_few_events)
                    continue

            elif self.number_of_events == 0:
                raise RuntimeError("No events found in this collection!")

            self.ts = time.time()       # Start clock for timing report
            for pulse_doc in self.cursor:

                pulse_time = pulse_doc['time'] * self.mongo_time_unit
                self.log.debug("Pulse time %s, in mongo units %s" % (
                   pulse_time, pulse_time / self.mongo_time_unit))
                pulse_data = snappy.decompress(pulse_doc['data'])
                pulse_data = np.fromstring(pulse_data, dtype="<i2")
                channel = pulse_doc['channel']

                if pulse_time != self.last_pulse_time:
                    # Yield current event, if there is one
                    if self.current_event is not None:

                        a = len(self.current_event.occurrences)
                        if a != self.n_real_channels:
                            raise RuntimeError("Event %d has %d occurrences, should be %d!"
                                               "This is assuming all channels always report data, "
                                               "if it's ok if they don't, just comment this error" % (
                                                   self.current_event.event_number, a, self.n_real_channels))

                        self.total_time_taken += (time.time() - self.ts) * 1000      # Store elapsed time
                        yield self.current_event
                        self.ts = time.time()       # Restart clock for timing report

                    # Prepare new event
                    self.event_number += 1
                    self.current_event = Event(n_channels=self.config['n_channels'],
                                               start_time=pulse_time,
                                               sample_duration=self.config['sample_duration'],
                                               event_number=self.event_number,
                                               stop_time=pulse_time + len(pulse_data) * self.config['sample_duration'])

                    self.log.debug("Started new event at %s, end at %s" % (
                        self.current_event.start_time,
                        self.current_event.stop_time))

                self.log.debug("Adding pulse at %d from channel %d" % (pulse_time, channel))

                # Add another pulse to existing event
                self.current_event.occurrences.append(Occurrence(left=0,
                                                                 raw_data=pulse_data,
                                                                 channel=channel))

                self.last_pulse_time = pulse_time

    def shutdown(self):
        del self.cursor     # Maybe unnecessary? Test!
