"""Live Data Monitoring

The DAQ uses a Django web interface for control. This introduces the possibility to easily view data online as it
comes out of the detector. This plugin is meant to be run in a standalone pax instance constantly polling the
Event Builder output database and processing as many events as it can get to. Each reconstructed event is saved
in its entirety to a capped collection on the data monitor's MongoDB database. Additionally, long-term diagnostic
spectra are filled on a per-run basis.

"""


import pymongo
from pax import plugin
import numpy
import operator
from bson.json_util import loads
from bson.objectid import ObjectId
from pax.plugins.plotting.Plotting import PlotEventSummary
import matplotlib.pyplot as plt
import pickle
import time
from pax import units

def get_bin(bin_value, num_bins, min_bin, max_bin):

    """ Gets which bin an entry falls into. Overflow bin is MAX, underflow is 0.
    """

    bin_size = (max_bin - min_bin) / num_bins
    bin_number = (bin_value - min_bin) / bin_size

    if bin_number > num_bins - 1:
        bin_number = num_bins - 1
    if bin_number < 0:
        bin_number = 0

    return int(bin_number)


class OnlineMonitorOutput(plugin.OutputPlugin):

    """
    Connectivity properties should be defined in the .ini file. Needs to output to a MongoDB with two collections.
    """

    def startup(self):
        self.log.debug("Connecting to %s" % self.config['address'])
        try:
            self.client = pymongo.MongoClient(self.config['address'])
            self.database = self.client[self.config['database']]
            self.event_collection = self.database[self.config['event_collection']]
            self.aggregate_collection = self.database[self.config['aggregate_collection']]
            self.plot_collection = self.database[self.config['plot_collection']]

            # Configure collections based on .ini file properties

        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise

        # This is a list of dicts. Each dict is a histogram.
        self.aggregates = self.config['aggregates']

    def write_event(self, event):

        #self.write_complete_event(event)
        self.update_aggregate_docs(event)
        self.write_plot_collection(event)

    def write_complete_event(self, event):

        """ Dump entire event to BSON and save in mongodb
        """

        # Some events will be too large. Waveforms are mostly zeros. We can compress them by removing zeros.
        smaller = self.compress_event(loads(event.to_json()))
        try:
            self.event_collection.insert(smaller)
        except Exception as e:
            self.log.warn("Error inserting a waveform doc. Likely that it is too large (16MB limit)")
            self.log.exception(e)

        return

    def write_plot_collection(self, event):

        # This is easiest if we instantiate a plot object (we want to make the display and pickle it)

        # We need a 'fake' config file
        fake_config = { 'output_dir': None,
                        'size_multiplier': 3.5,
                        #"horizontal_size_multiplier": 1,
                        'plot_largest_peaks': True,
                        'log_scale_entire_event': False,
                        'log_scale_s2': False,
                        'log_scale_s1': False,'waveforms_to_plot': (
                            {'internal_name': 'tpc',      'plot_label': 'TPC (hits only)',
                                'drawstyle': 'steps', 'color':'black'},
                            {'internal_name': 'tpc_raw',  'plot_label': 'TPC (raw)',
                                'drawstyle': 'steps', 'color':'black', 'alpha': 0.3},
                            {'internal_name': 'veto',     'plot_label': 'Veto (hits only)',
                                'drawstyle': 'steps', 'color':'red'},
                            {'internal_name': 'veto_raw', 'plot_label': 'Veto (raw)',
                                'drawstyle': 'steps', 'color':'red', 'alpha': 0.2})}
        config_total = self.config.copy()
        config_total.update(fake_config)
        plotter = PlotEventSummary(processor=self.processor, config_values=config_total)
        plotter.plot_event(event=event)
        plot = plt.gcf()
        # save as pickle in doc
        binary = pickle.dumps(plot)
        trigger_time_ns = (event.start_time + self.config.get('trigger_time_in_event', 0)) / units.ns
        print(trigger_time_ns)
        timestring = time.strftime("%Y/%m/%d, %H:%M:%S", time.gmtime(trigger_time_ns / 10 ** 9))
        print(timestring)

        try:
            self.plot_collection.insert({'data': binary, 'run_name': event.dataset_name,
                                         'event_date': timestring,
                                         'event_number': event.event_number })
        except:
            self.log.warn("BSON document for event display too big!")
        return

    def update_aggregate_docs(self, event):

        """ Take list of aggregates from config file and update histogram docs
        Do this on a run-by-run basis
        """

        for histogram_definition in self.aggregates:

            # Take a look if a doc for this plot exists. If not make one.
            query = {"name": histogram_definition['name'], "run": event.dataset_name}
            if self.aggregate_collection.find_one(query) is None:

                # The doc is just the definition plus the run name plus and array of zeroes for the bins
                insert_doc = histogram_definition
                insert_doc['run'] = event.dataset_name

                # Replace the axis definition with actual numbers in case they are expressions
                insert_doc['xaxis']['bins'] = eval(str(insert_doc['xaxis']['bins']))
                insert_doc['xaxis']['min'] = eval(str(insert_doc['xaxis']['min']))
                insert_doc['xaxis']['max'] = eval(str(insert_doc['xaxis']['max']))

                # Types are 'h1', 'h2', and 'scatter'
                if insert_doc['type'] == 'h1':
                    insert_doc['data'] = numpy.zeros(insert_doc['xaxis']['bins'],
                                                     dtype=int).tolist()

                else:

                    # Evaluate axis labels
                    insert_doc['yaxis']['bins'] = eval(str(insert_doc['yaxis']['bins']))
                    insert_doc['yaxis']['min'] = eval(str(insert_doc['yaxis']['min']))
                    insert_doc['yaxis']['max'] = eval(str(insert_doc['yaxis']['max']))

                    if insert_doc['type'] == 'h2':
                        insert_doc['data'] = numpy.zeros((insert_doc['xaxis']['bins'],
                                                          insert_doc['yaxis']['bins']), dtype=int).tolist()
                    if insert_doc['type'] == 'scatter':
                        insert_doc['data'] = dict(x=[], y=[])

                self.aggregate_collection.insert(insert_doc)

            # Now we can query the DB and pull the doc. Since we just inserted a doc in case one didn't exist this
            # should always work
            doc = self.aggregate_collection.find_one(query)
            if doc is None:
                raise RuntimeError("Cannot find MongoDB doc corresponding to requested aggregate plot.")

            entries = self.get_entry_list(event, doc)

            # For 1D histograms get the bin value and increment, then continue. No need to look at y-values.
            if doc['type'] == 'h1':
                for entry in entries:
                    binx = get_bin(entry, doc['xaxis']['bins'], doc['xaxis']['min'], doc['xaxis']['max'])
                    self.aggregate_collection.update({"_id": ObjectId(doc["_id"])},
                                                     {"$inc": {"data." + str(binx): 1}})
                continue

            # For scatters simply append the point to a list
            if doc['type'] == 'scatter':
                for val in entries['x']:
                    if 'suppress_zero' in doc['xaxis'].keys() and doc['xaxis']['suppress_zero'] is True and val == 0:
                        continue
                    if type(val).__module__ == numpy.__name__:
                        val = val.item()
                    try:
                        self.aggregate_collection.update({"_id": doc['_id']}, {"$push": {"data.x": val}})
                    except:
                        break
                for val in entries['y']:
                    if 'suppress_zero' in doc['yaxis'].keys() and doc['yaxis']['suppress_zero'] is True and val == 0:
                        continue
                    if type(val).__module__ == numpy.__name__:
                        val = val.item()
                    try:
                        self.aggregate_collection.update({"_id": doc['_id']}, {"$push": {"data.y": val}})
                    except:
                        break


                continue

            # For 2D histos do a slightly more complicated thing as was done for 1D
            if doc['type'] == 'h2':
                for entry in entries:
                    binx = get_bin(entry[0], doc['xaxis']['bins'], doc['xaxis']['min'], doc['xaxis']['max'])
                    biny = get_bin(entry[1], doc['yaxis']['bins'], doc['yaxis']['min'], doc['yaxis']['max'])
                    self.aggregate_collection.update({"_id": doc['_id']},
                                                     {"$inc": {"data." + str(binx) + "." + str(biny): 1}})
                continue
        return

    def get_entry_list(self, event, doc):

        """ Returns a list of entries for direct insertion
        """

        # Get x data. For 1D hist just return this
        x = self.get_data(doc['xaxis']['type'], doc['xaxis']['value'], event)
        if doc['type'] == 'h1':
            return x

        # For 2D things get y data too
        y = self.get_data(doc['yaxis']['type'], doc['yaxis']['value'], event)

        # Both should be the same length, otherwise we fail
        if len(x) != len(y):
            raise RuntimeError("x and y variable for online monitor plot must have same number of entries")
        if doc['type'] == 'h2':
            return zip(x, y)
        return dict(x=x, y=y)

    def get_data(self, value_type, value, event):

        """ Helps out by getting the data from the event object. Returns as an array.
        """

        # Attribute types are gotten directly (doesn't handle arrays)
        if value_type == 'attr':
            getter = operator.attrgetter(value)

            try:
                vals = getter(event)
            except:
                self.log.error("Invalid variable: " + value + " given in aggregate plot config.")
                raise

        # If it's more complicated we use an expression
        elif value_type == 'expr':

            try:
                vals = eval(value)
            except:
                self.log.warn("Invalid expression given for eval")
                return []
        else:
            raise RuntimeError("Unknown variable type " + value_type + " provided in aggregate plot config")

        # This function always returns an array but can handle expressions giving either arrays or values
        arr = []
        if type(vals) in [list, tuple, numpy.ndarray, dict]:
            arr = vals
        else:
            arr.append(vals)
        return arr

    def compress_event(self, event):

        """ Compresses an event by suppressing zeros in waveform in a way the frontend will understand
            Format is the char 'zn' where 'z' is a char and 'n' is the number of following zero bins
        """

        for x in range(0, len(event['sum_waveforms'])):

            waveform = event['sum_waveforms'][x]['samples']
            zeros = 0
            ret = []

            for i in range(0, len(waveform)):
                if waveform[i] == 0:
                    zeros += 1
                    continue
                else:
                    if zeros != 0:
                        ret.append('z')
                        ret.append(str(zeros))
                        zeros = 0
                    ret.append(str(waveform[i]))
            if zeros != 0:
                ret.append('z')
                ret.append(str(zeros))
            event['sum_waveforms'][x]['samples'] = ret

        # Unfortunately we also have to remove the pulses or some events are huge
        del event['pulses']
        return event
