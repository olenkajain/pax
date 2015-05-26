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


def get_bin(bin_value, num_bins, min_bin, max_bin):

    """Gets which bin the thing falls in. Overflow bin is MAX, underflow is 0.
    """

    bin_size = (max_bin - min_bin) / num_bins
    bin_number = (bin_value - min_bin) / bin_size

    if bin_number > num_bins - 1:
        bin_number = num_bins - 1
    if bin_number < 0:
        bin_number = 0

    yield bin_number


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

            # Configure collections based on .ini file properties

        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise

        # See config file. Has form: { "name": string, "type": string, "xaxis:" { "name", "var", "min", "max", "bins" },
        # "yaxis" : {"name", if 2d also var, min, max, bins }, "data": [ either x1, x2, ... or [x1,y1],[x2,y2], ...]}
        self.aggregates = self.config['aggregates']

    def write_event(self, event):

        self.write_complete_event(event)
        self.update_aggregate_docs(event)

    def write_complete_event(self, event):

        """ Dump entire event to BSON and save in mongodb
        """

        self.event_collection.insert(event)
        return

    def update_aggregate_docs(self, event):

        """ Take list of aggregates from config file and update histogram docs
        Do this on a run-by-run basis
        """

        for histogram_definition in self.aggregates:

            # Take a look if a doc for this plot exists. If not make one.
            query = {"name": histogram_definition['name'], "run": histogram_definition['run']}
            if self.aggregate_collection.find_one(query) is None:

                # The doc is just the definition plus the run name plus and array of zeroes for the bins
                insert_doc = histogram_definition
                insert_doc['run'] = event.dataset_name

                # Types are 'h1', 'h2', and 'scatter'
                if insert_doc['type'][0] == 'h1':
                    insert_doc['data'] = [0]*insert_doc['xaxis']['bins']
                if insert_doc['type'] == 'h2':
                    insert_doc['data'] = numpy.zeros((insert_doc['xaxis']['bins'], insert_doc['yaxis']['bins']))
                if insert_doc['type'] == 'scatter':
                    insert_doc['data'] = []

                self.aggregate_collection.insert(insert_doc)

            # Now we can query the DB and pull the doc. Since we just inserted a doc in case one didn't exist this
            # should always work
            doc = self.aggregate_collection.find_one(query)
            if doc is None:
                raise RuntimeError("Cannot find MongoDB doc corresponding to requested aggregate plot.")

            # The actual logic for the insertion varies depending on if we have a 1d hist, 2d hist, or scatter plot
            xgetter = operator.attrgetter(doc['xaxis']['var'])
            try:
                x_value = xgetter(event)
            except:
                self.log.fatal("Invalid x variable: " + doc['xaxis']['var'] + " given in aggregate plot options.")
                raise

            # For 1D histograms get the bin value and increment, then continue. No need to look at y-values.
            if doc['type'] == 'h1':
                binx = get_bin(x_value, doc['xaxis']['nbins'], doc['xaxis']['start'], doc['xaxis']['end'])
                self.aggregate_collection.update({doc: '_id'}, {"$inc": {"data." + str(binx): 1}})
                continue

            # 2D histos and scatter plots need y value
            ygetter = operator.attrgetter(doc['yaxis']['var'])
            try:
                y_value = ygetter(event)
            except:
                self.log.fatal("Invalid y variable: " + doc['yaxis']['var'] + " given in aggregate plot options.")
                raise

            # For scatters simply append the point to a list
            if doc['type'] == 'scatter':
                self.aggregate_collection.update({doc: '_id'}, {"$push": {"data", [x_value, y_value]}})
                continue

            # For 2D histos do a slightly more complicated thing as was done for 1D
            if doc['type'] == 'h2':
                binx = get_bin(x_value, doc['xaxis']['nbins'], doc['xaxis']['start'], doc['xaxis']['end'])
                biny = get_bin(y_value, doc['yaxis']['nbins'], doc['yaxis']['start'], doc['yaxis']['end'])
                self.aggregate_collection.update({doc: '_id'}, {"$inc": {"data." + str(binx) + "." + str(biny): 1}})
                continue

        return
