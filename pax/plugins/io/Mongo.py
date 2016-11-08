from pax import plugin, configuration, datastructure
from pax.MongoDB_ClientMaker import ClientMaker

# Query to ask for events (not metadata)
EVENT_QUERY = {'event_number': {'$exists': True}}


class MongoPlugin:

    def connect(self, run_name):
        mode = self.config.get('mode', 'collection_per_run')

        database_name = self.config.get('database_name', 'pax_events')
        collection_name = self.config.get('collection_name', 'pax_events')

        if mode == 'database_per_run':
            database_name = run_name
        elif mode == 'collection_per_run':
            collection_name = run_name

        self.client = ClientMaker(self.processor.config.get('Mongo')).get_client(database_name)
        db = self.client[database_name]
        self.collection = db.get_collection(collection_name)

    def shutdown(self):
        self.client.close()


class WriteMongo(plugin.OutputPlugin, MongoPlugin):

    def startup(self):
        self.connect(self.config['output_name'])

        # Insert the metadata into the same collection as the events
        metadata = self.processor.get_metadata()
        metadata = configuration.fix_sections_for_mongo(metadata)
        metadata['metadata'] = True
        self.collection.insert_one(metadata)

    def write_event(self, event):
        event.normalize_record_arrays()
        self.collection.insert_one(event.to_dict(convert_numpy_arrays_to='bytes'))

    def shutdown(self):
        self.client.close()


class ReadMongo(plugin.InputPlugin, MongoPlugin):

    def startup(self):
        self.connect(self.config['input_name'])
        self.number_of_events = self.collection.count() - 1    # -1 for metadata

    def get_events(self):
        for doc in self.collection.find(EVENT_QUERY):
            yield doc_to_event(doc)

    def get_single_event(self, event_number):
        doc = self.collection.find_one({'event_number': event_number})
        if doc is None:
            raise ValueError("No event numbered %d in the dataset!" % event_number)
        return doc_to_event(doc)


def doc_to_event(doc):
    return datastructure.Event(fields_to_ignore=['_id'], **doc)
