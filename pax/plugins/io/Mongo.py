import unittest

from pax import plugin
from pax.MongoDB_ClientMaker import ClientMaker


class WriteMongo(plugin.OutputPlugin):

    def startup(self):
        on = self.config['output_name']
        self.client = ClientMaker(self.processor.config.get('Mongo')).get_client(on)
        self.db = self.client[on]
        self.collection = self.db.get_collection('events')
        # Metadata broken due to dots in section names, have to replace with |
        # self.db.get_collection('metadata').insert_one(self.processor.get_metadata())

    def write_event(self, event):
        event._normalize_record_arrays()
        self.collection.insert(event.to_dict(convert_numpy_arrays_to='bytes'))

    def shutdown(self):
        self.client.close()


if __name__ == '__main__':
    unittest.main()
