from pax import plugin
from pax.MongoDB_ClientMaker import ClientMaker


class WriteMongo(plugin.OutputPlugin):

    def startup(self):
        self.client = ClientMaker(self.processor.config['Mongo']).get_client(self.config['input_name'])
        self.collection = self.client.get_collection('events')
        self.client.get_collection('metadata').insert_one(self.processor.get_metadata())

    def write_event(self, event):
        # Normalize record array classes
        self.collection.insert(event.to_dict(convert_numpy_arrays_to='bytes'))

    def shutdown(self):
        self.client.close()

#
# def WriteMongo(plugin.OutputPlugin):
#
#     def