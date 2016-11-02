import unittest
from pax import core


class TestMongoOutput(unittest.TestCase):

    def test_mongo_output(self):
        # Write an event to Mongo
        mypax = core.Processor(config_names='XENON100',
                               config_dict={'pax': {'output': ['Mongo.WriteMongo', 'Dummy.DummyOutput'],
                                                    'events_to_process': [0],
                                                    'encoder_plugin': None,
                                                    'output_name': 'test_mongo_output'},
                                            'Mongo': {'mongomock': True}})
        mypax.run()
        client = mypax.get_plugin_by_name('WriteMongo').client
        pax_event = mypax.get_plugin_by_name('DummyOutput').last_event
        del mypax

        self.assertIsNotNone(client)
        db = client['test_mongo_output']
        self.assertEqual(db['events'].count(), 1)
        # Metadata currently not stored
        # self.assertEqual(db['metadata'].count(), 1)

        # Todo: compare event properties
        mongo_event = db['events'].find_one({})
        self.assertEqual(len(mongo_event['peaks']), len(pax_event.peaks))

if __name__ == '__main__':
    unittest.main()
