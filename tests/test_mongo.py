import unittest
from pax import core
from pax.plugins.io.Mongo import EVENT_QUERY


class TestMongo(unittest.TestCase):

    def test_mongo(self):
        # Write an event to a mocked Mongo
        mypax = core.Processor(config_names='XENON100',
                               config_dict={'pax': {'output': ['Mongo.WriteMongo', 'Dummy.DummyOutput'],
                                                    'events_to_process': [0],
                                                    'encoder_plugin': None,
                                                    'output_name': 'test_mongo_output'},
                                            'Mongo': {'mongomock': True,
                                                      'mode': 'collection_per_run',
                                                      'database_name': 'pax_events'}})
        mypax.run()
        client = mypax.get_plugin_by_name('WriteMongo').client
        pax_event = mypax.get_plugin_by_name('DummyOutput').last_event
        del mypax

        self.assertIsNotNone(client)
        db = client['pax_events']
        c = db['test_mongo_output']
        self.assertEqual(c.count(), 2)
        self.assertEqual(c.find({'metadata': True}).count(), 1)
        self.assertEqual(c.find(EVENT_QUERY).count(), 1)

        mongo_event = c.find_one(EVENT_QUERY)
        self.assertIsNotNone(mongo_event)
        self.assertEqual(len(mongo_event['peaks']), len(pax_event.peaks))

        # Make a processor to read from mongo
        mypax = core.Processor(config_names='XENON100',
                               config_dict={'pax': {'input': 'Mongo.ReadMongo',
                                                    'output': 'Dummy.DummyOutput',
                                                    'encoder_plugin': None,
                                                    'decoder_plugin': None,
                                                    'pre_dsp': [],
                                                    'dsp': [],
                                                    'compute_properties': [],
                                                    'pre_analysis': [],
                                                    'pre_output': [],
                                                    'events_to_process': [0],
                                                    'input_name': 'test_mongo_output'},
                                            'Mongo': {'mongomock': True,
                                                      'mode': 'collection_per_run',
                                                      'database_name': 'pax_events'}})

        # Inject the event into the mongomock
        mypax.input_plugin.collection.insert_one(mongo_event)

        mypax.run()
        pax_event_2 = mypax.get_plugin_by_name('DummyOutput').last_event
        del mypax

        self.assertEqual(len(pax_event.peaks), len(pax_event_2.peaks))


if __name__ == '__main__':
    unittest.main()
