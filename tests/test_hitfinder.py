import unittest
import numpy as np

from pax import core, plugin, units
from pax.datastructure import Event, Occurrence

# 100 normal(16000,10)'s
example_noise = np.array([16010, 16001, 16000, 15992, 16025, 15992, 16007, 15988, 15997,
                          16014, 15977, 15993, 15987, 15987, 15988, 16003, 15998, 16003,
                          16004, 15987, 16003, 16007, 15992, 16012, 16006, 16020, 15976,
                          16006, 16020, 15988, 15999, 16006, 16002, 15999, 16008, 16004,
                          16015, 15993, 16000, 15983, 15989, 15991, 15997, 16009, 16000,
                          15998, 16018, 15997, 16002, 15998, 16008, 16000, 16008, 15993,
                          15990, 15993, 15998, 16006, 16000, 16004, 15982, 15999, 16000,
                          16005, 15993, 16008, 16020, 16003, 15986, 15988, 16009, 16014,
                          16020, 16004, 16000, 15998, 16001, 16007, 16013, 15996, 15997,
                          16018, 15995, 16003, 16009, 15996, 16013, 16010, 15988, 16000,
                          16002, 16005, 16009, 15998, 16008, 15999, 15988, 16012, 16005, 16002], dtype=np.int16)


class TestHitFinder(unittest.TestCase):
    def setUp(self):
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={
            'pax': {
                'plugin_group_names': ['test'],
                'test': 'HitFinder.FindHits',
                'logging_level':      'DEBUG'},
            'HitFinder.FindHits': {
                'bound_threshold': 3,    # Noise sigmas -- higher than normal, to be predictable
            }})
        self.plugin = self.pax.get_plugin_by_name('FindHits')
        self.event_number = 0

    @staticmethod
    def peak_at(left, right, amplitude):
        w = example_noise.copy()
        w[left:right + 1] = amplitude
        return w

    def test_instantiate_hitfinder(self):
        self.assertIsInstance(self.plugin, plugin.TransformPlugin)
        self.assertEqual(self.plugin.__class__.__name__, 'FindHits')

    def try_single_clear_peak(self, left, right):
        w = self.peak_at(left, right, amplitude=100)

        self.event_number += 1
        dt = int(10 * units.ns)
        e = Event(start_time=0,
                  n_channels=self.pax.config['DEFAULT']['n_channels'],
                  stop_time=len(w) * dt,
                  event_number=self.event_number,
                  sample_duration=dt)
        e.occurrences.append(Occurrence(left=0,
                                        right=len(w) - 1,
                                        channel=1,
                                        raw_data=w))
        e = self.plugin.transform_event(e)
        self.assertEqual(len(e.all_channel_peaks), 1)
        hit = e.all_channel_peaks[0]
        self.assertEqual(hit.left, left)
        self.assertEqual(hit.right, right)

    def test_single_peaks(self):
        # 10 samples wide
        self.try_single_clear_peak(50, 60)
        self.try_single_clear_peak(10, 20)
        self.try_single_clear_peak(0, 5)        # 0, 20 fails due to baselining stuff
        self.try_single_clear_peak(80, 99)

        # 2 samples wide
        self.try_single_clear_peak(5, 6)
        self.try_single_clear_peak(0, 1)
        self.try_single_clear_peak(98, 99)

        # 1 sample wide
        self.try_single_clear_peak(5, 5)
        self.try_single_clear_peak(0, 0)
        self.try_single_clear_peak(99, 99)


if __name__ == '__main__':
    unittest.main()
