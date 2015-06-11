import ROOT
import unittest
import tempfile

from pax import core


class TestROOTClass(unittest.TestCase):

    def test_write_read(self):
        # If this test errors in a strange way, the directory may not get deleted.
        # So make it somewhere the os knows to delete it sometime
        #tempdir = tempfile.TemporaryDirectory()
        file_name = tempfile.NamedTemporaryFile()

        # This print statement is necessary to help user figure out which plugin failed, if any does fail.

        config = {'pax': {'stop_after': 10,
                          'input': 'XED.XedInput',
                          'output': 'ROOT.ROOTClass'},
                 'ROOT.ROOTClass': {'output_name': file_name.name }}

        p = core.Processor(config_names='XENON100',
                                         config_dict=config)

        # Wrap this in a try-except, to ensure the read plugin shutdown is run BEFORE the tempdir shutdown
        try:
            p.run()
        except Exception as e:
            p.shutdown()
            raise e

        # Cleaning up the temporary dir explicitly (otherwise tempfile gives warning):
        #tempdir.cleanup()

if __name__ == '__main__':
    unittest.main()
