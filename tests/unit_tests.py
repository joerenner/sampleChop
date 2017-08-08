# testing classify API

import unittest
from sampleChop.sampleChop import classify
import numpy as np
from math import ceil

class TestSampleClass(unittest.TestCase):

    def test_cqt(self):
        # regular transform
        test_song = classify.Sample(file_name="../../Hear_Me.wav")
        normal_shape = np.shape(test_song.cqt)  # for use in assertions below
        self.assertEquals(np.shape(test_song.cqt)[0],48)
        # harmonic only transform
        test_song = classify.Sample(file_name="../../Hear_Me.wav", harmonic=True)
        self.assertTupleEqual(np.shape(test_song.cqt), normal_shape)
        # different start note
        test_song = classify.Sample(file_name="../../Hear_Me.wav", start_note='F4')
        self.assertTupleEqual(np.shape(test_song.cqt), normal_shape)
        # certain duration ( < than duration of whole sample)
        test_song = classify.Sample(file_name="../../Hear_Me.wav", duration=10.0)
        self.assertTupleEqual(np.shape(test_song.cqt), (48, ceil(10.0 * test_song.sr / test_song.hp_len)))
        # certain duration ( > than duration of sample)
        test_song = classify.Sample(file_name="../../Hear_Me.wav", duration=40.0)
        self.assertTupleEqual(np.shape(test_song.cqt), normal_shape)

if __name__ == '__main__':
    # initialize the test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # add tests to the test suite
    suite.addTests(loader.loadTestsFromTestCase(TestSampleClass))
    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)

