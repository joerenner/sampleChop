# testing classify API

import unittest
from sampleChop.sampleChop import classify
import numpy as np
from math import ceil
from sklearn.externals import joblib
import librosa as lb

class TestSampleInit(unittest.TestCase):

    # tests initial cqt generation options
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


class TestClassification(unittest.TestCase):

    def setUp(self):
        self.sample = classify.Sample(file_name="../../Hear_Me.wav", duration=10.0)

    # testing data generator function
    def test_getData(self):
        datagen = list(classify.getData(self.sample.cqt))
        self.assertEqual(len(datagen),1663)    # list of tuples is sufficiently long (for specific test song)
        self.assertGreater(datagen[1][0], datagen[0][0])            # indicies are sequential
        for i in xrange(len(datagen)):
            self.assertTupleEqual(np.shape(datagen[i][1]), (61,))   # shape of datapoint is correct
            self.assertFalse(np.isnan(datagen[i][1]).any())         # no nans in data point
            if i != 0:
                self.assertGreater(datagen[i][0], datagen[i - 1][0])    # indicies are sequential

    # testing time classifications and cleaning
    def test_classification(self):
        self.sample.classify()
        total_frames = ceil(10.0 * self.sample.sr / self.sample.hp_len)
        for i in xrange(len(self.sample.final_frames)):
            self.assertLess(self.sample.final_frames[i], total_frames)
            if i > 0:
                self.assertGreater(self.sample.final_frames[i],self.sample.final_frames[i-1])


if __name__ == '__main__':
    # initialize the test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # add tests to the test suite
    suite.addTests(loader.loadTestsFromTestCase(TestSampleInit))
    suite.addTests(loader.loadTestsFromTestCase(TestClassification))
    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)

