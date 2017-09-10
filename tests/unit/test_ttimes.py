'''
Created on Sep 4, 2017

@author: riccardo
'''
import numpy as np
from os.path import dirname, join, isfile
import unittest
from stream2segment.download.traveltimes.ttloader import TTTable
# from stream2segment.download.utils import get_min_travel_time
from click.testing import CliRunner
import os
from stream2segment.download.traveltimes.ttcreator import _filepath, StepIterator, min_traveltimes,\
    min_traveltime


class Test(unittest.TestCase):

    def setUp(self):
        
        

        self.iasp_ttp_10 = TTTable(join(dirname(dirname(__file__)), "data", "iasp91_ttp+_10.npz"))
        self.iasp_ttp_5 = TTTable(join(dirname(dirname(__file__)), "data", "iasp91_ttp+_5.npz"))
        self.iasp_tts_10 = TTTable(join(dirname(dirname(__file__)), "data", "iasp91_tts+_10.npz"))
        self.iasp_tts_5 = TTTable(join(dirname(dirname(__file__)), "data", "iasp91_tts+_5.npz"))
        self.ak135_ttp_10 = TTTable(join(dirname(dirname(__file__)), "data", "ak135_ttp+_10.npz"))
        self.ak135_ttp_5 = TTTable(join(dirname(dirname(__file__)), "data", "ak135_ttp+_5.npz"))
        self.ak135_tts_10 = TTTable(join(dirname(dirname(__file__)), "data", "ak135_tts+_10.npz"))
        self.ak135_tts_5 = TTTable(join(dirname(dirname(__file__)), "data", "ak135_tts+_5.npz"))
        
        self.ttables = [getattr(self, n) for n in dir(self) if isinstance(getattr(self, n), TTTable)]
        
#         self.ttable_03sec = TTTable(join(dirname(dirname(__file__)), "data", "iasp91_errtol=03sec.npz"))
# 
        self._values = np.array([(0, 0, 0), (1, 0, 0), (5, 0, 0), (700,0,0),
                                 (0, 0, 11.4), (1, 0, 11.4), (5, 0, 11.4), (700, 0, 11.4),
                                 (0, 0, 21.9), (1, 0, 21.9), (5, 0, 21.9), (700, 0, 21.9),
                                 (0, 0, 178.9), (1, 0, 178.9), (5, 0, 178.9), (700, 0, 178.9),
                                 (0, 0, 361.9), (1, 0, 361.9), (5, 0, 361.9), (700, 0, 361.9),
                           ])
    def tearDown(self):
        pass

    # this test passed, it takes too long, so skip it for the moment
    def tst_ttcreator(self):
        from stream2segment.download.traveltimes import ttcreator
        runner = CliRunner()
        with runner.isolated_filesystem():
            mydir = os.getcwd()
            result = runner.invoke(ttcreator.run, catch_exceptions=True)
            assert result.exit_code != 0

            result = runner.invoke(ttcreator.run, ['-o', mydir, '-m', 'iasp91', '-t', 30, '-p',
                                                   'ttp+', '-s', 111, '-r', 2, '-d', 113],
                                   catch_exceptions=False)
            assert result.exit_code == 0
            assert os.path.isfile(_filepath(mydir, 'iasp91', ['ttp+']) + ".npz")
            # fixme: we should load the file and assert something...

            # test with no receiver depths (set to 0)
            result = runner.invoke(ttcreator.run, ['-o', mydir, '-m', 'ak135', '-t', 10, '-p',
                                                   'ttp+', '-s', 111, '-r', 0 , '-d', 89.45],
                                   catch_exceptions=False)
            assert result.exit_code == 0
            assert os.path.isfile(_filepath(mydir, 'iasp91', ['ttp+']) + ".npz")
            # fixme: we should load the file and assert something...

    def test_stepiterator(self):
        '''test a step iterator which should give me approximately every 100's'''
        lastnum = -1
        results = []
        stepiterator = StepIterator(0, 700.0, 31.5)
        for val in stepiterator:
            if int(val / 100.0) > lastnum:  # condition whereby we crossed the 'mark' 
                if stepiterator.moveback():
                    continue
                else:
                    results.append(val)
                    lastnum += 1
        assert len(results) == 8

        # now try an edge case with a single value
        results = []
        stepiterator = StepIterator(0, 0, 31.5)
        for val in stepiterator:
            results.append(val)
        assert results == [0]


    # these tests are long: execute each of them as a separate method so that the progress "dot"
    # in the terminal while using pytest makes things les boring and more informative
    def test_ttable_1(self):
        self.tst_tt(self.iasp_ttp_5)
    def test_ttable_2(self):
        self.tst_tt(self.iasp_tts_5)
    def test_ttable_3(self):
        self.tst_tt(self.ak135_ttp_5)
    def test_ttable_4(self):
        self.tst_tt(self.ak135_tts_5)
        
    def tst_tt(self, ttable):
        
        # create a point where we expect to be the maximum error: in the middle of the
        # first 4 points (0,0), (0, half_hstep),
        # (half_vstep, 0) and (half_vstep, half_hstep)
        # get the half step (max distance along x axis = columns)
        half_hstep = ttable._distances[1]/2.0
        # get the half step (max distance along y axis = row)
        half_vstep = ttable._sourcedepths[0] / 2.0
        # each point is (source_depth_km, receiver_depth_km, distance_deg):
        values = np.vstack(([half_vstep, 0, half_hstep], self._values))
        results_c = ttable.min(values[:, 0], values[:, 1], values[:, 2], method='cubic')

        real_results = []
        for v in values:
            real_results.append(min_traveltime(str(ttable._modelname),
                                               v[0], v[1], v[2], ttable._phases.tolist()))

        assert np.allclose(results_c, real_results, rtol=0, atol=ttable._tt_errtol,
                           equal_nan=True)

        results_l = ttable.min(values[:, 0], values[:, 1], values[:, 2], method='linear')
        results_n = ttable.min(values[:, 0], values[:, 1], values[:, 2], method='nearest')

        # for some tts+ models, the linear case might lead to median that are
        # better than the cubic case

        err_c = np.abs(results_c-real_results)
        err_l = np.abs(results_l-real_results)
        err_n = np.abs(results_n-real_results)

        # for cubic vs nearest, we can simply assert this:
        assert np.nanmean(err_c) < np.nanmean(err_n)
        assert np.nanmedian(err_c) < np.nanmedian(err_n)
        assert np.nanmax(err_c) < np.nanmean(err_n)
        assert np.nanmean(err_l) < np.nanmean(err_n)
        assert np.nanmedian(err_l) < np.nanmedian(err_n)
        assert np.nanmax(err_l) < np.nanmedian(err_n)

        # on the other hand, sometimes (tts+ models) the mean linear is better than the cubic one
        # The reason is partly because there might be a bias on the (low) number of points
        # and also because it seems that linear outperforms cubic for the tts+ case
        # so do not do this:
#             assert np.nanmean(err_c) < np.nanmean(err_l)
#             assert np.nanmedian(err_c) < np.nanmedian(err_l)
#             assert np.nanmax(err_c) < np.nanmax(err_l)

    def test_edge_cases(self):
        for ttable in self.ttables:
            for method in ['linear', 'cubic', 'nearest']:
                # test scalar case. Assert is stupid is just to test no error is thrown
                assert ttable.min(101.2, 0, 16.6, method) >= 0
                # out of bounds: source depth
                assert np.isnan(ttable.min(701.2, 0, 16.6, method))
                # out of bounds: source depth + receiver depth
                assert np.isnan(ttable.min(701, 0.3, 16.6, method))
                # out of bounds: receiver depth
                assert np.isnan(ttable.min(567.5, 0.3, 16.6, method))
                # source depths < 0 are converted to 0
                assert ttable.min(-.5, 0, 16.6, method) == ttable.min(0, 0, 16.6, method)
                # receiver depths < 0 are converted to 0
                assert ttable.min(567.5, -0.3, 16.6, method) == ttable.min(567.5, 0, 16.6, method)
                # distances are never out of bounds to be compliant with obspy travel times
                # but they are modulus 360:
                assert np.allclose(ttable.min(567.5, 0, 1.66, method), ttable.min(567.5, 0, 361.66, method))
                # distances equidistant from 180 degree are also treated as equal:
                assert ttable.min(567.5, 0, 180+1.66, method) == ttable.min(567.5, 0, 180-1.66, method)

if __name__ == "__main__":
    unittest.main()
