"""
Test the extinction module
"""
from __future__ import print_function
import sys
import os
import unittest

import numpy as np
import requests.exceptions
from astropy import units as u
import numpy.testing as npt

# Append the module to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extinction import (query, f99_extinction, get_filter_extinction, 
                        FILTER_URIS)

# Expected data
response_gal_0_0 = {'EBV_SFD': 99.69757}
response_equ_0_0 = {'EBV_SFD': 0.03182}
response_equ_array = {'EBV_SFD': [0.03182, 0.03301]}


class TestQueryExtinction(unittest.TestCase):
    """ Test the query of extiction data 
    """
    def test_query_position_gal_0_0(self):
        self.assertEqual(query(0, 0, coordsys="gal"), response_gal_0_0)

    def test_query_position_equ_0_0(self):
        self.assertEqual(query(0, 0), response_equ_0_0)

    def test_query_equ_array(self):
        self.assertEqual(query([0, 1], [0, 1]), response_equ_array)

    def test_query_equ_out_limits(self):
        with self.assertRaises(requests.exceptions.HTTPError):
            query(100, 380, verbose=False)
    
    def test_query_out_of_size(self):
        #with self.assertRaises(requests.exceptions.HTTPError):
        #print(query(list(np.zeros(50000)), list(np.zeros(50000))))
        pass


class TestExtinctionCurve(unittest.TestCase):
    """ Test the computing of the extinction curve from Fitzpatrick 99
    """
    def test_fir_wavelenghts(self):
        self.assertEqual(f99_extinction(500*u.micron), [0.0010772042713472958])
    
    def test_normal_wavelenghts(self):
        self.assertEqual(f99_extinction(1*u.micron), [1.16611075588672])

    def test_normal_wavelenghts_change_units(self):
        npt.assert_array_max_ulp(f99_extinction(10000*u.Angstrom), np.array(1.16611075588672), dtype="float32")

    def test_normal_wavelenghts_array(self):
        npt.assert_array_max_ulp(f99_extinction([1, 1]*u.micron), np.array([1.16611075588672, 1.16611075588672]), dtype="float32")


class TestFilterExtinction(unittest.TestCase):
    """ Test the retrieval and computing of the extinction associated to 
    the main filters used.
    """
    def test_PanSTARRS_g(self):
        self.assertEqual(get_filter_extinction(FILTER_URIS["g"]), 3.6121011749827514)
    
    def test_PanSTARRS_r(self):
        self.assertEqual(get_filter_extinction(FILTER_URIS["r"]), 2.5687511251039137)
    
    def test_PanSTARRS_i(self):
        self.assertEqual(get_filter_extinction(FILTER_URIS["i"]), 1.897167710862949)

    def test_PanSTARRS_z(self):
        self.assertEqual(get_filter_extinction(FILTER_URIS["z"]), 1.4948335405125801)

    def test_PanSTARRS_y(self):
        self.assertEqual(get_filter_extinction(FILTER_URIS["y"]), 1.2478667172854474)

    def test_WISE_W1(self):
        self.assertEqual(get_filter_extinction(FILTER_URIS["W1"]), 0.19562893570345422)
    
    def test_WISE_W2(self):
        self.assertEqual(get_filter_extinction(FILTER_URIS["W2"]), 0.13438419437135862)

    def test_WISE_W3(self):
        self.assertEqual(get_filter_extinction(FILTER_URIS["W3"]), 0.046003159224496736)

    def test_WISE_W4(self):
        self.assertEqual(get_filter_extinction(FILTER_URIS["W4"]), 0.024851094687942197)


if __name__ == '__main__':
    unittest.main()
