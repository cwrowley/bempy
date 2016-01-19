import unittest
from pysces.body import *
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

class TestFactories(unittest.TestCase):
    def test_cylinder(self):
        num_points = 20
        body = cylinder(1, num_points)
        points = body.get_points()
        self.assertEqual(len(points), 20)

    def test_airfoil_invalid(self):
        num_points = 20
        self.assertRaises(ValueError, naca_airfoil, "10012", num_points)

    def test_airfoil_uniform(self):
        # just some quick tests to cover code
        npoints = 20
        body = naca_airfoil("0012", npoints, uniform=True)
        self.assertEqual(len(body.get_points()), 2*npoints - 1)
        body2 = naca_airfoil("2412", npoints, te_clamp=True)
        self.assertEqual(len(body2.get_points()), 2*npoints - 1)

    def test_joukowski_foil(self):
        npts = 32
        body = joukowski_foil(num_points=npts)
        self.assertEqual(len(body.get_points()),npts)

    def test_karman_trefftz_foil(self):
        npts = 32
        body = karman_trefftz_foil(num_points=npts)
        self.assertEqual(len(body.get_points()),npts)

    def test_van_de_vooren_foil(self):
        npts = 32
        body = van_de_vooren_foil(num_points=npts)
        self.assertEqual(len(body.get_points()),npts)

class TestBody(unittest.TestCase):
    def setUp(self):
        x1 = (0,0)
        x2 = (1,0)
        self.x = np.array([x1, x2])
        self.body = Body(self.x)

    def test_body(self):
        assert_array_equal(self.body.get_points(), self.x)

    def test_translation(self):
        dx = np.array((0,1))
        body = TransformedBody(self.body, displacement=(0,1))
        assert_array_equal(body.get_points(), self.x + dx)
        assert_array_equal(body.get_points(body_frame=True), self.x)
        self.assertEqual(body.get_body(), self.body)

    def test_rotation(self):
        body = TransformedBody(self.body, angle=90)
        assert_array_almost_equal(body.get_points(), [[0,0],[0,-1]])
        assert_array_equal(body.get_points(body_frame=True), self.x)

    def test_pitching(self):
        body = Pitching(self.body, 90, 2*np.pi, 0)
        body.time = 0
        assert_array_equal(body.get_points(), self.x)
        body.time = 0.25
        assert_array_almost_equal(body.get_points(), [[0,0],[0,-1]])
        body.time = 0
        assert_array_equal(body.get_points(), self.x)

    def test_heaving(self):
        body = Heaving(self.body, (0,1), 2*np.pi, 0)
        body.time = 0
        assert_array_equal(body.get_points(), self.x)
        body.time = 0.25
        assert_array_equal(body.get_points(), [[0,1],[1,1]])

    def test_composition(self):
        new_body = TransformedBody(self.body, displacement=(-1,0))
        new_body = TransformedBody(new_body, angle=45)
        self.assertEqual(new_body.get_body(), self.body)
