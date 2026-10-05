"""Geometry discretisation invariants.

These are the properties the propagation kernels rely on: `dx` must be a true
quadrature weight, normals must be unit length, and the arc-length coordinate
must line up with the samples.
"""

import math

import numpy as np
import pytest

from beamline_playground import Arc, Circle, Ellipse, Parabola, Segment, Vec2


@pytest.mark.parametrize("n", [1, 2, 7, 64, 201])
def test_sample_returns_n_elements(geometry, n):
    s = geometry.sample(n)
    for name, array in zip(s._fields, s, strict=True):
        assert array.shape == (n,), f"{name} has shape {array.shape}, expected ({n},)"


def test_element_widths_sum_to_arc_length(geometry):
    """sum(dx) == length() -- otherwise every integral is mis-weighted."""
    length = geometry.length()
    total = float(geometry.sample(512).dx.sum())
    assert total == pytest.approx(length, rel=1e-4)


def test_normals_are_unit_vectors(geometry):
    s = geometry.sample(128)
    magnitudes = np.hypot(s.normal_x, s.normal_y)
    np.testing.assert_allclose(magnitudes, 1.0, atol=1e-12)


def test_arclength_coord_is_centred_and_monotonic(geometry):
    s = geometry.sample(128)
    assert np.all(np.diff(s.s) > 0), "arc-length coordinate must increase"
    # Centred on the geometry, so the ends are symmetric about zero.
    assert s.s[0] == pytest.approx(-s.s[-1], rel=1e-9)
    half = geometry.length() / 2
    assert abs(s.s).max() < half


def test_samples_lie_strictly_inside_the_geometry(geometry):
    """Midpoint sampling must not place a sample on an endpoint."""
    s = geometry.sample(16)
    half = geometry.length() / 2
    assert abs(s.s).max() < half


class TestClosedForms:
    def test_segment_length(self):
        assert Segment(pos_a=Vec2(x=0, y=0), pos_b=Vec2(x=3, y=4)).length() == pytest.approx(5.0)

    def test_circle_length(self):
        assert Circle(pos=Vec2(x=0, y=0), radius=2, normal_inward=False).length() == pytest.approx(4 * math.pi)

    def test_arc_length(self):
        assert Arc(pos=Vec2(x=0, y=0), radius=3, angle=2.0, normal_inward=False).length() == pytest.approx(6.0)

    def test_circular_ellipse_matches_circle(self):
        """An ellipse with equal radii over a full turn is a circle."""
        e = Ellipse(pos=Vec2(x=0, y=0), radius_x=2, radius_y=2, angle=2 * math.pi, normal_inward=False)
        assert e.length() == pytest.approx(4 * math.pi, rel=1e-6)

    def test_flat_parabola_approaches_its_width(self):
        """As the focal length grows the parabola flattens toward a segment."""
        p = Parabola(pos=Vec2(x=0, y=0), focal_length=1e6, extent=4, normal_inward=False)
        assert p.length() == pytest.approx(4.0, rel=1e-9)


class TestNormalOrientation:
    """`normal_inward` must agree with its documented meaning."""

    def test_circle_inward_points_at_centre(self):
        s = Circle(pos=Vec2(x=0, y=0), radius=2, normal_inward=True).sample(16)
        # The outward radial direction is pos/|pos|.
        dot = (s.normal_x * s.pos_x + s.normal_y * s.pos_y) / 2.0
        np.testing.assert_allclose(dot, -1.0, atol=1e-12)

    def test_circle_outward_points_away_from_centre(self):
        s = Circle(pos=Vec2(x=0, y=0), radius=2, normal_inward=False).sample(16)
        dot = (s.normal_x * s.pos_x + s.normal_y * s.pos_y) / 2.0
        np.testing.assert_allclose(dot, 1.0, atol=1e-12)

    def test_arc_inward_points_at_centre(self):
        s = Arc(pos=Vec2(x=0, y=0), radius=2, angle=1.0, normal_inward=True).sample(16)
        dot = (s.normal_x * s.pos_x + s.normal_y * s.pos_y) / 2.0
        np.testing.assert_allclose(dot, -1.0, atol=1e-12)

    def test_inward_and_outward_are_opposite(self, geometry):
        inward = geometry.model_copy(update={"normal_inward": True}) if hasattr(geometry, "normal_inward") else None
        if inward is None:
            pytest.skip("geometry has no normal_inward")
        outward = geometry.model_copy(update={"normal_inward": False})
        a, b = inward.sample(32), outward.sample(32)
        np.testing.assert_allclose(a.normal_x, -b.normal_x, atol=1e-12)
        np.testing.assert_allclose(a.normal_y, -b.normal_y, atol=1e-12)

    def test_segment_normal_points_right_of_travel(self):
        """Walking from pos_a to pos_b along +Y, 'right' is +X."""
        s = Segment(pos_a=Vec2(x=0, y=-1), pos_b=Vec2(x=0, y=1)).sample(4)
        np.testing.assert_allclose(s.normal_x, 1.0, atol=1e-12)
        np.testing.assert_allclose(s.normal_y, 0.0, atol=1e-12)


class TestClosedCurves:
    def test_circle_does_not_duplicate_a_point(self):
        """Sampling [0, 2pi] inclusively would double count one element."""
        s = Circle(pos=Vec2(x=0, y=0), radius=1, normal_inward=False).sample(8)
        gap = math.hypot(s.pos_x[0] - s.pos_x[-1], s.pos_y[0] - s.pos_y[-1])
        expected = 2 * math.sin(math.pi / 8)  # chord between neighbouring elements
        assert gap == pytest.approx(expected, rel=1e-9)

    def test_circle_elements_are_evenly_spaced(self):
        s = Circle(pos=Vec2(x=0, y=0), radius=1, normal_inward=False).sample(32)
        spacing = np.hypot(np.diff(s.pos_x), np.diff(s.pos_y))
        np.testing.assert_allclose(spacing, spacing[0], rtol=1e-9)


class TestParabolaNormals:
    """These were hardcoded placeholders before; check them against calculus."""

    def test_matches_analytic_normal(self):
        f, extent = 2.0, 4.0
        s = Parabola(pos=Vec2(x=0, y=0), focal_length=f, extent=extent, normal_inward=True).sample(33)
        u = s.pos_x
        # p(u) = (u, u^2/4f); tangent (1, u/2f); inward normal (-u/2f, 1) normalised.
        nx, ny = -u / (2 * f), np.ones_like(u)
        norm = np.hypot(nx, ny)
        np.testing.assert_allclose(s.normal_x, nx / norm, atol=1e-12)
        np.testing.assert_allclose(s.normal_y, ny / norm, atol=1e-12)

    def test_normal_varies_along_the_curve(self):
        s = Parabola(pos=Vec2(x=0, y=0), focal_length=2, extent=4, normal_inward=True).sample(16)
        assert s.normal_x.std() > 0.1, "a curved mirror cannot have a constant normal"

    def test_vertex_normal_points_along_axis(self):
        s = Parabola(pos=Vec2(x=0, y=0), focal_length=2, extent=4, normal_inward=True).sample(2)
        # Symmetric sampling straddles the vertex; normals mirror each other.
        assert s.normal_x[0] == pytest.approx(-s.normal_x[1])
        np.testing.assert_allclose(s.normal_y, s.normal_y[0])

    def test_points_lie_on_the_parabola(self):
        f = 2.0
        s = Parabola(pos=Vec2(x=1, y=-1), focal_length=f, extent=4, normal_inward=True).sample(16)
        u = s.pos_x - 1.0
        np.testing.assert_allclose(s.pos_y, -1.0 + u**2 / (4 * f), atol=1e-12)


class TestEllipse:
    def test_element_widths_vary(self):
        """The parametrisation is not arc-length uniform, so dx must not be constant."""
        s = Ellipse(pos=Vec2(x=0, y=0), radius_x=5, radius_y=1, angle=3.0, normal_inward=False).sample(64)
        assert s.dx.std() / s.dx.mean() > 0.05

    def test_dx_tracks_actual_spacing(self):
        """Each dx must match the local distance between element edges."""
        s = Ellipse(pos=Vec2(x=0, y=0), radius_x=5, radius_y=1, angle=3.0, normal_inward=False).sample(2000)
        spacing = np.hypot(np.diff(s.pos_x), np.diff(s.pos_y))
        midpoint_dx = 0.5 * (s.dx[:-1] + s.dx[1:])
        np.testing.assert_allclose(midpoint_dx, spacing, rtol=1e-4)

    def test_normal_is_perpendicular_to_the_curve(self):
        s = Ellipse(pos=Vec2(x=0, y=0), radius_x=3, radius_y=1, angle=2.0, normal_inward=True).sample(512)
        # Central difference, so the tangent estimate is centred on the sample
        # the normal belongs to.
        tangent_x = s.pos_x[2:] - s.pos_x[:-2]
        tangent_y = s.pos_y[2:] - s.pos_y[:-2]
        dot = s.normal_x[1:-1] * tangent_x + s.normal_y[1:-1] * tangent_y
        np.testing.assert_allclose(dot / np.hypot(tangent_x, tangent_y), 0.0, atol=1e-5)


class TestSampleCount:
    def test_scales_with_density_and_wavelength(self):
        g = Segment(pos_a=Vec2(x=0, y=0), pos_b=Vec2(x=10, y=0))
        assert g.sample_count(samples_per_wavelength=1, wavelength=1.0) == 10
        assert g.sample_count(samples_per_wavelength=4, wavelength=1.0) == 40
        assert g.sample_count(samples_per_wavelength=1, wavelength=0.5) == 20

    def test_never_returns_zero(self):
        """A geometry far smaller than the wavelength still needs one element."""
        g = Segment(pos_a=Vec2(x=0, y=0), pos_b=Vec2(x=1e-9, y=0))
        assert g.sample_count(samples_per_wavelength=1, wavelength=1.0) == 1
