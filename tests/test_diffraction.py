"""Physics regression tests against closed-form results.

These are the tests that would actually catch a broken kernel. Each one
compares the simulated field to an analytic result, so they pin the physics
rather than the current implementation's output.
"""

import numpy as np
import pytest

from beamline_playground import Detector, Scene, Segment, Source, Vec2, simulate


def _normalised(detector):
    y = np.asarray(detector.sample_x)
    intensity = np.asarray(detector.intensity_y)
    return y, intensity / intensity.max()


class TestSingleSlitFraunhofer:
    """A uniformly illuminated aperture of width a gives a sinc^2 far field.

    I(theta) / I(0) = sinc^2(pi a sin(theta) / lambda)
    """

    WAVELENGTH = 0.5
    APERTURE = 4.0
    DISTANCE = 2000.0  # >> a^2/lambda = 32mm, so this is the Fraunhofer regime
    HALF_SCREEN = 600.0

    @pytest.fixture(scope="class")
    def pattern(self):
        scene = Scene(
            name="single slit",
            objs=[
                Source(geometry=Segment(pos_a=Vec2(x=0, y=-self.APERTURE / 2), pos_b=Vec2(x=0, y=self.APERTURE / 2))),
                Detector(
                    geometry=Segment(
                        pos_a=Vec2(x=self.DISTANCE, y=-self.HALF_SCREEN),
                        pos_b=Vec2(x=self.DISTANCE, y=self.HALF_SCREEN),
                    )
                ),
            ],
            dag=[[], [0]],
            wavelength=self.WAVELENGTH,
            samples_per_wavelength=4,
        )
        return _normalised(simulate(scene).objs[1])

    def _analytic(self, y):
        theta = np.arctan2(y, self.DISTANCE)
        u = self.APERTURE * np.sin(theta) / self.WAVELENGTH
        return np.sinc(u) ** 2

    def test_matches_analytic_envelope(self, pattern):
        y, intensity = pattern
        deviation = np.sqrt(np.mean((intensity - self._analytic(y)) ** 2))
        assert deviation < 5e-3, f"RMS deviation from sinc^2 was {deviation:.2e}"

    def test_peak_deviation_is_small(self, pattern):
        y, intensity = pattern
        assert np.abs(intensity - self._analytic(y)).max() < 1e-2

    def test_central_maximum_is_on_axis(self, pattern):
        y, intensity = pattern
        assert abs(y[intensity.argmax()]) < 1.0

    @pytest.mark.parametrize("order", [1, 2])
    def test_minima_land_where_predicted(self, pattern, order):
        """Zeros at sin(theta) = m * lambda / a."""
        y, intensity = pattern
        expected = self.DISTANCE * np.tan(np.arcsin(order * self.WAVELENGTH / self.APERTURE))
        window = np.abs(y - expected) < 30.0
        assert window.any(), "expected minimum lies outside the screen"
        observed = y[window][intensity[window].argmin()]
        assert observed == pytest.approx(expected, abs=10.0)
        assert intensity[window].min() < 1e-3

    def test_pattern_is_symmetric(self, pattern):
        _, intensity = pattern
        # The geometry is symmetric about y=0, so the pattern must be too.
        assert np.abs(intensity - intensity[::-1]).max() < 1e-6


class TestCylindricalDecay:
    """In 2D a line source radiates a cylindrical wave: I ~ 1/r."""

    WAVELENGTH = 0.5
    DISTANCES = (100.0, 200.0, 400.0, 800.0)

    @pytest.fixture(scope="class")
    def intensities(self):
        tiny = self.WAVELENGTH / 20  # much smaller than lambda, so effectively a point
        objs = [Source(geometry=Segment(pos_a=Vec2(x=0, y=-tiny / 2), pos_b=Vec2(x=0, y=tiny / 2)))]
        for d in self.DISTANCES:
            objs.append(Detector(geometry=Segment(pos_a=Vec2(x=d, y=-tiny / 2), pos_b=Vec2(x=d, y=tiny / 2))))
        scene = Scene(
            name="decay",
            objs=objs,
            dag=[[]] + [[0]] * len(self.DISTANCES),
            wavelength=self.WAVELENGTH,
            samples_per_wavelength=8,
        )
        result = simulate(scene)
        return np.array([np.mean(o.intensity_y) for o in result.objs[1:]])

    def test_intensity_falls_as_one_over_r(self, intensities):
        product = intensities * np.array(self.DISTANCES)
        assert product.max() / product.min() - 1 < 1e-5

    def test_amplitude_falls_as_one_over_sqrt_r(self, intensities):
        amplitude = np.sqrt(intensities)
        ratio = amplitude[0] / amplitude[-1]
        expected = np.sqrt(self.DISTANCES[-1] / self.DISTANCES[0])
        assert ratio == pytest.approx(expected, rel=1e-4)


class TestPhaseAccumulation:
    """Propagating a distance r must advance the phase by k*r."""

    def test_phase_matches_kr(self):
        wavelength = 1.0
        tiny = wavelength / 20
        distances = (50.0, 50.0 + wavelength / 4)
        objs = [Source(geometry=Segment(pos_a=Vec2(x=0, y=-tiny / 2), pos_b=Vec2(x=0, y=tiny / 2)))]
        for d in distances:
            objs.append(Detector(geometry=Segment(pos_a=Vec2(x=d, y=-tiny / 2), pos_b=Vec2(x=d, y=tiny / 2))))
        scene = Scene(
            name="phase",
            objs=objs,
            dag=[[]] + [[0]] * len(distances),
            wavelength=wavelength,
            samples_per_wavelength=8,
        )
        result = simulate(scene)
        phases = [np.angle(o.field.mean()) for o in result.objs[1:]]
        # A quarter wave of extra path is a pi/2 phase advance.
        delta = (phases[1] - phases[0]) % (2 * np.pi)
        assert delta == pytest.approx(np.pi / 2, abs=0.05)


class TestLinearity:
    """The propagation operator is linear in the source field."""

    def test_doubling_the_source_doubles_the_field(self):
        import jax.numpy as jnp

        from beamline_playground.algorithms import rayleigh_sommerfeld
        from beamline_playground.instance import SceneInstance

        scene = Scene(
            name="linear",
            objs=[
                Source(geometry=Segment(pos_a=Vec2(x=0, y=-1), pos_b=Vec2(x=0, y=1))),
                Detector(geometry=Segment(pos_a=Vec2(x=50, y=-1), pos_b=Vec2(x=50, y=1))),
            ],
            dag=[[], [0]],
            wavelength=0.5,
            samples_per_wavelength=4,
        )
        instance = SceneInstance(scene)
        k = 2 * np.pi / scene.wavelength
        src, dst = instance.objs[0], instance.objs[1]
        field = jnp.ones((src["pos_x"].shape[0],), dtype=jnp.complex64)

        single = rayleigh_sommerfeld(k, field, src, dst)
        double = rayleigh_sommerfeld(k, 2 * field, src, dst)
        np.testing.assert_allclose(np.asarray(double), 2 * np.asarray(single), rtol=1e-5)


class TestConvergence:
    """Refining the discretisation must converge, not drift."""

    def test_pattern_is_stable_under_refinement(self):
        def peak(samples_per_wavelength):
            scene = Scene(
                name="converge",
                objs=[
                    Source(geometry=Segment(pos_a=Vec2(x=0, y=-2), pos_b=Vec2(x=0, y=2))),
                    Detector(geometry=Segment(pos_a=Vec2(x=400, y=-100), pos_b=Vec2(x=400, y=100))),
                ],
                dag=[[], [0]],
                wavelength=0.5,
                samples_per_wavelength=samples_per_wavelength,
            )
            detector = simulate(scene).objs[1]
            return max(detector.intensity_y)

        coarse, fine, finer = peak(2), peak(4), peak(8)
        # Successive refinements must move the answer less and less.
        assert abs(finer - fine) < abs(fine - coarse)

