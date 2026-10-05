"""Simulation pipeline: type dispatch, chunking, and result shape."""

import jax.numpy as jnp
import numpy as np
import pytest

from beamline_playground import (
    Detector,
    Mirror,
    Parabola,
    Scene,
    Segment,
    Slit,
    Source,
    Vec2,
    simulate,
)
from beamline_playground.algorithms import chunk_size_for, rayleigh_sommerfeld
from beamline_playground.instance import SceneInstance
from beamline_playground.simulate import (
    FAR_FIELD_PROPAGATORS,
    NEAR_FIELD_PROPAGATORS,
)

from .conftest import segment


def full_scene(samples_per_wavelength=10):
    return Scene(
        name="Example Scene",
        objs=[
            Source(geometry=Segment(pos_a=Vec2(x=0, y=-1), pos_b=Vec2(x=0, y=1))),
            Mirror(geometry=Parabola(pos=Vec2(x=1, y=-1), focal_length=2, extent=1, normal_inward=True)),
            Slit(geometry=Segment(pos_a=Vec2(x=-2.1, y=2), pos_b=Vec2(x=2.1, y=2))),
            Detector(geometry=Segment(pos_a=Vec2(x=-1, y=3), pos_b=Vec2(x=1, y=3))),
        ],
        dag=[[], [0], [1], [2]],
        wavelength=5,
        samples_per_wavelength=samples_per_wavelength,
    )


class TestResultShape:
    @pytest.fixture(scope="class")
    def result(self):
        return simulate(full_scene())

    def test_one_result_per_object(self, result):
        assert len(result.objs) == 4

    def test_preserves_identity(self, result):
        scene = full_scene()
        for obj, res in zip(scene.objs, result.objs, strict=True):
            assert res.type == obj.type

    def test_scene_name_is_echoed(self, result):
        assert result.scene_name == "Example Scene"

    def test_arrays_are_the_same_length(self, result):
        for obj in result.objs:
            n = len(obj.sample_x)
            assert len(obj.field_re) == n
            assert len(obj.field_im) == n
            assert len(obj.intensity_y) == n

    def test_sample_count_matches_the_scene(self, result):
        expected = full_scene().sample_counts()
        assert [len(o.sample_x) for o in result.objs] == expected

    def test_intensity_is_the_squared_magnitude(self, result):
        for obj in result.objs:
            np.testing.assert_allclose(np.abs(obj.field) ** 2, obj.intensity_y, rtol=1e-6)

    def test_by_id_finds_objects(self, result):
        first = result.objs[0]
        assert result.by_id(first.id) is first
        with pytest.raises(KeyError):
            result.by_id("nonexistent")


class TestJsonSerialisation:
    """A JS client must receive plain numbers, never stringified complexes."""

    def test_field_components_are_numbers(self):
        import json

        payload = json.loads(simulate(full_scene()).model_dump_json())
        for obj in payload["objs"]:
            assert all(isinstance(v, (int, float)) for v in obj["field_re"])
            assert all(isinstance(v, (int, float)) for v in obj["field_im"])

    def test_no_complex_field_remains(self):
        import json

        payload = json.loads(simulate(full_scene()).model_dump_json())
        assert "field_y" not in payload["objs"][0]

    def test_field_property_reassembles_complex(self):
        obj = simulate(full_scene()).objs[-1]
        field = obj.field
        assert np.iscomplexobj(field)
        np.testing.assert_allclose(field.real, obj.field_re)
        np.testing.assert_allclose(field.imag, obj.field_im)

    def test_round_trips(self):
        from beamline_playground import SimulationResult

        original = simulate(full_scene())
        again = SimulationResult.model_validate_json(original.model_dump_json())
        assert again.model_dump() == original.model_dump()


class TestTypeDispatch:
    def test_sources_start_at_unit_amplitude(self):
        result = simulate(full_scene())
        np.testing.assert_allclose(result.objs[0].intensity_y, 1.0, rtol=1e-6)

    def test_an_unlit_object_stays_dark(self):
        """Only sources emit; a mirror with no dependencies must stay dark."""
        scene = Scene(
            objs=[
                Mirror(geometry=segment(y=0)),
                Detector(geometry=segment(y=1)),
            ],
            dag=[[], [0]],
            wavelength=5,
            samples_per_wavelength=4,
        )
        result = simulate(scene)
        np.testing.assert_allclose(result.objs[0].intensity_y, 0.0)
        np.testing.assert_allclose(result.objs[1].intensity_y, 0.0)

    def test_detectors_do_not_re_radiate(self):
        """Light reaching a detector is absorbed, so it cannot illuminate anything."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="detectors absorb"):
            Scene(
                objs=[
                    Source(geometry=segment(y=0)),
                    Detector(geometry=segment(y=1)),
                    Detector(geometry=segment(y=2)),
                ],
                dag=[[], [0], [1]],
                wavelength=5,
                samples_per_wavelength=4,
            )

    def test_near_field_map_is_usable(self):
        result = simulate(full_scene(), propagators=NEAR_FIELD_PROPAGATORS)
        assert len(result.objs) == 4
        assert max(result.objs[-1].intensity_y) > 0

    def test_far_and_near_field_differ(self):
        """They are different physics; identical output would mean dispatch is dead."""
        far = simulate(full_scene(), propagators=FAR_FIELD_PROPAGATORS)
        near = simulate(full_scene(), propagators=NEAR_FIELD_PROPAGATORS)
        assert not np.allclose(far.objs[-1].intensity_y, near.objs[-1].intensity_y)

    def test_missing_propagator_is_reported(self):
        with pytest.raises(ValueError, match="no propagator"):
            simulate(full_scene(), propagators={"source": rayleigh_sommerfeld})


class TestChunking:
    """Chunking bounds memory; it must not change the answer."""

    @pytest.fixture(scope="class")
    def pair(self):
        scene = full_scene()
        instance = SceneInstance(scene)
        k = 2 * np.pi / scene.wavelength
        src, dst = instance.objs[0], instance.objs[2]
        field = jnp.ones((src["pos_x"].shape[0],), dtype=jnp.complex64)
        return k, field, src, dst

    @pytest.mark.parametrize("budget", [1 << 30, 64, 7, 1])
    def test_chunked_matches_unchunked(self, pair, budget):
        k, field, src, dst = pair
        reference = rayleigh_sommerfeld(k, field, src, dst, max_matrix_entries=1 << 30)
        chunked = rayleigh_sommerfeld(k, field, src, dst, max_matrix_entries=budget)
        np.testing.assert_allclose(np.asarray(chunked), np.asarray(reference), rtol=1e-5, atol=1e-7)

    def test_handles_a_remainder_chunk(self, pair):
        """n_dst need not be a multiple of the chunk size."""
        k, field, src, dst = pair
        n_dst = dst["pos_x"].shape[0]
        chunk = chunk_size_for(src["pos_x"].shape[0], n_dst, 7)
        assert n_dst % chunk != 0 or chunk == 1
        assert rayleigh_sommerfeld(k, field, src, dst, max_matrix_entries=7).shape == (n_dst,)

    def test_chunk_size_respects_the_budget(self):
        assert chunk_size_for(n_src=100, n_dst=1000, max_entries=1000) == 10
        assert chunk_size_for(n_src=100, n_dst=5, max_entries=1000) == 5
        assert chunk_size_for(n_src=10_000, n_dst=1000, max_entries=1) == 1

    def test_chunk_size_never_zero(self):
        assert chunk_size_for(n_src=10**9, n_dst=10, max_entries=1) >= 1


class TestSceneInstance:
    def test_element_counts_match_the_scene(self):
        scene = full_scene()
        instance = SceneInstance(scene)
        assert [o["pos_x"].shape[0] for o in instance.objs] == scene.sample_counts()

    def test_dag_is_hashable_for_jit(self):
        """The DAG is a static jit argument, so it must be hashable."""
        hash(SceneInstance(full_scene()).dag)

    def test_sample_coords_are_not_jax_arrays(self):
        """Output-only metadata must stay off the device and out of traces."""
        for coords in SceneInstance(full_scene()).sample_coords:
            assert isinstance(coords, np.ndarray)

    def test_all_kernel_arrays_present(self):
        for obj in SceneInstance(full_scene()).objs:
            assert set(obj) == {"pos_x", "pos_y", "normal_x", "normal_y", "dx"}

    def test_varying_the_wavelength_changes_the_result(self):
        """k is traced rather than static, so it must still affect the output."""

        def peak(wavelength):
            scene = Scene(
                objs=[
                    Source(geometry=Segment(pos_a=Vec2(x=0, y=-1), pos_b=Vec2(x=0, y=1))),
                    Detector(geometry=Segment(pos_a=Vec2(x=40, y=-1), pos_b=Vec2(x=40, y=1))),
                ],
                dag=[[], [0]],
                wavelength=wavelength,
                samples_per_wavelength=4,
            )
            return max(simulate(scene).objs[-1].intensity_y)

        assert peak(5.0) != pytest.approx(peak(4.0), rel=1e-3)

