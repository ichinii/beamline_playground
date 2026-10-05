"""Scene-level validation.

Bad input used to surface as an IndexError deep inside a jit trace. These
tests pin it to a clear ValidationError at the boundary instead.
"""

import pytest
from pydantic import ValidationError

from beamline_playground import Detector, Mirror, Scene, Source, Vec2

from .conftest import segment


def scene(dag, types=("source", "mirror", "detector")):
    kinds = {"source": Source, "detector": Detector, "mirror": Mirror}
    objs = [kinds[t](geometry=segment(y=i)) for i, t in enumerate(types)]
    return Scene(objs=objs, dag=dag, wavelength=5, samples_per_wavelength=4)


class TestStructure:
    def test_accepts_a_chain(self):
        assert scene([[], [0], [1]]) is not None

    def test_accepts_a_fan_out(self):
        assert scene([[], [0], [0]]) is not None

    def test_rejects_wrong_dag_length(self):
        with pytest.raises(ValidationError, match="exactly one entry per object"):
            scene([[], [0]])

    def test_rejects_out_of_range_index(self):
        with pytest.raises(ValidationError, match="out of range"):
            scene([[], [0], [7]])

    def test_rejects_negative_index(self):
        with pytest.raises(ValidationError, match="out of range"):
            scene([[], [0], [-1]])

    def test_rejects_self_dependency(self):
        with pytest.raises(ValidationError, match="depends on itself"):
            scene([[], [1], [0]])

    def test_rejects_duplicate_dependency(self):
        with pytest.raises(ValidationError, match="more than once"):
            scene([[], [0], [0, 0]])


class TestAcyclicity:
    def test_rejects_two_cycle(self):
        with pytest.raises(ValidationError, match="cycle"):
            scene([[], [2], [1]], types=("source", "mirror", "mirror"))

    def test_rejects_three_cycle(self):
        with pytest.raises(ValidationError, match="cycle"):
            scene([[2], [0], [1]], types=("mirror", "mirror", "mirror"))

    def test_accepts_a_diamond(self):
        """Two paths reconverging is not a cycle."""
        s = scene(
            [[], [0], [0], [1, 2]],
            types=("source", "mirror", "mirror", "detector"),
        )
        assert s is not None

    def test_deep_chain_does_not_overflow_the_stack(self):
        """Validation is iterative, so a long chain must not blow up."""
        n = 3000
        objs = [Source(geometry=segment(y=0))] + [Mirror(geometry=segment(y=i + 1)) for i in range(n - 1)]
        dag = [[]] + [[i] for i in range(n - 1)]
        s = Scene(objs=objs, dag=dag, wavelength=5, samples_per_wavelength=1)
        assert len(s.objs) == n


class TestPhysics:
    def test_rejects_a_source_with_dependencies(self):
        with pytest.raises(ValidationError, match="sources emit light"):
            scene([[], [0], [1]], types=("detector", "source", "detector"))

    def test_rejects_depending_on_a_detector(self):
        with pytest.raises(ValidationError, match="detectors absorb light"):
            scene([[], [0], [1]], types=("source", "detector", "mirror"))

    def test_allows_a_mirror_chain(self):
        assert scene([[], [0], [1]], types=("source", "mirror", "detector")) is not None


class TestFieldConstraints:
    def test_rejects_non_positive_wavelength(self):
        with pytest.raises(ValidationError):
            Scene(objs=[Source(geometry=segment())], dag=[[]], wavelength=0, samples_per_wavelength=4)

    def test_rejects_non_positive_sampling(self):
        with pytest.raises(ValidationError):
            Scene(objs=[Source(geometry=segment())], dag=[[]], wavelength=1, samples_per_wavelength=0)

    def test_rejects_degenerate_radius(self):
        from beamline_playground import Circle

        with pytest.raises(ValidationError):
            Circle(pos=Vec2(x=0, y=0), radius=0, normal_inward=True)


class TestCostEstimates:
    def test_sample_counts_match_geometry(self):
        s = scene([[], [0], [1]])
        assert s.sample_counts() == [g.geometry.sample_count(4, 5) for g in s.objs]

    def test_total_is_the_sum(self):
        s = scene([[], [0], [1]])
        assert s.total_sample_count() == sum(s.sample_counts())

    def test_pair_count_counts_each_edge(self):
        s = scene([[], [0], [1]])
        n = s.sample_counts()
        assert s.propagation_pair_count() == n[0] * n[1] + n[1] * n[2]

    def test_pair_count_grows_quadratically_with_density(self):
        objs = [Source(geometry=segment(y=0)), Detector(geometry=segment(y=1))]
        cheap = Scene(objs=objs, dag=[[], [0]], wavelength=1, samples_per_wavelength=2)
        dear = Scene(objs=objs, dag=[[], [0]], wavelength=1, samples_per_wavelength=4)
        assert dear.propagation_pair_count() == pytest.approx(4 * cheap.propagation_pair_count(), rel=0.1)

    def test_isolated_objects_cost_no_pairs(self):
        s = Scene(objs=[Source(geometry=segment())], dag=[[]], wavelength=1, samples_per_wavelength=2)
        assert s.propagation_pair_count() == 0


class TestSerialisation:
    def test_round_trips_through_json(self):
        s = scene([[], [0], [1]])
        again = Scene.model_validate_json(s.model_dump_json())
        assert again.model_dump() == s.model_dump()

    def test_geometry_discriminator_selects_the_right_type(self):
        payload = {
            "objs": [
                {
                    "type": "source",
                    "geometry": {"type": "circle", "pos": {"x": 0, "y": 0}, "radius": 2, "normal_inward": True},
                }
            ],
            "dag": [[]],
            "wavelength": 1,
            "samples_per_wavelength": 1,
        }
        s = Scene.model_validate(payload)
        from beamline_playground import Circle

        assert isinstance(s.objs[0].geometry, Circle)

    def test_ids_are_unique_by_default(self):
        a, b = Source(geometry=segment()), Source(geometry=segment())
        assert a.id != b.id
