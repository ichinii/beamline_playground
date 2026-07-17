import jax
import jax.numpy as jnp
import numpy as np
from algorithms import rayleigh_sommerfeld
# from algorithms import hankel

def homogeneous_source(n, field):
    return jnp.full((n,), field, dtype=jnp.complex128)

def _tuple_trace_dag(trace_dag):
    return tuple(tuple(deps) for deps in trace_dag)

def _propagate(k, src_field, src_obj, dst_obj):
    return rayleigh_sommerfeld.propagate(k, src_field, src_obj, dst_obj)

# TODO: maybe transpose `objs`
# TODO: feature: allow k to be a vector of wavenumbers for multi-wavelength propagation
@jax.jit(static_argnames=["k", "trace_dag"])
def _trace(k, objs, trace_dag):
    fields = [None] * len(trace_dag)

    def get(i):
        return { key: objs[key][i] for key in objs.keys() }

    for i, deps in enumerate(trace_dag):
        if len(deps) == 0:
            fields[i] = homogeneous_source(get(i)["pos_x"].shape[0], 1.0 + 0.0j)
        else:
            # this introduces a memory dependency (acc_field_1 depends acc_field_0) which prevents jax from parallelizing the loop, but it is usefull to limit the memory usage, since the intermediate fields are not stored in memory
            acc_field = jnp.zeros_like(get(i)["pos_x"], dtype=jnp.complex128)
            for d in deps:
                acc_field = acc_field + _propagate(k, fields[d], get(d), get(i))
            fields[i] = acc_field

    return fields

def trace(scene_instance):
    print(f"tracing scene: \"{scene_instance.name}\"")
    print(f"num objects: {len(scene_instance.objs["pos_x"])}")
    print(f"wavelength: {(2.0*np.pi/scene_instance.k*1e6)}nm")

    return _trace(scene_instance.k, scene_instance.objs, _tuple_trace_dag(scene_instance.trace_dag))
