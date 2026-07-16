#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
import timeit
import jax
import jax.numpy as jnp
import jax.lax as lax
from algorithms import rayleigh_sommerfeld
from algorithms import hankel
from scene import Scene
import itertools

jax.config.update("jax_enable_x64", True) # enable support for complex128
# jax.config.update("jax_transfer_guard", "log")
# jax.config.update("jax_transfer_guard", "log_explicit")

### simulate ###

@jax.jit
def accumulate(d_fields):
    stacked = jnp.stack(d_fields, axis=0)
    return jnp.sum(stacked, axis=0)

@jax.jit
def intensity(d_field):
    return jnp.abs(d_field) ** 2

@jax.jit
def total_power(d_intensity, d_dx):
    return jnp.sum(d_intensity * d_dx)

def create_sequential_trace_dag(n):
    if n == 0: return []
    dag = [[]]
    for i in range(n-1):
        dag.append([i])
    return dag

@jax.jit
def simple_source(d_obj, d_field):
    num_samples = d_obj["pos_x"].shape[0]
    return jnp.full((num_samples,), d_field, dtype=jnp.complex128)

def trace(algorithm, scene):
    print(f"tracing scene: \"{scene.name}\" with {len(scene.objs)} objects")

    # if there is no DAG preset. create one where each object depends on the previous one (sequential propagation)
    if hasattr(scene, "trace_dag") == False:
        scene.trace_dag = create_sequential_trace_dag(len(scene.objs))

    start_time = [timeit.default_timer()]
    def elapsed_time_ms():
        ms = (timeit.default_timer() - start_time[0]) * 1000.0
        start_time[0] = timeit.default_timer()
        return f"{ms:.2f}"

    d_scene = scene.instantiate()
    print(f"- instantiate scene {elapsed_time_ms()} ms")
    d_fields = []

    for i, deps in enumerate(scene.trace_dag):
        if len(deps) == 0:
            d_field = jnp.array(1.0 + 0.0j)
            d_fields.append(simple_source(d_scene.objs[i], d_field))
            print(f"- init source ({i}) {elapsed_time_ms()} ms")
        else:
            d_dst_fields = []
            for d in deps:
                d_slit = scene.objs[d].get("slit", False)
                i_slit = scene.objs[i].get("slit", False)
                d_dst_field = algorithm.propagate(d_scene, d, i, d_fields[d], d_slit, i_slit)
                print(f"- propagate ({d} -> {i}) {elapsed_time_ms()} ms")
                d_dst_fields.append(d_dst_field)
            d_fields.append(accumulate(d_dst_fields))
            print(f"- accumulate ({i}) {elapsed_time_ms()} ms")

    d_intensities = [intensity(field) for field in d_fields]
    print(f"- calc intensity {elapsed_time_ms()} ms")
    h_intensities = [np.array(jax.device_get(intensity)) for intensity in d_intensities]
    print(f"- transfer intensity {elapsed_time_ms()} ms")
    d_total_powers = [total_power(d_intensity, d_scene.objs[i]["dx"]) for i, d_intensity in enumerate(d_intensities)]
    print(f"- calc total power {elapsed_time_ms()} ms")
    h_total_powers = [np.array(jax.device_get(d_total_power)) for d_total_power in d_total_powers]
    print(f"- transfer total power {elapsed_time_ms()} ms")

    for i, deps in enumerate(scene.trace_dag):
        tot = h_total_powers[i]
        tot_deps = sum([h_total_powers[d] for d in deps])
        print(f"{deps} -> {i}:")
        print(f"  total power = {h_total_powers[i]}")
        if tot_deps > 0:
            print(f"  total power conserved (%) = {tot/tot_deps*100.0}")
    print(f"- print info {elapsed_time_ms()} ms")

    return h_intensities

### analysis ###

def plot(scene, intensities):
    for i, intensity in enumerate(intensities):
        if len(intensity) == 1:
            plt.plot(intensity, label=f'{i}', marker='o')
        else:
            plt.plot(intensity, label=f'{i}')

    plt.title(scene.name)
    plt.xlabel('sample index')
    plt.ylabel('intensity')
    fig = plt.gcf()
    plt.legend()
    plt.show()
    plt.draw()
    fig.savefig('img/prev.png')

### experiment ###

def rotate(pos, angle):
    x, y = pos
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [x * cos_a - y * sin_a, x * sin_a + y * cos_a]

samples_per_wavelength = 4
wavelength = 0.0123456789

def create_scene_law_of_reflection():
    scene = Scene("Law Of Reflection", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    scene.append_slit([-10, 1], [-1, 10])
    scene.append_line([0.1, 0], [-0.1, 0])
    scene.append_line([8, 10], [12, 10])
    scene.append_line([22, 0], [18, 0])
    return scene

def create_scene_hard_cutoff():
    scene = Scene("Hard Cutoff", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    # scene.append_point([-10, 0])
    scene.append_slit([-10, -5], [-10, 5])
    scene.append_slit([0, 0], [0, 20])
    scene.append_line([10, 10], [10, -10])
    return scene

def create_scene_single_slit():
    scene = Scene("Single Slit", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    slit_width = scene.wavelength * 32

    scene.append_slit([-10, -10], [-10, 10])
    scene.append_slit([0, slit_width / -2], [0, slit_width / 2])
    scene.append_slit([10, -10], [10, 10])
    return scene

def create_scene_double_slit():
    scene = Scene("Double Slit", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    slit_width = scene.wavelength * 4
    slit_spacing = scene.wavelength * 16 * 8

    scene.append_slit([-10, -10], [-10, 10])
    slit_radius = slit_width / 2
    slit_spacer = slit_spacing / 2 + slit_radius
    scene.append_slit([0, -slit_radius - slit_spacer], [0, slit_radius - slit_spacer])
    scene.append_slit([0, -slit_radius + slit_spacer], [0, slit_radius + slit_spacer])
    scene.append_slit([10, -10], [10, 10])
    scene.trace_dag = [
        [],
        [0],
        [0],
        [1, 2],
    ]
    return scene

def create_scene_sequential_beam(n):
    scene = Scene("Sequential Beam", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    for i in range(n):
        x = i/(n-1)*10
        scene.append_slit([x, 0], [x, 10])
    return scene

def create_scene_diagonals():
    scene = Scene("Diagonals", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    scene.append_slit([-11, -10], [-11, 10])
    scene.append_line([5, 5], [-5, -5])
    scene.append_line([-5, 5], [5, 15])
    scene.append_slit([11, 0], [11, 20])
    return scene

def create_scene_transmissive_grating(wavelength):
    scene = Scene("Transmissive Grating", samples_per_wavelength=samples_per_wavelength, wavelength=wavelength)
    scene.append_slit([-1000, -1], [-1000, 1])
    # scene.append_slit([-25, -5], [-5, -25])
    scene.trace_dag = [[]]

    d = 5e-3  # grating spacing
    a = d/2.0 # grating slit width
    n = 500   # number of slits
    s = n * d # grating size
    print(f"create_scene_transmissive_grating() wavelength = {wavelength}, grating spacing = {d}, slit width = {a}, number of slits = {n}, grating size = {s}")

    for i in range(n):
        # start = i * d - s/2
        # end = start + a
        a = (i*2.0)/(n*2.0)
        b = (i*2.0+1.0)/(n*2.0)
        a = a * s - s/2
        b = b * s - s/2
        scene.append_slit([0, a], [0, b])
        scene.trace_dag.append([0])

    scene.append_slit([1000, -10], [1000, 150])
    # scene.append_slit([50, -50], [50, 50])
    # scene.append_arc([0, 0], 50, -math.pi/2, math.pi/2, normal_inward=True)
    scene.trace_dag.append([i+1 for i in range(n)])
    return scene

def run_experiment_transmissive_grating():
    colors = ['r', 'g', 'b']
    wavelengths = [650e-6, 532e-6, 350e-6]

    scenes = [create_scene_transmissive_grating(i) for i in wavelengths]
    results = [trace(rayleigh_sommerfeld, scene) for scene in scenes]
    plt.figure()
    for i, (scene, result) in enumerate(zip(scenes, results)):
        x = np.linspace(0, 1, len(result[-1]))
        plt.plot(x, result[-1], label=f'{scene.wavelength*1e6:.0f} nm', color=colors[i])

    plt.title("Transmissive Grating")
    plt.xlabel('detector position')
    plt.ylabel('intensity')
    plt.legend()
    plt.show()

# run_experiment_transmissive_grating()

trace(rayleigh_sommerfeld, create_scene_law_of_reflection())

# scenes = []
# scenes.append(create_scene_law_of_reflection())
# scenes.append(create_scene_hard_cutoff())
# scenes.append(create_scene_single_slit())
# scenes.append(create_scene_double_slit())
# scenes.append(create_scene_sequential_beam(3))
# scenes.append(create_scene_diagonals())
# algorithms = []
# algorithms.append(hankel)
# algorithms.append(rayleigh_sommerfeld)
# [plot(scene, trace(algorithm, scene)) for scene, algorithm in itertools.product(scenes, algorithms)]

# def run():
#     [trace(algorithm, scene) for algorithm, scene in itertools.product(algorithms, scenes)]
# print(timeit.timeit(lambda: run() , number=1))
# print(timeit.timeit(lambda: run() , number=1))
