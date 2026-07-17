import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

@jax.jit
def intensity(field):
    return jnp.abs(field) ** 2

@jax.jit
def intensities(fields):
    return [intensity(field) for field in fields]

@jax.jit
def _total_power(intensity, dx):
    return jnp.sum(intensity * dx)

@jax.jit
def total_powers(intensities, dxs):
    return [_total_power(intensities[i], dxs[i]) for i in range(len(dxs))]

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
