import pymc as pm
import numpy as np

def energy(x):
    return (x-1)**2  # energy minimum at x=1

with pm.Model() as model:
    x = pm.Flat("x")
    pm.Potential("energy_term", -energy(x))  # NEGATE energy

with model:
    trace = pm.sample(draws=500, tune=0)
    
print(np.mean(trace["x"]))  # Should be near 1