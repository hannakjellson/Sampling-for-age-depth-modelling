#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Define your original 2D function E(x, y)
# ------------------------
def E(x, y):
    # Example: hills + paraboloid
    return np.sin(3*x) * np.cos(2*y) + 0.1*(x**2 + y**2)

# ------------------------
# Smooth 1D functions
# ------------------------
def smooth_tanh_1d(x, a, b, delta=0.1):
    return 0.5 * (np.tanh((x - a)/delta) - np.tanh((x - b)/delta))

def smoothstep(t):
    return 3*t**2 - 2*t**3

def smooth_poly_1d(x, a, b):
    w = np.zeros_like(x)
    inside = (x > a) & (x < b)
    t = (x[inside] - a) / (b - a)
    w[inside] = smoothstep(t)
    return w

# ------------------------
# Smooth scaling in 2D
# ------------------------
def smooth_tanh_scale_2d(x, y, a_box, b_box, gamma, delta=0.1):
    s_x = smooth_tanh_1d(x, a_box[0], b_box[0], delta)
    s_y = smooth_tanh_1d(y, a_box[1], b_box[1], delta)
    w = 1 - (1 - 1/gamma) * (s_x * s_y)
    return E(x, y) * w

def smooth_poly_scale_2d(x, y, a_box, b_box, gamma):
    s_x = smooth_poly_1d(x, a_box[0], b_box[0])
    s_y = smooth_poly_1d(y, a_box[1], b_box[1])
    w = 1 - (1 - 1/gamma) * (s_x * s_y)
    return E(x, y) * w

# ------------------------
# Parameters
# ------------------------
a_box = [1.0, 1.0]    # lower corner of box [x_min, y_min]
b_box = [3.0, 2.5]    # upper corner of box [x_max, y_max]
gamma = 2.0           # scaling factor inside box
delta = 0.1           # smoothness for tanh method

# Grid for plotting
nx, ny = 300, 300
x = np.linspace(0, 4, nx)
y = np.linspace(0, 4, ny)
X, Y = np.meshgrid(x, y)

# Compute functions
E_orig = E(X, Y)
E_tanh = smooth_tanh_scale_2d(X, Y, a_box, b_box, gamma, delta)
E_poly = smooth_poly_scale_2d(X, Y, a_box, b_box, gamma)

# ------------------------
# Plot contours
# ------------------------
fig, axs = plt.subplots(1, 3, figsize=(18,5))

levels = 30
cs0 = axs[0].contourf(X, Y, E_orig, levels=levels, cmap='viridis')
axs[0].set_title('Original E(x, y)')
fig.colorbar(cs0, ax=axs[0])

cs1 = axs[1].contourf(X, Y, E_tanh, levels=levels, cmap='viridis')
axs[1].set_title('Tanh smooth scaling')
fig.colorbar(cs1, ax=axs[1])
axs[1].add_patch(plt.Rectangle(a_box, b_box[0]-a_box[0], b_box[1]-a_box[1],
                               fill=False, color='white', linestyle='--', linewidth=2))

cs2 = axs[2].contourf(X, Y, E_poly, levels=levels, cmap='viridis')
axs[2].set_title('Polynomial smoothstep scaling')
fig.colorbar(cs2, ax=axs[2])
axs[2].add_patch(plt.Rectangle(a_box, b_box[0]-a_box[0], b_box[1]-a_box[1],
                               fill=False, color='white', linestyle='--', linewidth=2))

for ax in axs:
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()
