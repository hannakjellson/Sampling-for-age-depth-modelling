import numpy as np
import matplotlib.pyplot as plt

# --- Smooth cap transform ---
def smooth_cap(E, A, epsilon=1.0, alpha=0.01, p=2.0):
    """Smoothly compress energy E above threshold A.
       - E unchanged below A
       - Continuous value + derivative at A
       - Asymptotic slope ~ alpha (0 < alpha < 1)"""
    E = np.asarray(E)
    T = np.empty_like(E)
    below = E <= A
    T[below] = E[below]
    above = ~below
    if np.any(above):
        z = E[above] - A
        s = (1 - alpha) / (1 + (z/epsilon)**p) + alpha
        T[above] = A + z * s
    return T

# --- Example energy function ---
x = np.linspace(-3, 3, 1000)
E = 5*np.sin(3*x)**2 + 0.5*x**2   # original "energy landscape"
A = 4.0                           # cap threshold

E_capped = smooth_cap(E, A, epsilon=1, alpha=0.2, p=2)

# --- Plot 1: Transform T(E) vs E ---
plt.figure(figsize=(6,4))
EE = np.linspace(0, np.max(E)+1, 500)
plt.plot(EE, smooth_cap(EE, A), label="T(E)")
plt.plot(EE, EE, "--", label="Identity")
plt.axhline(A, color="red", linestyle=":", label="Cap")
plt.xlabel("E"); plt.ylabel("T(E)")
plt.title("Smooth capped transform")
plt.legend(); plt.tight_layout()

# --- Plot 2: Original vs capped along x ---
plt.figure(figsize=(6,4))
plt.plot(x, E, label="E(x)")
plt.plot(x, E_capped, label="T(E(x))")
plt.axhline(A, color="red", linestyle=":", label="Cap")
plt.xlabel("x"); plt.ylabel("Energy")
plt.title("Original vs capped energy landscape")
plt.legend(); plt.tight_layout()

plt.show()
