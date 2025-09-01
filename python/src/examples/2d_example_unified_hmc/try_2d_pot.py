import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def main():
    # Create a grid of x and y values
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    rotated_X = np.cos(-0.6 * np.pi / 4) * X - np.sin(-0.6 * np.pi / 4) * Y
    rotated_Y = np.sin(-0.6 * np.pi / 4) * X + np.cos(-0.6 * np.pi / 4) * Y
    # Define the surface function: Z = f(X, Y)
    Z = rotated_X**4 + rotated_Y**4 -2*(rotated_X**2)-4*(rotated_Y**2)+rotated_X*rotated_Y +0.3*rotated_X+0.1*rotated_Y

    # Create a filled contour plot
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')  # Filled contours
    plt.colorbar(contour, label="Z value")  # Add colorbar

    # Optional: add contour lines
    lines = plt.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
    plt.clabel(lines, inline=True, fontsize=8)

    # Labels and title
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.xlim((-3, 3))
    plt.ylim((-3,3))
    plt.title("2D Contour Plot with Colors")

    plt.show()
if __name__ == '__main__':
    main()