import numpy as np
import matplotlib.pyplot as plt

# Quaternion (x, y, z, w)
q = np.array([0, 0.3827, 0, 0.9239])  # example

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q  # <-- scalar last
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

# Rotation matrix
R = quaternion_to_rotation_matrix(q)

# Base vector
v = np.array([1, 0, 0])

# Rotated vector
v_rot = R @ v

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original
ax.quiver(0, 0, 0, v[0], v[1], v[2])

# Rotated
ax.quiver(0, 0, 0, v_rot[0], v_rot[1], v_rot[2])

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

plt.show()