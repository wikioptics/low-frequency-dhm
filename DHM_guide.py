import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter
from pathlib import Path


# =========================
# 1. Make a simple 3D object
# =========================
print("[DHM] Building 3D object...", flush=True)
N = 48
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
z = np.linspace(-1, 1, N)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# True object: a small sphere at center
radius = 0.28
obj = ((X**2 + Y**2 + Z**2) < radius**2).astype(float)

# Smooth a bit to make plotting nicer
obj = gaussian_filter(obj, sigma=1.0)


# ==========================================
# 2. Simulate 2D phase maps from 3 angles
# ==========================================
print("[DHM] Simulating phase maps...", flush=True)
angles = [0, 25, -25]  # degrees
phase_maps = []

for ang in angles:
    # Rotate object around y-axis
    obj_rot = rotate(obj, angle=ang, axes=(0, 2), reshape=False, order=1)
    
    # Projection along z-axis -> "phase map"
    phase = np.sum(obj_rot, axis=2)
    phase = gaussian_filter(phase, sigma=1.0)
    phase_maps.append(phase)


# ==========================================
# 3. Backproject each phase map into 3D
# ==========================================
print("[DHM] Backprojecting into 3D...", flush=True)
backprojections = []

for phase, ang in zip(phase_maps, angles):
    # Expand 2D phase into 3D by copying along z
    vol = np.repeat(phase[:, :, np.newaxis], N, axis=2)

    # Blur along z so it looks like a "thick fog"
    vol = gaussian_filter(vol, sigma=(0, 0, 6))

    # Rotate back into object coordinates
    vol_back = rotate(vol, angle=-ang, axes=(0, 2), reshape=False, order=1)
    backprojections.append(vol_back)

# Combine all backprojections
init_vol = np.mean(backprojections, axis=0)
init_vol = gaussian_filter(init_vol, sigma=1.0)


# ==========================================
# 4. Visualization helpers
# ==========================================
def show_slice(ax, img, title):
    ax.imshow(img, cmap="viridis", origin="lower")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_voxels(ax, vol, threshold, title):
    # 3D voxel rendering is often extremely slow (and may appear to "hang").
    # Use a downsampled point cloud instead for responsive plotting.
    mask = vol > threshold
    ds = 2
    mask_ds = mask[::ds, ::ds, ::ds]

    pts = np.argwhere(mask_ds)
    max_points = 25_000
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], size=max_points, replace=False)]

    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="royalblue", alpha=0.7, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


# ==========================================
# 5. Plot everything
# ==========================================
print("[DHM] Rendering plots (3D voxel render can be slow)...", flush=True)
fig = plt.figure(figsize=(16, 10))

# Row 1: true object and phase maps
ax1 = fig.add_subplot(2, 4, 1, projection='3d')
plot_voxels(ax1, obj, threshold=0.3, title="True 3D object")

ax2 = fig.add_subplot(2, 4, 2)
show_slice(ax2, phase_maps[0], "2D phase map (0°)")

ax3 = fig.add_subplot(2, 4, 3)
show_slice(ax3, phase_maps[1], "2D phase map (+25°)")

ax4 = fig.add_subplot(2, 4, 4)
show_slice(ax4, phase_maps[2], "2D phase map (-25°)")

# Row 2: backprojected volumes and combined initial guess
ax5 = fig.add_subplot(2, 4, 5, projection='3d')
plot_voxels(ax5, backprojections[0] / backprojections[0].max(), threshold=0.35,
            title="Backprojection from 0°")

ax6 = fig.add_subplot(2, 4, 6, projection='3d')
plot_voxels(ax6, backprojections[1] / backprojections[1].max(), threshold=0.35,
            title="Backprojection from +25°")

ax7 = fig.add_subplot(2, 4, 7, projection='3d')
plot_voxels(ax7, backprojections[2] / backprojections[2].max(), threshold=0.35,
            title="Backprojection from -25°")

ax8 = fig.add_subplot(2, 4, 8, projection='3d')
plot_voxels(ax8, init_vol / init_vol.max(), threshold=0.45,
            title="Combined 3D initial guess")

plt.tight_layout()
out_path = Path(__file__).with_name("DHM_guide_output.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"[DHM] Figure saved to: {out_path}")

try:
    plt.show()
except Exception as e:
    print(f"[DHM] plt.show() failed: {e!r}")