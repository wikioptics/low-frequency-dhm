import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import gaussian_filter, rotate


def gaussian(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((u - mu) / (sigma + 1e-12)) ** 2)


def phase_profile_with_mirror(
    u: np.ndarray,
    angle_deg: float,
    z_real: float,
    sigma_u: float,
    amp_real: float = 1.0,
    amp_img: float = 0.55,
) -> np.ndarray:
    """1D phase: real sphere center z=+z_real, image at z=-z_real (Gaussian blobs)."""
    a = np.deg2rad(angle_deg)
    mu_real = z_real * np.sin(a)
    mu_img = -z_real * np.sin(a)
    return amp_real * gaussian(u, mu_real, sigma_u) + amp_img * gaussian(u, mu_img, sigma_u)


def backproject_xz(
    xg: np.ndarray,
    zg: np.ndarray,
    angles_deg: list[float],
    z_real: float,
    sigma_u: float,
    amp_real: float = 1.0,
    amp_img: float = 0.55,
) -> np.ndarray:
    B = np.zeros_like(xg, dtype=np.float64)
    for ang in angles_deg:
        a = np.deg2rad(ang)
        u = xg * np.cos(a) + zg * np.sin(a)
        p = phase_profile_with_mirror(
            u,
            angle_deg=ang,
            z_real=z_real,
            sigma_u=sigma_u,
            amp_real=amp_real,
            amp_img=amp_img,
        )
        B += p
    return B


# ----- Cylinder (rotation axis = z): grid + rotate/sum like mirror_bp_chatgpt -----


def build_cylinder_masks(
    X: np.ndarray,
    Z: np.ndarray,
    gap: float,
    radius_x: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical cylinder: |x|<=radius_x, z in [gap, gap+height]. Mirror z→-z for image."""
    z_bot = float(gap)
    z_top = float(gap + height)
    real = ((np.abs(X) <= radius_x) & (Z >= z_bot) & (Z <= z_top)).astype(np.float32)
    z_img_bot = -z_top
    z_img_top = -z_bot
    img = ((np.abs(X) <= radius_x) & (Z >= z_img_bot) & (Z <= z_img_top)).astype(np.float32)
    return real, img


def phase_map_from_grid(
    real: np.ndarray,
    img: np.ndarray,
    angle_deg: float,
    image_weight: float,
    obj_smooth: float,
    phase_blur: float,
) -> np.ndarray:
    eff = real + image_weight * img
    if obj_smooth > 0:
        eff = gaussian_filter(eff, sigma=obj_smooth)
    eff_rot = rotate(eff, angle=angle_deg, reshape=False, order=1)
    phase = np.sum(eff_rot, axis=0)
    if phase_blur > 0:
        phase = gaussian_filter(phase, sigma=phase_blur)
    return phase.astype(np.float32)


def backproject_2d(
    phase: np.ndarray,
    angle_deg: float,
    n: int,
    spread_z: float,
) -> np.ndarray:
    vol = np.repeat(phase[np.newaxis, :], n, axis=0)
    if spread_z > 0:
        vol = gaussian_filter(vol, sigma=(spread_z, 0.0))
    return rotate(vol, angle=-angle_deg, reshape=False, order=1).astype(np.float32)


def draw_row_sphere(
    axes_row,
    gap: float,
    radius: float,
    z_real: float,
    x_axis: np.ndarray,
    angles_deg: list[float],
    sigma_u: float,
    amp_real: float,
    amp_img: float,
):
    ax = axes_row[0]
    ax.set_title(f"Geometry\n(gap d = {gap:.2f})", fontsize=10)
    th = np.linspace(0, 2 * np.pi, 400)
    xr = radius * np.cos(th)
    zr = z_real + radius * np.sin(th)
    zi = -z_real + radius * np.sin(th)
    ax.plot(xr, zr, color="#1f77b4", lw=2)
    ax.fill(xr, zr, color="#1f77b4", alpha=0.12)
    ax.plot(xr, zi, color="#5fa2d9", lw=1.5, alpha=0.65)
    ax.fill(xr, zi, color="#5fa2d9", alpha=0.08)
    ax.axhline(0.0, color="white", lw=1.0)
    ax.text(-0.98, 0.03, "mirror", color="gray", fontsize=8, va="bottom")
    ax.text(-0.17, z_real, "real sphere", color="#1f77b4", fontsize=8)
    ax.text(-0.24, -z_real - 0.05, "image sphere", color="#5fa2d9", fontsize=8, alpha=0.85)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("z", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    pmax = 0.1
    for j, ang in enumerate(angles_deg):
        axp = axes_row[j + 1]
        p = phase_profile_with_mirror(
            x_axis,
            angle_deg=ang,
            z_real=z_real,
            sigma_u=sigma_u,
            amp_real=amp_real,
            amp_img=amp_img,
        )
        pmax = max(pmax, float(p.max()))
        axp.fill_between(x_axis, p, color="#9ecae1", alpha=0.6)
        axp.plot(x_axis, p, color="#1f77b4", lw=1.5)
        axp.set_title(f"Phase map\n({ang:+.0f}°)", fontsize=10)
        axp.set_xlim(x_axis.min(), x_axis.max())
        axp.set_xlabel("detector x", fontsize=8)
        if j == 0:
            axp.set_ylabel("phase amplitude", fontsize=8)
        axp.tick_params(labelsize=7)
        axp.grid(alpha=0.2)

    for j in range(len(angles_deg)):
        axes_row[j + 1].set_ylim(0, pmax * 1.15)

    axb = axes_row[4]
    gx = np.linspace(-1, 1, 250)
    gz = np.linspace(-1, 1, 250)
    X, Z = np.meshgrid(gx, gz, indexing="xy")
    B = backproject_xz(
        X,
        Z,
        angles_deg=angles_deg,
        z_real=z_real,
        sigma_u=sigma_u,
        amp_real=amp_real,
        amp_img=amp_img,
    )
    im = axb.imshow(
        B,
        extent=[-1, 1, -1, 1],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    axb.set_title("Combined backprojection", fontsize=10)
    axb.axhline(0.0, color="white", lw=1.0)
    axb.text(-0.98, 0.03, "mirror", color="white", fontsize=8, va="bottom", alpha=0.9)
    axb.set_xlabel("x", fontsize=8)
    axb.set_ylabel("z", fontsize=8)
    axb.tick_params(labelsize=7)
    plt.colorbar(im, ax=axb, fraction=0.046, pad=0.02)


def draw_row_cylinder(
    axes_row,
    gap: float,
    x: np.ndarray,
    z: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    radius_x: float,
    height: float,
    angles_deg: list[float],
    image_weight: float,
    obj_smooth: float,
    phase_blur: float,
    bp_spread_z: float,
    combined_blur: float,
    truth_faint_img: float,
):
    real, img = build_cylinder_masks(X, Z, gap, radius_x, height)
    truth = real + truth_faint_img * img

    ax = axes_row[0]
    ax.set_title(f"Geometry\n(gap d = {gap:.2f})", fontsize=10)
    im0 = ax.imshow(
        truth,
        origin="lower",
        extent=[x.min(), x.max(), z.min(), z.max()],
        cmap="Blues",
        aspect="equal",
    )
    ax.axhline(0.0, color="gray", lw=1.5)
    ax.text(-0.98, 0.03, "mirror", color="gray", fontsize=8, va="bottom")
    z_mid = gap + 0.5 * height
    ax.text(0.02, z_mid, "real cylinder\n(z-axis)", color="white", fontsize=7, ha="center", va="center", weight="bold")
    ax.text(0.02, -z_mid, "image cylinder", color="#406a9f", fontsize=7, ha="center", va="center")
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("z", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.02)

    phase_maps = [
        phase_map_from_grid(real, img, a, image_weight, obj_smooth, phase_blur) for a in angles_deg
    ]
    pmax = max(float(p.max()) for p in phase_maps) if phase_maps else 0.1

    for j, (ang, ph) in enumerate(zip(angles_deg, phase_maps)):
        axp = axes_row[j + 1]
        axp.fill_between(x, ph, color="#9ecae1", alpha=0.6)
        axp.plot(x, ph, color="#1f77b4", lw=1.5)
        axp.set_title(f"Phase map\n({ang:+.0f}°)", fontsize=10)
        axp.set_xlim(x.min(), x.max())
        axp.set_ylim(0, pmax * 1.08)
        axp.set_xlabel("detector x", fontsize=8)
        if j == 0:
            axp.set_ylabel("phase amplitude", fontsize=8)
        else:
            axp.set_yticklabels([])
        axp.tick_params(labelsize=7)
        axp.grid(alpha=0.2)

    n = real.shape[0]
    bps = [backproject_2d(p, a, n, bp_spread_z) for p, a in zip(phase_maps, angles_deg)]
    combined = np.mean(np.stack(bps, axis=0), axis=0)
    if combined_blur > 0:
        combined = gaussian_filter(combined, sigma=combined_blur)

    axb = axes_row[4]
    im = axb.imshow(
        combined,
        origin="lower",
        extent=[x.min(), x.max(), z.min(), z.max()],
        cmap="viridis",
        aspect="equal",
    )
    axb.set_title("Combined backprojection", fontsize=10)
    axb.axhline(0.0, color="white", lw=1.0)
    axb.text(-0.98, 0.03, "mirror", color="white", fontsize=8, va="bottom", alpha=0.9)
    axb.set_xlabel("x", fontsize=8)
    axb.set_ylabel("z", fontsize=8)
    axb.tick_params(labelsize=7)
    plt.colorbar(im, ax=axb, fraction=0.046, pad=0.02)


def main():
    st.set_page_config(page_title="Reflective Backprojection vs Gap", layout="wide")
    st.title("Reflective backprojection vs object–mirror gap")
    st.markdown(
        "미러(`z=0`)가 있을 때, **구(가우시안)** 또는 **z축 실린더(격자 모델)** 에 대해 "
        "gap에 따른 phase map과 combined backprojection을 보는 개념 시뮬레이션입니다."
    )

    with st.sidebar:
        st.header("Parameters")
        object_mode = st.selectbox(
            "Object shape",
            options=["Sphere (analytic Gaussians)", "Cylinder (z-axis, grid)"],
            index=0,
        )
        gaps_text = st.text_input("gap list (comma-separated)", value="0.05,0.20,0.40")
        angles_text = st.text_input("angles (deg)", value="-25,0,25")

        if object_mode.startswith("Sphere"):
            radius = st.slider("sphere radius", min_value=0.05, max_value=0.35, value=0.14, step=0.01)
            sigma_u = st.slider("phase width (sigma)", min_value=0.02, max_value=0.25, value=0.09, step=0.01)
            amp_real = st.slider("real amplitude", min_value=0.2, max_value=2.0, value=1.0, step=0.05)
            amp_img = st.slider("image amplitude", min_value=0.0, max_value=2.0, value=0.55, step=0.05)
        else:
            cyl_r = st.slider("cylinder radius |x|≤R", min_value=0.05, max_value=0.35, value=0.16, step=0.01)
            cyl_h = st.slider("cylinder height (z extent)", min_value=0.05, max_value=0.8, value=0.32, step=0.01)
            grid_n = st.slider("grid N (higher = slower)", min_value=64, max_value=256, value=128, step=32)
            image_weight = st.slider("mirror image weight", min_value=0.0, max_value=1.5, value=0.9, step=0.05)
            obj_smooth = st.slider("object smooth (sigma)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            phase_blur = st.slider("phase blur after projection", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            bp_spread_z = st.slider("backproject spread along z", min_value=0.0, max_value=16.0, value=8.0, step=0.5)
            combined_blur = st.slider("combined backprojection blur", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
            truth_faint = st.slider("geometry: faint image alpha", min_value=0.0, max_value=0.8, value=0.35, step=0.05)

    try:
        gaps = [float(x.strip()) for x in gaps_text.split(",") if x.strip()]
        angles = [float(x.strip()) for x in angles_text.split(",") if x.strip()]
    except ValueError:
        st.error("숫자 형식이 올바르지 않습니다. 예: gaps='0.05,0.20,0.40', angles='-25,0,25'")
        return

    if len(gaps) == 0 or len(angles) == 0:
        st.error("gap 또는 angle 리스트가 비어 있습니다.")
        return

    if len(angles) != 3:
        st.warning("레이아웃은 각도 3개 기준입니다. 처음 3개만 사용합니다.")
        angles = angles[:3]
        while len(angles) < 3:
            angles.append(0.0)

    n_rows = len(gaps)
    fig, axes = plt.subplots(n_rows, 5, figsize=(16, 4.2 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])

    if object_mode.startswith("Sphere"):
        x_axis = np.linspace(-1.0, 1.0, 600)
        for i, gap in enumerate(gaps):
            z_real = gap + radius
            draw_row_sphere(
                axes_row=axes[i],
                gap=gap,
                radius=radius,
                z_real=z_real,
                x_axis=x_axis,
                angles_deg=angles,
                sigma_u=sigma_u,
                amp_real=amp_real,
                amp_img=amp_img,
            )
        fig.suptitle(
            "Reflective backprojection (sphere: analytic Gaussians)\n"
            "Smaller gap: real/image contributions overlap more on tilted views",
            fontsize=14,
        )
    else:
        n = int(grid_n)
        x = np.linspace(-1.0, 1.0, n)
        z = np.linspace(-1.0, 1.0, n)
        X, Z = np.meshgrid(x, z, indexing="xy")
        for i, gap in enumerate(gaps):
            draw_row_cylinder(
                axes_row=axes[i],
                gap=gap,
                x=x,
                z=z,
                X=X,
                Z=Z,
                radius_x=cyl_r,
                height=cyl_h,
                angles_deg=angles,
                image_weight=image_weight,
                obj_smooth=obj_smooth,
                phase_blur=phase_blur,
                bp_spread_z=bp_spread_z,
                combined_blur=combined_blur,
                truth_faint_img=truth_faint,
            )
        fig.suptitle(
            "Reflective backprojection (cylinder: z-axis, rotate+sum / slab backprojection)\n"
            "Vertical slab in x–z; mirror image below z=0",
            fontsize=14,
        )

    st.pyplot(fig, width="stretch")

    st.info(
        "참고: 개념 시뮬레이션입니다. 실린더 모드는 `mirror_bp_chatgpt` 스타일(격자 마스크, rotate, z합, slab backprojection)입니다."
    )


if __name__ == "__main__":
    main()
