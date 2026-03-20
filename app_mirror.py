import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


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
    """
    Conceptual 1D phase profile on detector coordinate u.
    Real sphere at z=+z_real and mirror-image sphere at z=-z_real.
    """
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
    """
    Backprojection in x-z slice:
      B(x,z) = sum_theta p_theta(u = x cos(theta) + z sin(theta))
    """
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


def draw_row(
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
    # 1) Geometry
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

    # 2~4) Phase profiles for three angles
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
        axp.fill_between(x_axis, p, color="#9ecae1", alpha=0.6)
        axp.plot(x_axis, p, color="#1f77b4", lw=1.5)
        axp.set_title(f"Phase map\n({ang:+.0f}°)", fontsize=10)
        axp.set_xlim(x_axis.min(), x_axis.max())
        axp.set_ylim(0, max(0.1, p.max() * 1.15))
        axp.set_xlabel("detector x", fontsize=8)
        if j == 0:
            axp.set_ylabel("phase amplitude", fontsize=8)
        axp.tick_params(labelsize=7)
        axp.grid(alpha=0.2)

    # 5) Combined backprojection
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


def main():
    st.set_page_config(page_title="Reflective Backprojection vs Gap", layout="wide")
    st.title("Reflective backprojection vs sphere-mirror gap")
    st.markdown(
        "미러가 있을 때, 구-미러 거리(`gap`)에 따라 각도별 phase map과 combined backprojection이 "
        "어떻게 달라지는지 보여주는 개념 시뮬레이션입니다."
    )

    with st.sidebar:
        st.header("Parameters")
        gaps_text = st.text_input("gap list (comma-separated)", value="0.05,0.20,0.40")
        radius = st.slider("sphere radius", min_value=0.05, max_value=0.35, value=0.14, step=0.01)
        sigma_u = st.slider("phase width (sigma)", min_value=0.02, max_value=0.25, value=0.09, step=0.01)
        amp_real = st.slider("real amplitude", min_value=0.2, max_value=2.0, value=1.0, step=0.05)
        amp_img = st.slider("image amplitude", min_value=0.0, max_value=2.0, value=0.55, step=0.05)
        angles_text = st.text_input("angles (deg)", value="-25,0,25")

    try:
        gaps = [float(x.strip()) for x in gaps_text.split(",") if x.strip()]
        angles = [float(x.strip()) for x in angles_text.split(",") if x.strip()]
    except ValueError:
        st.error("숫자 형식이 올바르지 않습니다. 예: gaps='0.05,0.20,0.40', angles='-25,0,25'")
        return

    if len(gaps) == 0 or len(angles) == 0:
        st.error("gap 또는 angle 리스트가 비어 있습니다.")
        return

    # Use up to 5 angles for readable layout; this app is designed around 3 columns.
    if len(angles) != 3:
        st.warning("현재 레이아웃은 각도 3개에 최적화되어 있습니다. 처음 3개 각도만 사용합니다.")
        angles = angles[:3]
        if len(angles) < 3:
            while len(angles) < 3:
                angles.append(0.0)

    x_axis = np.linspace(-1.0, 1.0, 600)
    n_rows = len(gaps)
    fig, axes = plt.subplots(n_rows, 5, figsize=(16, 4.2 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])

    for i, gap in enumerate(gaps):
        # real sphere center above mirror z=0
        z_real = gap + radius
        draw_row(
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
        "Reflective backprojection concept with varying sphere-mirror distance\n"
        "Smaller gap: stronger overlap between real/image contributions at tilted angles",
        fontsize=14,
    )
    st.pyplot(fig, width="stretch")

    st.info(
        "참고: 이 코드는 개념 시뮬레이션입니다. full-wave scattering, Fresnel 반사계수, "
        "복소장 결합 등 엄밀 광학 모델은 포함하지 않습니다."
    )


if __name__ == "__main__":
    main()
