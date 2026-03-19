import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates


@dataclass(frozen=True)
class VolumeParams:
    n: int = 48
    radius: float = 0.28
    obj_sigma: float = 1.0


@dataclass(frozen=True)
class SimParams:
    phase_sigma: float = 1.0
    z_blur_sigma: float = 6.0
    recon_sigma: float = 1.0
    ray_samples: int = 72


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    a = axis / n
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return I + s * K + (1 - c) * (K @ K)


def _rot_matrix_align_vec_to_z(v: np.ndarray) -> np.ndarray:
    """
    Return R such that R @ v == +z (approximately), with R orthonormal.
    """
    v = np.asarray(v, dtype=np.float64)
    vn = np.linalg.norm(v)
    if vn == 0:
        return np.eye(3, dtype=np.float64)
    v = v / vn
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(v, ez), -1.0, 1.0))

    if abs(dot - 1.0) < 1e-8:
        return np.eye(3, dtype=np.float64)
    if abs(dot + 1.0) < 1e-8:
        # 180° rotation: pick any axis orthogonal to z
        return _rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi)

    axis = np.cross(v, ez)  # rotate v toward ez
    angle = math.acos(dot)
    return _rotation_matrix_from_axis_angle(axis, angle)


def _apply_rotation(vol: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Apply 3D rotation to a volume using affine_transform.
    R is a 3x3 rotation matrix mapping output coords to input coords.
    """
    vol = np.asarray(vol)
    n0, n1, n2 = vol.shape
    center = np.array([(n0 - 1) / 2.0, (n1 - 1) / 2.0, (n2 - 1) / 2.0], dtype=np.float64)
    offset = center - (R @ center)
    out = affine_transform(
        vol,
        matrix=R,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
        output=np.float32,
    )
    return out


def spherical_direction(theta_deg: float, phi_deg: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    # Internally phi는 0~360° 주기로만 중요하므로, 음수를 포함한 어떤 값이 와도
    # 0~360° 범위로 정규화한 뒤 사용한다.
    ph = math.radians(float(phi_deg) % 360.0)
    return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)], dtype=np.float64)


def detector_basis_from_v(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an orthonormal basis (u, w) spanning the detector plane, perpendicular to v.
    """
    v = np.asarray(v, dtype=np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(ez, v)
    if np.linalg.norm(u) < 1e-8:
        u = np.cross(ex, v)
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(v, u)
    w = w / (np.linalg.norm(w) + 1e-12)
    return u, w


def simulate_phase_map_ray(obj: np.ndarray, theta_deg: float, phi_deg: float, sim: SimParams) -> np.ndarray:
    """
    Ray-driven line integral. For each detector pixel (i,j), integrate obj(x0 + t v) dt.
    Detector plane is perpendicular to v, centered at origin, with pixel grid matching volume extent.
    """
    n = int(obj.shape[0])
    v = spherical_direction(theta_deg, phi_deg)
    u, w = detector_basis_from_v(v)

    coords_1d = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    A, B = np.meshgrid(coords_1d, coords_1d, indexing="ij")
    x0 = A[..., None] * u[None, None, :] + B[..., None] * w[None, None, :]

    t_max = math.sqrt(3.0)
    ns = int(max(4, sim.ray_samples))
    ts = np.linspace(-t_max, t_max, ns, dtype=np.float64)
    dt = float(ts[1] - ts[0])

    acc = np.zeros((n, n), dtype=np.float64)
    for t in ts:
        x = x0 + t * v[None, None, :]
        ix = (x[..., 0] + 1.0) * (n - 1) / 2.0
        iy = (x[..., 1] + 1.0) * (n - 1) / 2.0
        iz = (x[..., 2] + 1.0) * (n - 1) / 2.0
        samples = map_coordinates(
            obj,
            [ix.ravel(), iy.ravel(), iz.ravel()],
            order=1,
            mode="constant",
            cval=0.0,
        ).reshape(n, n)
        acc += samples

    phase = (acc * dt).astype(np.float32)
    if sim.phase_sigma and sim.phase_sigma > 0:
        phase = gaussian_filter(phase, sigma=float(sim.phase_sigma)).astype(np.float32)
    return phase


def backproject_phase_ray(phase: np.ndarray, theta_deg: float, phi_deg: float, n: int, sim: SimParams) -> np.ndarray:
    """
    Simple (nearest-voxel) ray backprojection: spread each detector pixel value along its ray.
    This is a coarse demo backprojection, not a high-quality reconstruction.
    """
    n = int(n)
    phase = np.asarray(phase, dtype=np.float64)
    v = spherical_direction(theta_deg, phi_deg)
    u, w = detector_basis_from_v(v)

    coords_1d = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    A, B = np.meshgrid(coords_1d, coords_1d, indexing="ij")
    x0 = A[..., None] * u[None, None, :] + B[..., None] * w[None, None, :]

    t_max = math.sqrt(3.0)
    ns = int(max(4, sim.ray_samples))
    ts = np.linspace(-t_max, t_max, ns, dtype=np.float64)

    vol = np.zeros((n, n, n), dtype=np.float64)
    per_sample = phase / float(ns)

    for t in ts:
        x = x0 + t * v[None, None, :]
        ix = np.rint((x[..., 0] + 1.0) * (n - 1) / 2.0).astype(np.int32)
        iy = np.rint((x[..., 1] + 1.0) * (n - 1) / 2.0).astype(np.int32)
        iz = np.rint((x[..., 2] + 1.0) * (n - 1) / 2.0).astype(np.int32)
        m = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (iz >= 0) & (iz < n)
        if not np.any(m):
            continue
        np.add.at(vol, (ix[m], iy[m], iz[m]), per_sample[m])

    vol = vol.astype(np.float32)
    if sim.z_blur_sigma and sim.z_blur_sigma > 0:
        vol = gaussian_filter(vol, sigma=(0, 0, float(sim.z_blur_sigma))).astype(np.float32)
    return vol


def make_object(p: VolumeParams) -> np.ndarray:
    n = int(p.n)
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    z = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    obj = ((X**2 + Y**2 + Z**2) < float(p.radius) ** 2).astype(np.float32)
    if p.obj_sigma and p.obj_sigma > 0:
        obj = gaussian_filter(obj, sigma=float(p.obj_sigma))
    return obj


def rotate_spherical(vol: np.ndarray, theta_deg: float, phi_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Spherical coordinates convention (physics):
      - theta: polar angle from +z, range [0, 180]
      - phi: azimuth angle in x-y plane from +x toward +y, range (-180, 180]

    For a given (theta, phi), we define the viewing direction:
      v = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))

    We rotate the object so that v aligns with +z, then project along z.
    Returns (rotated_volume, R) where R is the rotation matrix used.
    """
    v = spherical_direction(theta_deg, phi_deg)
    R = _rot_matrix_align_vec_to_z(v)
    return _apply_rotation(vol, R), R


def simulate_phase_map_rotate_sum(obj: np.ndarray, theta_deg: float, phi_deg: float, sim: SimParams) -> np.ndarray:
    obj_rot, _R = rotate_spherical(obj, theta_deg=theta_deg, phi_deg=phi_deg)
    phase = np.sum(obj_rot, axis=2).astype(np.float32)
    if sim.phase_sigma and sim.phase_sigma > 0:
        phase = gaussian_filter(phase, sigma=float(sim.phase_sigma)).astype(np.float32)
    return phase


def backproject_phase_rotate_sum(phase: np.ndarray, theta_deg: float, phi_deg: float, n: int, sim: SimParams) -> np.ndarray:
    vol = np.repeat(phase[:, :, np.newaxis], int(n), axis=2).astype(np.float32)
    if sim.z_blur_sigma and sim.z_blur_sigma > 0:
        vol = gaussian_filter(vol, sigma=(0, 0, float(sim.z_blur_sigma)))
    # Rotate back to original coordinates using inverse rotation (transpose).
    v = spherical_direction(theta_deg, phi_deg)
    R = _rot_matrix_align_vec_to_z(v)
    vol_back = _apply_rotation(vol, R.T)
    return vol_back.astype(np.float32)


def angle_sweep(start: float, stop: float, step: float) -> list[float]:
    step = float(step)
    if step == 0:
        return [float(start)]
    if (stop - start) / step < 0:
        step = -step
    vals = np.arange(float(start), float(stop) + math.copysign(1e-9, step), step, dtype=np.float32)
    return [float(v) for v in vals]


def volume_to_point_cloud(vol: np.ndarray, threshold: float, ds: int, max_points: int, seed: int = 0):
    m = vol > float(threshold)
    ds = max(1, int(ds))
    m = m[::ds, ::ds, ::ds]
    pts = np.argwhere(m)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    if pts.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        pts = pts[rng.choice(pts.shape[0], size=int(max_points), replace=False)]
    return pts


def fig_phase(phase: np.ndarray, title: str):
    n0, n1 = int(phase.shape[0]), int(phase.shape[1])
    x = np.arange(n1, dtype=np.int32)
    y = np.arange(n0, dtype=np.int32)
    fig = go.Figure(data=go.Heatmap(z=phase, x=x, y=y, colorscale="Viridis"))
    fig.update_layout(title=title, height=360, margin=dict(l=10, r=10, t=40, b=10))
    # Tight axis ranges to avoid awkward whitespace, keep square pixels.
    fig.update_xaxes(range=[-0.5, n1 - 0.5], constrain="domain")
    fig.update_yaxes(range=[n0 - 0.5, -0.5], scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def fig_cloud(pts: np.ndarray, title: str, show_axes: bool = True, axis_len: float | None = None):
    data = [
        go.Scatter3d(
            x=pts[:, 0] if pts.size else [],
            y=pts[:, 1] if pts.size else [],
            z=pts[:, 2] if pts.size else [],
            mode="markers",
            marker=dict(size=2, color="royalblue", opacity=0.65),
            name="points",
            showlegend=False,
        )
    ]

    if show_axes:
        if axis_len is None:
            axis_len = float(np.max(pts) if pts.size else 10.0)
        L = float(max(1.0, axis_len))
        data.extend(
            [
                go.Scatter3d(x=[0, L], y=[0, 0], z=[0, 0], mode="lines", line=dict(color="red", width=6), name="x"),
                go.Scatter3d(x=[0, 0], y=[0, L], z=[0, 0], mode="lines", line=dict(color="green", width=6), name="y"),
                go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, L], mode="lines", line=dict(color="blue", width=6), name="z"),
            ]
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
        ),
    )
    return fig


def ramp_filter_2d(img: np.ndarray, window: str = "hann", strength: float = 1.0) -> np.ndarray:
    """
    Simple 2D ramp (|k|) filter in Fourier domain.
    This is a CT-like high-pass filter often used before backprojection.
    """
    img = np.asarray(img, dtype=np.float32)
    n0, n1 = img.shape
    F = np.fft.fft2(img)
    fy = np.fft.fftfreq(n0).astype(np.float32)
    fx = np.fft.fftfreq(n1).astype(np.float32)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    R = np.sqrt(FX**2 + FY**2).astype(np.float32)

    if window == "hann":
        wx = 0.5 * (1.0 + np.cos(np.pi * np.clip(FX / (np.max(np.abs(fx)) + 1e-12), -1, 1))).astype(np.float32)
        wy = 0.5 * (1.0 + np.cos(np.pi * np.clip(FY / (np.max(np.abs(fy)) + 1e-12), -1, 1))).astype(np.float32)
        W = wx * wy
    elif window == "none":
        W = 1.0
    else:  # "hamming"
        wx = (0.54 + 0.46 * np.cos(np.pi * np.clip(FX / (np.max(np.abs(fx)) + 1e-12), -1, 1))).astype(np.float32)
        wy = (0.54 + 0.46 * np.cos(np.pi * np.clip(FY / (np.max(np.abs(fy)) + 1e-12), -1, 1))).astype(np.float32)
        W = wx * wy

    s = float(np.clip(strength, 0.0, 2.0))
    H = (1.0 - s) + s * (R * W)
    out = np.real(np.fft.ifft2(F * H)).astype(np.float32)
    return out


@st.cache_data(show_spinner=False)
def run_sweep_cached(
    sweep_mode: str,
    thetas: tuple[float, ...],
    phis: tuple[float, ...],
    vol_p: VolumeParams,
    sim_p: SimParams,
    projection_key: str,
    fbp_enabled: bool,
    fbp_window: str,
    fbp_strength: float,
):
    obj = make_object(vol_p)
    n = vol_p.n

    angles = []
    if sweep_mode == "theta":
        angles = [(t, phis[0]) for t in thetas]
    elif sweep_mode == "phi":
        angles = [(thetas[0], p) for p in phis]
    else:  # "grid"
        angles = [(t, p) for t in thetas for p in phis]

    phase_maps = []
    backprojs = []
    for theta, phi in angles:
        if projection_key == "ray":
            phase = simulate_phase_map_ray(obj, theta_deg=theta, phi_deg=phi, sim=sim_p)
            back = backproject_phase_ray(phase, theta_deg=theta, phi_deg=phi, n=n, sim=sim_p)
        else:
            phase = simulate_phase_map_rotate_sum(obj, theta_deg=theta, phi_deg=phi, sim=sim_p)
            back = backproject_phase_rotate_sum(phase, theta_deg=theta, phi_deg=phi, n=n, sim=sim_p)
        if fbp_enabled:
            phase = ramp_filter_2d(phase, window=fbp_window, strength=fbp_strength)
        phase_maps.append(phase)
        # If phase is filtered, backproject the filtered one for CT-like behavior.
        if fbp_enabled:
            if projection_key == "ray":
                back = backproject_phase_ray(phase, theta_deg=theta, phi_deg=phi, n=n, sim=sim_p)
            else:
                back = backproject_phase_rotate_sum(phase, theta_deg=theta, phi_deg=phi, n=n, sim=sim_p)
        backprojs.append(back)

    init_vol = np.mean(np.stack(backprojs, axis=0), axis=0).astype(np.float32)
    if sim_p.recon_sigma and sim_p.recon_sigma > 0:
        init_vol = gaussian_filter(init_vol, sigma=float(sim_p.recon_sigma)).astype(np.float32)

    return obj, angles, phase_maps, backprojs, init_vol


st.set_page_config(page_title="DHM Theta/Phi Sweep", layout="wide")
st.title("DHM Angle Sweep (theta / phi) Viewer")
st.markdown(
    "<div style='font-size:0.85rem; color:#666;'>"
    "구면좌표계 정의: theta=+z에서 내려오는 polar(0~180°), "
    "phi=xy평면 azimuth(+x→+y, 0~360° 주기)"
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("3D 오브젝트")
    n = st.slider("해상도 N", min_value=24, max_value=96, value=48, step=8)
    radius = st.slider("구 반지름", min_value=0.10, max_value=0.60, value=0.28, step=0.02)
    obj_sigma = st.slider("오브젝트 Gaussian sigma", min_value=0.0, max_value=3.0, value=1.0, step=0.25)

    st.subheader("시뮬레이션/복원")
    projection_mode = st.radio(
        "투영 방식",
        options=["회전 후 z합(빠름)", "광선 선적분(정확, 느림)"],
        index=0,
        help=(
            "회전 후 z합: 볼륨을 회전시키고 z축으로 합(sum)합니다.\n"
            "광선 선적분: (theta,phi) 방향의 실제 광선을 따라 선적분합니다(계산량↑)."
        ),
    )
    phase_sigma = st.slider("Phase map blur sigma", min_value=0.0, max_value=3.0, value=1.0, step=0.25)
    z_blur = st.slider("Backprojection z-blur sigma", min_value=0.0, max_value=12.0, value=6.0, step=0.5)
    recon_sigma = st.slider("초기볼륨 blur sigma", min_value=0.0, max_value=3.0, value=1.0, step=0.25)
    ray_samples = st.slider(
        "선적분 샘플 수(클수록 정확/느림)",
        min_value=16,
        max_value=160,
        value=72,
        step=8,
        help="광선 선적분/역투영에서 t 샘플 개수입니다. 각도 개수와 곱으로 비용이 늘어납니다.",
    )
    st.subheader("Filtered Backprojection(실험)")
    fbp_enabled = st.checkbox(
        "FBP 스타일 필터(ramp) 적용",
        value=False,
        help="2D phase map에 2D ramp(|k|) 고역통과 필터를 적용한 뒤 backprojection 합니다.",
    )
    fbp_window = st.selectbox("FBP 윈도우", options=["hann", "hamming", "none"], index=0, disabled=not fbp_enabled)
    fbp_strength = st.slider(
        "FBP 강도",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        disabled=not fbp_enabled,
        help="0이면 필터 없음, 1이면 ramp 필터 최대.",
    )

    st.subheader("3D 표시(점구름)")
    show_axes = st.checkbox("좌표축 표시(x=빨강, y=초록, z=파랑)", value=True)
    thr = st.slider("임계값(threshold)", min_value=0.05, max_value=0.90, value=0.35, step=0.05)
    ds = st.slider("다운샘플(클수록 빠름)", min_value=1, max_value=6, value=2, step=1)
    max_pts = st.slider("최대 점 개수", min_value=5_000, max_value=80_000, value=25_000, step=5_000)

tab_theta, tab_phi, tab_grid = st.tabs(["Theta Sweep", "Phi Sweep", "Theta×Phi Grid"])

vol_p = VolumeParams(n=n, radius=radius, obj_sigma=obj_sigma)
sim_p = SimParams(phase_sigma=phase_sigma, z_blur_sigma=z_blur, recon_sigma=recon_sigma, ray_samples=ray_samples)
projection_key = "ray" if projection_mode.startswith("광선") else "rotate_sum"

with st.expander("투영 방식 설명", expanded=False):
    st.markdown(
        "- **회전 후 z합(빠름)**: 볼륨을 회전 → z축 합(sum). 빠르고 단순하지만 3D 재샘플링(보간) 영향이 큽니다.\n"
        "- **광선 선적분(정확, 느림)**: 볼륨은 고정하고 (theta,phi) 방향 광선을 따라 \\(\\int f(x_0+t\\,v)\\,dt\\)를 계산합니다.\n"
        "- **주의**: backprojection은 교육용 ‘초기 추정’입니다(각도 수가 적으면 아티팩트 큼)."
    )


def _state_key(mode_name: str) -> str:
    return f"sweep_result_{mode_name}"


def _params_fingerprint(mode_name: str) -> str:
    return "|".join([mode_name, projection_key, repr(vol_p), repr(sim_p), str(fbp_enabled), fbp_window, str(fbp_strength)])


def render_common(obj, angles, phase_maps, init_vol, mode_name: str):
    st.caption(f"Sweep mode: {mode_name} | number of angles: {len(angles)}")
    colA, colB = st.columns(2)
    with colA:
        pts_obj = volume_to_point_cloud(obj / (obj.max() + 1e-8), threshold=0.30, ds=ds, max_points=max_pts)
        st.plotly_chart(
            fig_cloud(pts_obj, "True 3D object (point cloud)", show_axes=show_axes),
            width="stretch",
            key=f"cloud_true_{mode_name}",
        )
    with colB:
        vol_norm = init_vol / (init_vol.max() + 1e-8)
        pts_init = volume_to_point_cloud(vol_norm, threshold=thr, ds=ds, max_points=max_pts)
        st.plotly_chart(
            fig_cloud(pts_init, "Combined 3D initial guess (point cloud)", show_axes=show_axes),
            width="stretch",
            key=f"cloud_init_{mode_name}",
        )

    st.divider()
    idx = st.slider(
        "미리보기 인덱스",
        min_value=0,
        max_value=max(0, len(angles) - 1),
        value=min(st.session_state.get(f"preview_idx_{mode_name}", 0), max(0, len(angles) - 1)),
        step=1,
        key=f"preview_idx_{mode_name}",
    )
    theta, phi = angles[idx]
    phi_disp = (phi % 360.0 + 360.0) % 360.0
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            fig_phase(phase_maps[idx], f"2D phase map (theta={theta:.1f}°, phi={phi_disp:.1f}°)"),
            width="stretch",
            key=f"phase_{mode_name}_{idx}",
        )
    with c2:
        st.write("각도 리스트(일부)")
        lines = []
        for i, (t, p) in list(enumerate(angles))[: min(30, len(angles))]:
            p_disp = (p % 360.0 + 360.0) % 360.0
            lines.append(f"{i:03d}: theta={t:.1f}°, phi={p_disp:.1f}°")
        st.code("\n".join(lines))


with tab_theta:
    st.subheader("Theta sweep (phi fixed)")
    c1, c2, c3 = st.columns(3)
    with c1:
        t0 = st.number_input("theta start (deg, polar)", value=0.0, step=1.0)
        t1 = st.number_input("theta stop (deg, polar)", value=40.0, step=1.0)
    with c2:
        tstep = st.number_input("theta step (deg)", value=5.0, step=1.0)
    with c3:
        phi_fixed = st.number_input("phi fixed (deg, azimuth)", value=0.0, step=1.0)

    thetas = tuple(angle_sweep(t0, t1, tstep))
    phis = (float(phi_fixed),)

    if len(thetas) > 121:
        st.warning("theta 각도 개수가 너무 많습니다(>121). 속도를 위해 step을 키우는 것을 권장합니다.")

    if st.button("Run theta sweep", type="primary"):
        with st.spinner("계산 중..."):
            obj, angles, phase_maps, _backprojs, init_vol = run_sweep_cached(
                "theta",
                thetas,
                phis,
                vol_p,
                sim_p,
                projection_key,
                fbp_enabled,
                fbp_window,
                fbp_strength,
            )
        st.session_state[_state_key("theta")] = dict(
            fingerprint=_params_fingerprint("theta"),
            obj=obj,
            angles=angles,
            phase_maps=phase_maps,
            init_vol=init_vol,
        )

    saved = st.session_state.get(_state_key("theta"))
    if saved and saved.get("fingerprint") == _params_fingerprint("theta"):
        render_common(saved["obj"], saved["angles"], saved["phase_maps"], saved["init_vol"], "theta")
    else:
        st.info("After running the sweep, the result is cached and changing the preview index will not erase it.")


with tab_phi:
    st.subheader("Phi sweep (theta fixed)")
    c1, c2, c3 = st.columns(3)
    with c1:
        p0 = st.number_input("phi start (deg, azimuth)", value=0.0, step=1.0)
        p1 = st.number_input("phi stop (deg, azimuth)", value=360.0, step=1.0)
    with c2:
        pstep = st.number_input("phi step (deg)", value=5.0, step=1.0)
    with c3:
        theta_fixed = st.number_input("theta fixed (deg, polar)", value=40.0, step=1.0)

    phis = tuple(angle_sweep(p0, p1, pstep))
    thetas = (float(theta_fixed),)

    if len(phis) > 121:
        st.warning("phi 각도 개수가 너무 많습니다(>121). 속도를 위해 step을 키우는 것을 권장합니다.")

    if st.button("Run phi sweep", type="primary"):
        with st.spinner("계산 중..."):
            obj, angles, phase_maps, _backprojs, init_vol = run_sweep_cached(
                "phi",
                thetas,
                phis,
                vol_p,
                sim_p,
                projection_key,
                fbp_enabled,
                fbp_window,
                fbp_strength,
            )
        st.session_state[_state_key("phi")] = dict(
            fingerprint=_params_fingerprint("phi"),
            obj=obj,
            angles=angles,
            phase_maps=phase_maps,
            init_vol=init_vol,
        )

    saved = st.session_state.get(_state_key("phi"))
    if saved and saved.get("fingerprint") == _params_fingerprint("phi"):
        render_common(saved["obj"], saved["angles"], saved["phase_maps"], saved["init_vol"], "phi")
    else:
        st.info("After running the sweep, the result is cached and changing the preview index will not erase it.")


with tab_grid:
    st.subheader("Theta×Phi grid (both swept)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gt0 = st.number_input("theta start (deg, polar) ", value=0.0, step=1.0)
        gt1 = st.number_input("theta stop (deg, polar)  ", value=40.0, step=1.0)
    with c2:
        gp0 = st.number_input("phi start (deg, azimuth) ", value=0.0, step=1.0)
        gp1 = st.number_input("phi stop (deg, azimuth)  ", value=360.0, step=1.0)
    with c3:
        gtstep = st.number_input("theta step (deg) ", value=10.0, step=1.0)
    with c4:
        gpstep = st.number_input("phi step (deg)  ", value=10.0, step=1.0)

    thetas = tuple(angle_sweep(gt0, gt1, gtstep))
    phis = tuple(angle_sweep(gp0, gp1, gpstep))
    total = len(thetas) * len(phis)
    st.caption(f"총 각도 조합: {total} (= {len(thetas)} × {len(phis)})")
    if total > 200:
        st.warning("그리드 조합이 큽니다(>200). 속도를 위해 step을 키우는 것을 권장합니다.")

    if st.button("Run grid sweep", type="primary"):
        with st.spinner("계산 중..."):
            obj, angles, phase_maps, _backprojs, init_vol = run_sweep_cached(
                "grid",
                thetas,
                phis,
                vol_p,
                sim_p,
                projection_key,
                fbp_enabled,
                fbp_window,
                fbp_strength,
            )
        st.session_state[_state_key("grid")] = dict(
            fingerprint=_params_fingerprint("grid"),
            obj=obj,
            angles=angles,
            phase_maps=phase_maps,
            init_vol=init_vol,
        )

    saved = st.session_state.get(_state_key("grid"))
    if saved and saved.get("fingerprint") == _params_fingerprint("grid"):
        render_common(saved["obj"], saved["angles"], saved["phase_maps"], saved["init_vol"], "grid")
    else:
        st.info("After running the sweep, the result is cached and changing the preview index will not erase it.")

