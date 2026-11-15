import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.linalg import solve as _lapack_solve
from dataclasses import dataclass
import time
from scipy.interpolate import UnivariateSpline



def normalize(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, float).copy()
    x_le = p[:, 0].min()
    chord = p[:, 0].max() - x_le
    if chord <= 0:
        raise ValueError('degenerate chord')
    p[:, 0] = (p[:, 0] - x_le) / chord
    p[:, 1] = p[:, 1] / chord
    return p


def split_surfaces(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pts, float)
    i_le = int(np.argmin(p[:, 0] + 1e-09 * np.abs(p[:, 1])))
    upper_te2le = p[:i_le + 1]
    lower_le2te = p[i_le:]
    upper = upper_te2le[::-1]
    lower = lower_le2te
    if upper[-1, 0] < 0.5 or lower[-1, 0] < 0.5:
        upper = p[p[:, 1] >= -1e-12]
        lower = p[p[:, 1] <= 1e-12]
        upper = _sort_le2te(upper)
        lower = _sort_le2te(lower)
    return (upper, lower)


def _sort_le2te(surf: np.ndarray) -> np.ndarray:
    s = np.asarray(surf, float)
    order = np.argsort(s[:, 0], kind='stable')
    return s[order]


def arclength(pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(d)))


def resample_closed(pts: np.ndarray, n: int, *, clustering: str='cosine') -> np.ndarray:
    p = np.asarray(pts, float)
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    pp = np.vstack([p, p[:1]])
    ss = arclength(pp)
    L = ss[-1]
    u = np.arange(n) / n
    if clustering == 'cosine':
        strength = 0.8
        t = u - strength / (4.0 * np.pi) * np.sin(4.0 * np.pi * u)
        t = np.mod(t, 1.0)
    else:
        t = u
    target = t * L
    return np.column_stack([np.interp(target, ss, pp[:, 0]), np.interp(target, ss, pp[:, 1])])


def cylinder_analytic(x: np.ndarray, y: np.ndarray, *, R: float=1.0, V: float=1.0, gamma: float=0.0) -> dict:
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    r_safe = np.maximum(r, 1e-12)
    a2 = R * R
    Vr = V * (1.0 - a2 / r_safe ** 2) * np.cos(th)
    Vt = -V * (1.0 + a2 / r_safe ** 2) * np.sin(th) - gamma / (2.0 * np.pi * r_safe)
    Vx = Vr * np.cos(th) - Vt * np.sin(th)
    Vy = Vr * np.sin(th) + Vt * np.cos(th)
    V2 = Vx ** 2 + Vy ** 2
    Cp = 1.0 - V2 / V ** 2
    return dict(r=r, theta=th, Vr=Vr, Vt=Vt, Vx=Vx, Vy=Vy, V2=V2, Cp=Cp)


def cylinder_o_grid(n_theta: int=240, n_radial: int=120, r_far: float=40.0, growth: float=1.035, wake_cluster: float=0.45, R: float=1.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n_theta, endpoint=False)
    theta = 2.0 * np.pi * t - wake_cluster * np.sin(2.0 * np.pi * t)
    nint = n_radial - 1
    h0 = (r_far - R) * (growth - 1.0) / (growth ** nint - 1.0)
    dr = h0 * growth ** np.arange(nint + 1)
    r = np.concatenate(([R], R + np.cumsum(dr[:-1])))
    r[-1] = r_far
    TH, RR = np.meshgrid(theta, r, indexing='ij')
    X = RR * np.cos(TH)
    Y = RR * np.sin(TH)
    return (X, Y)


def airfoil_surface_points(name: str, n_surface: int=220) -> np.ndarray:
    p = normalize(read_airfoil(name))
    return resample_closed(p, n_surface, clustering='cosine')


def _outward_normals(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, float)
    tang = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-30
    nrm = np.column_stack([tang[:, 1], -tang[:, 0]])
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area2 < 0:
        nrm = -nrm
    return nrm


def _smoothstep(s: np.ndarray | float, a: float, b: float) -> np.ndarray:
    t = np.clip((np.asarray(s, float) - a) / max(b - a, 1e-30), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _solve_growth(h0: float, nint: int, L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, float)
    lo = np.ones_like(L) * (1.0 + 1e-09)
    hi = np.ones_like(L) * 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = h0 * (mid ** nint - 1.0) / np.maximum(mid - 1.0, 1e-300) - L
        too_big = f > 0
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)
    return 0.5 * (lo + hi)


def airfoil_o_grid(name: str='NACA0015', n_surface: int=220, n_normal: int=100, r_far: float=30.0, first_height: float=0.0002, blend_distance: float=0.3, n_near: int=20, n_smooth: int=6, center: tuple[float, float]=(0.5, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    surf = airfoil_surface_points(name, n_surface)
    n = surf.shape[0]
    nrm = _outward_normals(surf)
    for _ in range(n_smooth):
        nrm = 0.5 * nrm + 0.25 * (np.roll(nrm, 1, axis=0) + np.roll(nrm, -1, axis=0))
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    C = np.asarray(center, float)
    n1 = int(min(max(n_near, 3), n_normal - 3))
    n2 = n_normal - 1 - n1
    X = np.empty((n, n_normal))
    Y = np.empty((n, n_normal))
    X[:, 0] = surf[:, 0]
    Y[:, 0] = surf[:, 1]
    g1 = float(_solve_growth(first_height, n1, np.array([blend_distance]))[0])
    step1 = first_height * g1 ** np.arange(n1)
    p = surf.copy()
    for j in range(n1):
        p = p + step1[j] * nrm
        X[:, j + 1] = p[:, 0]
        Y[:, j + 1] = p[:, 1]
    r_end = np.linalg.norm(p - C, axis=1)
    L_k = r_far - r_end
    h_last = step1[-1]
    g2 = _solve_growth(h_last, max(n2, 1), L_k)
    for k in range(n2):
        rad = (p - C) / np.maximum(np.linalg.norm(p - C, axis=1), 1e-30)[:, None]
        p = p + (h_last * g2 ** k)[:, None] * rad
        X[:, n1 + 1 + k] = p[:, 0]
        Y[:, n1 + 1 + k] = p[:, 1]
    return (X, Y)


@dataclass
class GridMetrics:
    x_xi: np.ndarray
    x_eta: np.ndarray
    y_xi: np.ndarray
    y_eta: np.ndarray
    J: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray

    @property
    def area(self) -> np.ndarray:
        return np.abs(self.J)

    def jacobian_sign_ok(self) -> bool:
        return bool(np.all(self.J > 0) or np.all(self.J < 0))


def _d_along(a: np.ndarray, axis: int, periodic: bool) -> np.ndarray:
    a = np.asarray(a, float)
    if periodic:
        return 0.5 * (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis))
    out = np.empty_like(a)
    sl_first = [slice(None)] * a.ndim
    sl_first[axis] = 0
    sl_second = list(sl_first)
    sl_second[axis] = 1
    sl_last = list(sl_first)
    sl_last[axis] = -1
    sl_second_last = list(sl_first)
    sl_second_last[axis] = -2
    out[tuple(sl_first)] = a[tuple(sl_second)] - a[tuple(sl_first)]
    out[tuple(sl_last)] = a[tuple(sl_last)] - a[tuple(sl_second_last)]
    ip = [slice(None)] * a.ndim
    ip[axis] = slice(2, None)
    im = [slice(None)] * a.ndim
    im[axis] = slice(None, -2)
    idx = [slice(None)] * a.ndim
    idx[axis] = slice(1, -1)
    out[tuple(idx)] = 0.5 * (a[tuple(ip)] - a[tuple(im)])
    return out


def curvilinear_metrics(X: np.ndarray, Y: np.ndarray, *, periodic_i: bool=True) -> GridMetrics:
    x_xi = _d_along(X, 0, periodic_i)
    x_eta = _d_along(X, 1, False)
    y_xi = _d_along(Y, 0, periodic_i)
    y_eta = _d_along(Y, 1, False)
    J = x_xi * y_eta - x_eta * y_xi
    alpha = x_eta ** 2 + y_eta ** 2
    beta = x_xi * x_eta + y_xi * y_eta
    gamma = x_xi ** 2 + y_xi ** 2
    return GridMetrics(x_xi, x_eta, y_xi, y_eta, J, alpha, beta, gamma)


def panel_nodes(name: str='NACA0015', n_panel: int=201) -> tuple[np.ndarray, dict]:
    pts = normalize(name if isinstance(name, np.ndarray) else read_airfoil(name))
    upper, lower = split_surfaces(pts)
    upper = np.array(upper, float, copy=True)
    lower = np.array(lower, float, copy=True)
    upper[:, 1] -= upper[-1, 1] * upper[:, 0]
    lower[:, 1] -= lower[-1, 1] * lower[:, 0]
    n_half = max(4, (int(n_panel) + 2) // 2)
    t = np.linspace(0.0, 1.0, n_half)
    xc = 0.5 * (1.0 - np.cos(np.pi * t))
    xu = np.interp(xc, upper[:, 0], upper[:, 1])
    xl = np.interp(xc, lower[:, 0], lower[:, 1])
    up = np.column_stack([xc, xu])
    lo = np.column_stack([xc, xl])
    nodes = np.vstack([up[::-1], lo[1:]])
    if not np.allclose(nodes[0], nodes[-1]):
        nodes[-1] = nodes[0]
    info = dict(name=str(name), n_panel_actual=int(nodes.shape[0] - 1), chord=1.0)
    return (nodes, info)


def _panel_arrays(nodes: np.ndarray) -> dict:
    p1 = nodes[:-1]
    p2 = nodes[1:]
    d = p2 - p1
    ell = np.hypot(d[:, 0], d[:, 1])
    ell = np.maximum(ell, 1e-14)
    tx = d[:, 0] / ell
    ty = d[:, 1] / ell
    nx = ty
    ny = -tx
    mid = 0.5 * (p1 + p2)
    return dict(p1=p1, p2=p2, ell=ell, tx=tx, ty=ty, nx=nx, ny=ny, mid=mid)


def _induce_single_panel(qx: np.ndarray, qy: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ell = max(np.hypot(dx, dy), 1e-14)
    tx, ty = (dx / ell, dy / ell)
    e2x, e2y = (-ty, tx)
    mx, my = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
    px = qx - mx
    py = qy - my
    xl = px * tx + py * ty
    yl = px * e2x + py * e2y
    h = 0.5 * ell
    r1 = (xl + h) ** 2 + yl ** 2
    r2 = (xl - h) ** 2 + yl ** 2
    us = np.log(np.maximum(r1, 1e-300) / np.maximum(r2, 1e-300)) / (4.0 * np.pi)
    vs = (np.arctan2(yl, xl - h) - np.arctan2(yl, xl + h)) / (2.0 * np.pi)
    u_s = us * tx + vs * e2x
    v_s = us * ty + vs * e2y
    u_v = -vs * tx + us * e2x
    v_v = -vs * ty + us * e2y
    return (u_s, v_s, u_v, v_v)


def _influence_matrices(geo: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p1, p2 = (geo['p1'], geo['p2'])
    tx, ty, nx, ny = (geo['tx'], geo['ty'], geo['nx'], geo['ny'])
    mx, my = (geo['mid'][:, 0], geo['mid'][:, 1])
    N = p1.shape[0]
    dx = p2[:, 0] - p1[:, 0]
    dy = p2[:, 1] - p1[:, 1]
    ell = np.maximum(np.hypot(dx, dy), 1e-14)
    ptx, pty = (dx / ell, dy / ell)
    pe2x, pe2y = (-pty, ptx)
    pmx = 0.5 * (p1[:, 0] + p2[:, 0])
    pmy = 0.5 * (p1[:, 1] + p2[:, 1])
    px = mx[:, None] - pmx[None, :]
    py = my[:, None] - pmy[None, :]
    xl = px * ptx[None, :] + py * pty[None, :]
    yl = px * pe2x[None, :] + py * pe2y[None, :]
    h = 0.5 * ell[None, :]
    r1 = (xl + h) ** 2 + yl ** 2
    r2 = (xl - h) ** 2 + yl ** 2
    us = np.log(np.maximum(r1, 1e-300) / np.maximum(r2, 1e-300)) / (4.0 * np.pi)
    vs = (np.arctan2(yl, xl - h) - np.arctan2(yl, xl + h)) / (2.0 * np.pi)
    u_s = us * ptx[None, :] + vs * pe2x[None, :]
    v_s = us * pty[None, :] + vs * pe2y[None, :]
    u_v = -vs * ptx[None, :] + us * pe2x[None, :]
    v_v = -vs * pty[None, :] + us * pe2y[None, :]
    An = u_s * nx[:, None] + v_s * ny[:, None]
    At = u_s * tx[:, None] + v_s * ty[:, None]
    Bn = u_v * nx[:, None] + v_v * ny[:, None]
    Bt = u_v * tx[:, None] + v_v * ty[:, None]
    idx = np.arange(N)
    An[idx, idx] = 0.5
    Bn[idx, idx] = 0.0
    At[idx, idx] = 0.0
    Bt[idx, idx] = 0.5
    return (An, At, Bn, Bt)


def solve_panel(alpha_deg: float, name: str='NACA0015', n_panel: int=201) -> dict:
    nodes, info = panel_nodes(name, n_panel)
    geo = _panel_arrays(nodes)
    An, At, Bn, Bt = _influence_matrices(geo)
    N = geo['p1'].shape[0]
    tx, ty, nx, ny = (geo['tx'], geo['ty'], geo['nx'], geo['ny'])
    tx0, ty0 = (tx[0], ty[0])
    txN, tyN = (tx[-1], ty[-1])
    a = np.deg2rad(float(alpha_deg))
    ux, uy = (np.cos(a), np.sin(a))
    A = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)
    A[:N, :N] = An
    A[:N, N] = Bn.sum(axis=1)
    b[:N] = -(ux * nx + uy * ny)
    A[N, :N] = At[0, :] + At[-1, :]
    A[N, N] = Bt[0, :].sum() + Bt[-1, :].sum()
    b[N] = -(ux * tx0 + uy * ty0) - (ux * txN + uy * tyN)
    from scipy.linalg import solve as _solve
    sol = _solve(A, b, assume_a='gen', check_finite=False)
    sigma = sol[:N]
    gamma = float(sol[N])
    q = ux * tx + uy * ty + At @ sigma + gamma * Bt.sum(axis=1)
    Ue = np.abs(q)
    Cp = 1.0 - Ue ** 2
    p1, p2 = (geo['p1'], geo['p2'])
    dxp = p2[:, 0] - p1[:, 0]
    dyp = p2[:, 1] - p1[:, 1]
    Fx = -np.sum(Cp * dyp)
    Fy = np.sum(Cp * dxp)
    Cl = -Fx * np.sin(a) + Fy * np.cos(a)
    Cd = Fx * np.cos(a) + Fy * np.sin(a)
    rx = geo['mid'][:, 0] - 0.25
    ry = geo['mid'][:, 1]
    dFx = -Cp * dyp
    dFy = Cp * dxp
    Mz = np.sum(rx * dFy - ry * dFx)
    Cm = -Mz / 1.0 ** 2
    Gamma_ccw = float(np.sum(gamma * geo['ell']))
    Cl_KJ = -2.0 * Gamma_ccw
    s_mid = np.concatenate(([0.0], np.cumsum(geo['ell'])))[:-1] + 0.5 * geo['ell']
    n_half = (N + 2) // 2
    i_up = np.arange(0, n_half - 1)
    i_lo = np.arange(n_half - 1, N)

    def _surface(idx: np.ndarray, reverse: bool) -> dict:
        ii = idx[::-1] if reverse else idx
        s_surf = np.abs(s_mid[ii] - s_mid[ii[0]])
        return dict(s=s_surf, x=geo['mid'][ii, 0].copy(), y=geo['mid'][ii, 1].copy(), Cp=Cp[ii].copy(), Ue=Ue[ii].copy())
    upper = _surface(i_up, reverse=True)
    lower = _surface(i_lo, reverse=False)
    res = dict(s=s_mid, x=geo['mid'][:, 0].copy(), y=geo['mid'][:, 1].copy(), Cp=Cp, Ue=Ue, Cl=float(Cl), Cm=float(Cm), Cd_pressure=float(Cd), Gamma=Gamma_ccw, alpha_deg=float(alpha_deg), Cl_KJ=float(Cl_KJ), sigma=sigma, gamma_panel=gamma, panel_length=geo['ell'], panel_nodes=nodes, upper=upper, lower=lower, name=info['name'], n_panel_actual=info['n_panel_actual'], Uinf=1.0)
    return res


__all__ = ['solve_bl', 'michel_critical_re_theta', 'thwaites_l', 'thwaites_H', 'head_H1_from_H', 'head_H_from_H1', 'ludwieg_tillmann_cf']


_EPS = 1e-12


def thwaites_l(lam: np.ndarray | float) -> np.ndarray:
    lam_c = np.clip(np.asarray(lam, dtype=float), -0.1, 0.36)
    return 0.22 + 1.57 * lam_c - 1.8 * lam_c ** 2


def thwaites_H(lam: np.ndarray | float) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    lam_c = np.clip(lam, -0.1, 0.36)
    H_neg = 2.088 + 0.0731 / (lam_c + 0.14)
    H_pos = 2.61 - 3.75 * lam_c + 5.24 * lam_c ** 2
    return np.where(lam_c < 0.0, H_neg, H_pos)


def michel_critical_re_theta(Re_x: np.ndarray | float) -> np.ndarray:
    Re_x = np.asarray(Re_x, dtype=float)
    return 1.174 * (1.0 + 22400.0 / np.maximum(Re_x, _EPS)) * np.maximum(Re_x, _EPS) ** 0.46


def head_H1_from_H(H: np.ndarray | float) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    H1_lo = 3.3 + 0.8234 * np.maximum(H - 1.1, 1e-06) ** (-1.287)
    H1_hi = 3.3 + 1.5501 * np.maximum(H - 0.6778, 1e-06) ** (-3.064)
    return np.where(H <= 1.6, H1_lo, H1_hi)


def head_H_from_H1(H1: np.ndarray | float) -> np.ndarray:
    H1 = np.asarray(H1, dtype=float)
    s = np.maximum(H1 - 3.3, 1e-06)
    H_lo = 1.1 + (0.8234 / s) ** (1.0 / 1.287)
    H_hi = 0.6778 + (1.5501 / s) ** (1.0 / 3.064)
    return np.where(H1 >= 5.32, H_lo, H_hi)


def ludwieg_tillmann_cf(H: np.ndarray | float, Re_theta: np.ndarray | float) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    Re_theta = np.maximum(np.asarray(Re_theta, dtype=float), 1.0)
    return 0.246 * 10.0 ** (-0.678 * H) * Re_theta ** (-0.268)


def solve_bl(s, Ue, Re, *, side: str='upper', forced_transition=None, H_sep_lam: float=3.5, H_sep_turb: float=2.4, n_sub: int=8, H_transition: float=1.5, transition_at_separation: bool=True, transition_window: float=0.02, lsb: bool=False, lsb_length: float=0.06, lsb_height: float=0.006, lsb_force_start: float | None=None) -> dict:
    s = np.asarray(s, dtype=float).ravel()
    Ue = np.abs(np.asarray(Ue, dtype=float).ravel())
    if s.shape != Ue.shape:
        raise ValueError(f'length mismatch: {s.shape} vs {Ue.shape}')
    n = s.size
    if n < 5:
        raise ValueError('need at least 5 stations')
    if not np.all(np.diff(s) > 0.0):
        raise ValueError('s must be strictly increasing')
    Re = float(Re)
    if not np.isfinite(Re) or Re <= 0:
        raise ValueError(f'invalid Re: {Re}')
    s = s - s[0]
    ds = np.diff(s)
    Ue = _smooth_arclen(s, Ue, n_uniform=401, window=5)
    f5 = Ue ** 5
    integ = np.concatenate(([0.0], np.cumsum(0.5 * (f5[:-1] + f5[1:]) * ds)))
    Ue6 = np.maximum(Ue, 1e-10) ** 6
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        theta_lam = np.sqrt(0.45 / Re * integ / Ue6)
    theta_lam[~np.isfinite(theta_lam)] = 0.0
    theta_lam[0] = 0.0
    theta_lam = np.minimum(theta_lam, 0.05 * np.maximum(s, 1e-06))
    dUe_ds = np.gradient(Ue, s, edge_order=2)
    dUe_ds = _smooth_arclen(s, dUe_ds, n_uniform=401, window=7)
    lam = theta_lam ** 2 * Re * dUe_ds
    H_lam = thwaites_H(lam)
    Re_theta_lam = Re * Ue * theta_lam
    with np.errstate(divide='ignore', invalid='ignore'):
        Cf_lam = 2.0 * thwaites_l(lam) / np.maximum(Re_theta_lam, 1e-08)
    Cf_lam[~np.isfinite(Cf_lam)] = 0.0
    Cf_lam[0] = 0.0
    Re_x = Re * Ue * s
    Re_theta_lam = Re * Ue * theta_lam
    i_lamsep = -1
    lam_raw = theta_lam ** 2 * Re * dUe_ds
    sep_lam = np.where((H_lam > H_sep_lam) | (lam_raw < -0.09))[0]
    if sep_lam.size and sep_lam[0] > 2:
        i_lamsep = int(sep_lam[0])
    if lsb and lsb_force_start is not None:
        i_forced = int(np.searchsorted(s, float(lsb_force_start) * s[-1]))
        i_lamsep = int(np.clip(i_forced, 2, n - 2))
    i_tr = -1
    if forced_transition is not None:
        fs = float(forced_transition)
        if not 0.0 < fs <= 1.0:
            raise ValueError('forced_transition must be in (0, 1]')
        i_tr = int(np.searchsorted(s, fs * s[-1]))
        i_tr = min(max(i_tr, 1), n - 2)
    else:
        with np.errstate(invalid='ignore'):
            crit = michel_critical_re_theta(Re_x)
        ok = (Re_x > 10000.0) & (Re_theta_lam > crit)
        idx = np.where(ok)[0]
        if idx.size:
            i_tr = int(idx[0])
        if transition_at_separation and i_lamsep > 0 and (i_tr < 0 or i_lamsep < i_tr):
            i_tr = i_lamsep
    theta = theta_lam.copy()
    H = H_lam.copy()
    Cf = Cf_lam.copy()
    H1 = head_H1_from_H(H)
    laminar = np.ones(n, dtype=bool)
    bubble = np.zeros(n, dtype=bool)
    delta_bubble = np.zeros(n)
    i_sep = -1
    x_tr = float('nan')
    x_sep = float('nan')
    x_reatt = float('nan')
    lsb_active = False
    win = float(transition_window) * s[-1]
    if i_tr > 0:
        x_tr = float(s[i_tr])
        if lsb and i_lamsep > 0 and (i_lamsep <= i_tr):
            lsb_active = True
            i_sep_b = i_lamsep
            x_sep = float(s[i_sep_b])
            L_b = float(lsb_length) * s[-1]
            i_reatt = int(np.searchsorted(s, x_sep + L_b))
            i_reatt = int(np.clip(i_reatt, i_sep_b + 1, n - 1))
            x_reatt = float(s[i_reatt])
            L_b = max(x_reatt - x_sep, 1e-06)
            xi = np.clip((s[i_sep_b:i_reatt] - x_sep) / L_b, 0.0, 1.0)
            h_b = float(lsb_height) * s[-1]
            delta_bubble[i_sep_b:i_reatt] = h_b * np.sin(np.pi * xi)
            bubble[i_sep_b:i_reatt] = True
            laminar[i_sep_b:i_reatt] = True
            theta_b = float(theta_lam[i_sep_b])
            H_b = float(H_sep_lam)
            theta[i_sep_b:i_reatt] = theta_b
            H[i_sep_b:i_reatt] = H_b
            H1[i_sep_b:i_reatt] = float(head_H1_from_H(H_b))
            Cf[i_sep_b:i_reatt] = 0.0
            thr = _march_turbulent(s, ds, Ue, dUe_ds, Re, i0=i_reatt, theta0=1.25 * theta_b, H0=float(H_transition), n_sub=n_sub, H_sep_turb=H_sep_turb)
            w2 = np.clip((s - x_reatt) / max(win, 1e-09), 0.0, 1.0)
            w2 = w2 * w2 * (3.0 - 2.0 * w2)
            good = np.isfinite(thr['theta'])
            for k in range(n):
                if k < i_reatt:
                    continue
                wk = w2[k]
                if good[k]:
                    theta[k] = (1.0 - wk) * theta_b + wk * thr['theta'][k]
                    H[k] = (1.0 - wk) * H_b + wk * thr['H'][k]
                    Cf[k] = wk * thr['Cf'][k]
                    H1[k] = (1.0 - wk) * float(head_H1_from_H(H_b)) + wk * thr['H1'][k]
                else:
                    theta[k] = np.nan
                    H[k] = np.nan
                    Cf[k] = np.nan
                    H1[k] = np.nan
            laminar[i_reatt:] = w2[i_reatt:] < 0.5
            if thr['i_sep'] >= 0:
                i_sep = int(thr['i_sep'])
        else:
            i_lo = int(np.searchsorted(s, s[i_tr] - win))
            i_lo = int(np.clip(i_lo, 1, n - 2))
            i_hi = int(np.searchsorted(s, s[i_tr] + win))
            i_hi = int(np.clip(i_hi, i_lo + 1, n - 1))
            thr = _march_turbulent(s, ds, Ue, dUe_ds, Re, i0=i_lo, theta0=float(theta_lam[i_lo]), H0=float(H_transition), n_sub=n_sub, H_sep_turb=H_sep_turb)
            s_lo, s_hi = (float(s[i_lo]), float(s[i_hi]))
            w = np.clip((s - s_lo) / max(s_hi - s_lo, 1e-09), 0.0, 1.0)
            w = w * w * (3.0 - 2.0 * w)
            good = np.isfinite(thr['theta'])
            for k in range(n):
                wk = w[k]
                if wk <= 0.0:
                    continue
                if good[k]:
                    theta[k] = (1.0 - wk) * theta_lam[k] + wk * thr['theta'][k]
                    H[k] = (1.0 - wk) * H_lam[k] + wk * thr['H'][k]
                    Cf[k] = (1.0 - wk) * Cf_lam[k] + wk * thr['Cf'][k]
                    H1[k] = (1.0 - wk) * head_H1_from_H(H_lam[k]) + wk * thr['H1'][k]
                elif wk > 0.5:
                    theta[k] = np.nan
                    H[k] = np.nan
                    Cf[k] = np.nan
                    H1[k] = np.nan
            laminar = w < 0.5
            if thr['i_sep'] >= 0:
                i_sep = int(thr['i_sep'])
    elif i_lamsep > 0:
        i_sep = i_lamsep
        theta[i_sep + 1:] = np.nan
        H[i_sep + 1:] = np.nan
        Cf[i_sep + 1:] = np.nan
        H1[i_sep + 1:] = np.nan
        laminar[i_sep + 1:] = False
    if i_sep >= 0:
        theta[i_sep + 1:] = np.nan
        H[i_sep + 1:] = np.nan
        Cf[i_sep + 1:] = np.nan
        H1[i_sep + 1:] = np.nan
        delta_bubble[i_sep + 1:] = 0.0
        bubble[i_sep + 1:] = False
    delta_star = theta * H + delta_bubble
    Re_theta = Re * Ue * theta
    if i_sep >= 0:
        Re_theta[i_sep + 1:] = np.nan
    return dict(s=s, theta=theta, H=H, Cf=Cf, delta_star=delta_star, x_tr=x_tr, i_sep=int(i_sep), Re_x=Re_x, Re_theta=Re_theta, H1=H1, laminar=laminar, bubble=bubble, delta_bubble=delta_bubble, x_sep=x_sep, x_reatt=x_reatt, lsb_active=bool(lsb_active), lsb_force_start=float(lsb_force_start) if lsb_force_start is not None else float('nan'), transition_index=int(i_tr), separation_index=int(i_sep), side=str(side), Re=Re)


def _march_turbulent(s, ds, Ue, dUe_ds, Re, *, i0: int, theta0: float, H0: float, n_sub: int, H_sep_turb: float) -> dict:
    n = s.size
    th = np.full(n, np.nan)
    Hh = np.full(n, np.nan)
    cf = np.full(n, np.nan)
    h1 = np.full(n, np.nan)
    i_sep = -1
    theta_cur = max(float(theta0), 1e-09)
    H1_cur = float(head_H1_from_H(H0))
    E_cur = H1_cur * theta_cur
    th[i0] = theta_cur
    Hh[i0] = float(H0)
    h1[i0] = H1_cur
    cf[i0] = float(ludwieg_tillmann_cf(H0, Re * Ue[i0] * theta_cur))
    for i in range(i0, n - 1):
        h = ds[i]
        stopped = False
        theta_i = theta_cur
        E_i = E_cur
        for k in range(n_sub):
            xa = s[i] + h * k / n_sub
            xb = s[i] + h * (k + 1) / n_sub
            xm = 0.5 * (xa + xb)
            Ue_m = max(float(np.interp(xm, s, Ue)), 1e-08)
            dUe_m = float(np.interp(xm, s, dUe_ds))
            dtheta, dE, _ = _turb_rhs(theta_i, E_i, Ue_m, dUe_m, Re)
            th_p = max(theta_i + h / n_sub * dtheta, 1e-08)
            E_p = max(E_i + h / n_sub * dE, 0.0)
            dtheta2, dE2, _ = _turb_rhs(th_p, E_p, Ue_m, dUe_m, Re)
            theta_i = max(theta_i + 0.5 * h / n_sub * (dtheta + dtheta2), 1e-09)
            E_i = max(E_i + 0.5 * h / n_sub * (dE + dE2), 0.0)
            if E_i / theta_i <= 3.31:
                stopped = True
                break
        theta_cur = theta_i
        E_cur = E_i
        H1_c = E_cur / max(theta_cur, _EPS)
        H_c = float(head_H_from_H1(H1_c))
        Cf_c = float(ludwieg_tillmann_cf(H_c, Re * float(Ue[i + 1]) * theta_cur))
        if stopped or H_c > H_sep_turb or Cf_c <= 0.0:
            i_sep = i + 1
            break
        th[i + 1] = theta_cur
        Hh[i + 1] = H_c
        cf[i + 1] = Cf_c
        h1[i + 1] = H1_c
    return dict(theta=th, H=Hh, Cf=cf, H1=h1, i_sep=i_sep)


def _smooth_arclen(s: np.ndarray, y: np.ndarray, *, n_uniform: int=401, window: int=5) -> np.ndarray:
    y = np.asarray(y, float)
    n = y.size
    if n < 12 or window < 3:
        return y.copy()
    su = np.linspace(s[0], s[-1], int(max(n_uniform, 2 * window + 3)))
    yu = np.interp(su, s, y)
    pad = window // 2
    kernel = np.ones(window) / float(window)
    yp = np.concatenate((yu[:pad][::-1], yu, yu[-pad:][::-1]))
    sm = np.convolve(yp, kernel, mode='valid')
    sm[:pad] = yu[:pad]
    sm[-pad:] = yu[-pad:]
    return np.interp(s, su, sm)


def _turb_rhs(theta: float, E: float, Ue: float, dUe_ds: float, Re: float):
    theta = max(theta, 1e-10)
    H1 = max(E / theta, 3.31)
    H = float(head_H_from_H1(H1))
    Re_theta = max(Re * Ue * theta, 1.0)
    Cf = float(ludwieg_tillmann_cf(H, Re_theta))
    dtheta = Cf / 2.0 - (2.0 + H) * theta / Ue * dUe_ds
    dE = 0.0306 * max(H1 - 3.0, 0.0001) ** (-0.653)
    return (dtheta, dE, H)


def _d_xi(a: np.ndarray, periodic: bool=True) -> np.ndarray:
    if periodic:
        return 0.5 * (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0))
    out = np.empty_like(a)
    out[0] = a[1] - a[0]
    out[-1] = a[-1] - a[-2]
    out[1:-1] = 0.5 * (a[2:] - a[:-2])
    return out


def _d_eta(a: np.ndarray) -> np.ndarray:
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] - a[:, 0]
    out[:, -1] = a[:, -1] - a[:, -2]
    out[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2])
    return out


class LaplacianOperator:

    def __init__(self, X: np.ndarray, Y: np.ndarray, *, periodic_i: bool=True):
        self.X = np.asarray(X, float)
        self.Y = np.asarray(Y, float)
        self.ni, self.nj = self.X.shape
        self.periodic_i = periodic_i
        self.metrics = curvilinear_metrics(self.X, self.Y, periodic_i=periodic_i)
        m = self.metrics
        J = m.J
        if not (np.all(J > 0.0) or np.all(J < 0.0)):
            raise ValueError('non-uniform Jacobian sign')
        self.jacobian_sign = 1.0 if J[0, 0] > 0.0 else -1.0
        A = m.alpha / J
        B = m.beta / J
        C = m.gamma / J
        A_xi = _d_xi(A, periodic_i)
        B_xi = _d_xi(B, periodic_i)
        B_eta = _d_eta(B)
        C_eta = _d_eta(C)
        invJ = 1.0 / J
        self.invJ = invJ
        self.coef = {(0, 0): invJ * (-2.0 * A - 2.0 * C), (1, 0): invJ * (A + 0.5 * (A_xi - B_eta)), (-1, 0): invJ * (A - 0.5 * (A_xi - B_eta)), (0, 1): invJ * (C + 0.5 * (C_eta - B_xi)), (0, -1): invJ * (C - 0.5 * (C_eta - B_xi)), (1, 1): invJ * (-0.5 * B), (1, -1): invJ * (0.5 * B), (-1, 1): invJ * (0.5 * B), (-1, -1): invJ * (-0.5 * B)}
        self.nabla_eta_wall = np.sqrt(m.gamma[:, 0]) / np.abs(J[:, 0])
        self.nabla_eta_far = np.sqrt(m.gamma[:, -1]) / np.abs(J[:, -1])

    def apply(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, float)
        out = np.zeros_like(f)
        for (di, dj), c in self.coef.items():
            nb = np.roll(f, -di, axis=0)
            if dj == 0:
                out += c * nb
            else:
                out[:, 1:-1] += c[:, 1:-1] * nb[:, 1 + dj:self.nj - 1 + dj]
        return out


class PoissonSolver:

    def __init__(self, lap: LaplacianOperator, kind: str, *, dt: float=0.0, nu: float=0.0):
        if kind not in ('streamfunction', 'streamfunction_neumann', 'pressure', 'helmholtz'):
            raise ValueError(f'unknown kind: {kind}')
        self.lap = lap
        self.kind = kind
        self.dt = float(dt)
        self.nu = float(nu)
        self.ni, self.nj = (lap.ni, lap.nj)
        self._lu = self._factorize()

    def _factorize(self):
        ni, nj = (self.ni, self.nj)
        idx = np.arange(ni * nj, dtype=np.int64).reshape(ni, nj)
        rows, cols, vals = ([], [], [])
        j_int = np.arange(1, nj - 1)
        row_int = idx[:, j_int].ravel()
        for (di, dj), c in self.lap.coef.items():
            col = np.roll(idx, -di, axis=0)[:, j_int + dj].ravel()
            rows.append(row_int)
            cols.append(col)
            if self.kind == 'helmholtz':
                cc = -self.dt * self.nu * c[:, j_int]
                if di == 0 and dj == 0:
                    cc = 1.0 + cc
                vals.append(cc.ravel())
            else:
                vals.append(c[:, j_int].ravel())
        row_wall = idx[:, 0]
        row_far = idx[:, -1]
        if self.kind == 'helmholtz':
            rows += [row_wall, row_far]
            cols += [row_wall, row_far]
            vals += [np.ones(ni), np.ones(ni)]
        elif self.kind == 'streamfunction':
            rows += [row_wall, row_far]
            cols += [row_wall, row_far]
            vals += [np.ones(ni), np.ones(ni)]
        elif self.kind == 'streamfunction_neumann':
            fac = self.lap.nabla_eta_far
            rows += [row_wall, row_far, row_far, row_far]
            cols += [row_wall, idx[:, -1], idx[:, -2], idx[:, -3]]
            vals += [np.ones(ni), 1.5 * fac, -2.0 * fac, 0.5 * fac]
        else:
            fac = self.lap.nabla_eta_wall
            rows += [row_wall, row_wall, row_wall, row_far]
            cols += [idx[:, 0], idx[:, 1], idx[:, 2], row_far]
            vals += [-1.5 * fac, 2.0 * fac, -0.5 * fac, np.ones(ni)]
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        val = np.concatenate(vals)
        mat = sparse.csc_matrix((val, (row, col)), shape=(ni * nj, ni * nj))
        return splu(mat, permc_spec='COLAMD')

    def solve(self, interior_rhs: np.ndarray, *, wall_rhs=None, far_rhs=None) -> np.ndarray:
        ni, nj = (self.ni, self.nj)
        rhs = np.zeros((ni, nj))
        rhs[:, 1:-1] = interior_rhs[:, 1:-1]
        if self.kind == 'helmholtz':
            rhs[:, 1:-1] = interior_rhs[:, 1:-1]
            rhs[:, 0] = 0.0 if wall_rhs is None else np.broadcast_to(np.asarray(wall_rhs, float), (ni,))
            rhs[:, -1] = 0.0
            sol = self._lu.solve(rhs.ravel())
            return sol.reshape(ni, nj)
        if self.kind == 'streamfunction':
            rhs[:, 0] = 0.0
            rhs[:, -1] = 0.0 if far_rhs is None else np.broadcast_to(np.asarray(far_rhs, float), (ni,))
        elif self.kind == 'streamfunction_neumann':
            rhs[:, 0] = 0.0
            rhs[:, -1] = 0.0 if far_rhs is None else np.broadcast_to(np.asarray(far_rhs, float), (ni,))
        else:
            rhs[:, 0] = 0.0 if wall_rhs is None else np.broadcast_to(np.asarray(wall_rhs, float), (ni,))
            rhs[:, -1] = 0.0
        sol = self._lu.solve(rhs.ravel())
        return sol.reshape(ni, nj)


class StreamfunctionSolver(PoissonSolver):

    def __init__(self, lap: LaplacianOperator):
        super().__init__(lap, 'streamfunction')

    def solve(self, omega: np.ndarray, far_psi: np.ndarray) -> np.ndarray:
        return super().solve(-np.asarray(omega, float), far_rhs=far_psi)


class PressureSolver(PoissonSolver):

    def __init__(self, lap: LaplacianOperator):
        super().__init__(lap, 'pressure')

    def solve(self, src: np.ndarray, wall_grad: np.ndarray) -> np.ndarray:
        return super().solve(np.asarray(src, float), wall_rhs=wall_grad)


_RK3_IMAG = 1.5


def fit_circle_center(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.column_stack([2.0 * x, 2.0 * y, -np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = (sol[0], sol[1])
    r = float(np.sqrt(np.mean((x - cx) ** 2 + (y - cy) ** 2)))
    return (float(cx), float(cy), r)


def wall_vorticity(psi: np.ndarray, alpha_wall: np.ndarray, order: int=2) -> np.ndarray:
    if order <= 1:
        return -2.0 * psi[:, 1] / alpha_wall
    return -(4.0 * psi[:, 1] - 0.5 * psi[:, 2]) / alpha_wall


def velocity_from_psi(psi: np.ndarray, m) -> tuple[np.ndarray, ...]:
    d_xi = _d_xi(psi)
    d_eta = _d_eta(psi)
    u = (m.x_xi * d_eta - m.x_eta * d_xi) / m.J
    v = (m.y_xi * d_eta - m.y_eta * d_xi) / m.J
    return (u, v, d_eta, -d_xi)


def solve_ns(X, Y, Re: float, *, alpha_deg: float=0.0, Uinf: float=1.0, max_steps: int=1000000, cfl: float=0.9, dt_max: float=0.04, t_end: float | None=None, steady_tol: float=0.0002, steady_window: float=10.0, steady_min_t: float=15.0, pressure_every: int=4, L_ref: float | None=None, wall_order: int=2, upwind_blend: float=0.0, implicit_diffusion: bool=False, refactor_every: int=150, max_wall_time: float | None=None, verbose: bool=False, label: str='') -> dict:
    t_wall0 = time.time()
    X = np.array(X, float, copy=True)
    Y = np.array(Y, float, copy=True)
    ni, nj = X.shape
    if L_ref is None:
        L_ref = float(max(np.ptp(X[:, 0]), np.ptp(Y[:, 0])))
    X /= L_ref
    Y /= L_ref
    nu = Uinf / Re
    lap = LaplacianOperator(X, Y, periodic_i=True)
    m = lap.metrics
    psi_solver = PoissonSolver(lap, 'streamfunction_neumann')
    p_solver = PoissonSolver(lap, 'pressure')
    invJ = lap.invJ
    J = m.J
    alpha = m.alpha
    gamma = m.gamma
    cx, cy, r_far = fit_circle_center(X[:, -1], Y[:, -1])
    Xf, Yf = (X[:, -1], Y[:, -1])
    rf = np.hypot(Xf - cx, Yf - cy)
    a = np.deg2rad(alpha_deg)
    far_g = Uinf * (np.cos(a) * (Yf - cy) / rf - np.sin(a) * (Xf - cx) / rf)
    Jf_xi = 0.5 * (J + np.roll(J, -1, axis=0))
    alpha_f_xi = 0.5 * (alpha + np.roll(alpha, -1, axis=0))
    Jf_eta = 0.5 * (J[:, :-1] + J[:, 1:])
    gamma_f_eta = 0.5 * (gamma[:, :-1] + gamma[:, 1:])
    blend = float(upwind_blend)
    diff_rate_unit = 4.0 * (alpha + gamma) * invJ ** 2

    def adv_rate(U, V):
        return (np.abs(U) + np.abs(V)) * np.abs(invJ)

    def compute_dt(U, V):
        rmax = float(adv_rate(U, V).max())
        if not np.isfinite(rmax) or rmax <= 0:
            return dt_max
        dt_adv = cfl * _RK3_IMAG / rmax
        if implicit_diffusion:
            return float(min(dt_adv, dt_max))
        rdiff = nu * float(diff_rate_unit.max())
        dt_diff = 0.9 * 2.2 / max(rdiff, 1e-30)
        return float(min(dt_adv, dt_diff, dt_max))

    def advection(omega, U, V):
        Uf = 0.5 * (U + np.roll(U, -1, axis=0))
        om_f = 0.5 * (omega + np.roll(omega, -1, axis=0))
        if blend > 0.0:
            pe = np.abs(Uf) * np.abs(Jf_xi) / np.maximum(nu * alpha_f_xi, 1e-300)
            w = blend * np.clip(1.0 - 2.0 / np.maximum(pe, 1e-30), 0.0, 1.0)
            om_up = np.where(Uf > 0, omega, np.roll(omega, -1, axis=0))
            om_f = (1.0 - w) * om_f + w * om_up
        F = Uf * om_f
        div = F - np.roll(F, 1, axis=0)
        Vf = 0.5 * (V[:, :-1] + V[:, 1:])
        om_f2 = 0.5 * (omega[:, :-1] + omega[:, 1:])
        if blend > 0.0:
            pe = np.abs(Vf) * np.abs(Jf_eta) / np.maximum(nu * gamma_f_eta, 1e-300)
            w = blend * np.clip(1.0 - 2.0 / np.maximum(pe, 1e-30), 0.0, 1.0)
            om_up2 = np.where(Vf > 0, omega[:, :-1], omega[:, 1:])
            om_f2 = (1.0 - w) * om_f2 + w * om_up2
        G = Vf * om_f2
        div[:, 1:-1] += G[:, 1:] - G[:, :-1]
        return -div * invJ

    def rk3_advection(omega, U, V, dt):
        w0 = omega
        k1 = advection(w0, U, V)
        w1 = w0 + dt * k1
        w1[:, 0] = w0[:, 0]
        w1[:, -1] = 0.0
        k2 = advection(w1, U, V)
        w2 = 0.75 * w0 + 0.25 * (w1 + dt * k2)
        w2[:, 0] = w0[:, 0]
        w2[:, -1] = 0.0
        k3 = advection(w2, U, V)
        out = 1.0 / 3.0 * w0 + 2.0 / 3.0 * (w2 + dt * k3)
        out[:, 0] = w0[:, 0]
        out[:, -1] = 0.0
        return out
    omega = np.zeros((ni, nj))
    psi = psi_solver.solve(-omega, far_rhs=far_g)
    u, v, U, V = velocity_from_psi(psi, m)
    omega[:, 0] = wall_vorticity(psi, alpha[:, 0], wall_order)
    omega[:, -1] = 0.0
    Gamma = 0.0
    t, steps = (0.0, 0)
    t_hist, cd_hist, cl_hist, res_hist = ([], [], [], [])
    p = np.zeros_like(X)
    if t_end is None:
        t_end = np.inf
    dt = compute_dt(U, V)
    helm = None
    if implicit_diffusion:
        helm = PoissonSolver(lap, 'helmholtz', dt=dt, nu=nu)
    n_refactor = 1 if implicit_diffusion else 0
    Fx_p = Fy_p = Fx_mu = Fy_mu = 0.0
    converged = steady = False
    last_wall = omega[:, 0].copy()
    while steps < max_steps and t < t_end:
        dt_target = compute_dt(U, V)
        if implicit_diffusion:
            if dt_target < 0.7 * dt or (steps % refactor_every == 0 and dt_target > 1.25 * dt):
                dt = dt_target
                helm = PoissonSolver(lap, 'helmholtz', dt=dt, nu=nu)
                n_refactor += 1
            dt_use = dt
        else:
            dt_use = dt_target
        wadv = rk3_advection(omega, U, V, dt_use)
        if implicit_diffusion:
            wnew = helm.solve(wadv, wall_rhs=omega[:, 0])
        else:
            wnew = wadv + dt_use * nu * lap.apply(wadv)
            wnew[:, 0] = omega[:, 0]
            wnew[:, -1] = 0.0
        res = float(np.max(np.abs(wnew[:, 1:-1] - omega[:, 1:-1])) / (dt_use * max(np.max(np.abs(wnew)), 1e-30)))
        omega = wnew
        t += dt_use
        steps += 1
        Gamma = circulation(omega, J)
        psi = psi_solver.solve(-omega, far_rhs=far_g)
        u, v, U, V = velocity_from_psi(psi, m)
        omega[:, 0] = wall_vorticity(psi, alpha[:, 0], wall_order)
        omega[:, -1] = 0.0
        if steps % pressure_every == 0:
            u_x = (_d_xi(u) * m.y_eta - _d_eta(u) * m.y_xi) / J
            u_y = (m.x_xi * _d_eta(u) - m.x_eta * _d_xi(u)) / J
            v_x = (_d_xi(v) * m.y_eta - _d_eta(v) * m.y_xi) / J
            v_y = (m.x_xi * _d_eta(v) - m.x_eta * _d_xi(v)) / J
            src = -(u_x ** 2 + 2.0 * u_y * v_x + v_y ** 2)
            q_wall = wall_pressure_gradient(omega[:, 0], m, Re)
            p = p_solver.solve(src, wall_rhs=q_wall)
            Fx_p, Fy_p = pressure_force(p[:, 0], m)
            Fx_mu, Fy_mu = friction_force(omega[:, 0], m, Re)
            cd, cl = force_coefficients(np.array([Fx_p + Fx_mu, Fy_p + Fy_mu]), alpha_deg)
            t_hist.append(t)
            cd_hist.append(cd)
            cl_hist.append(cl)
            res_hist.append(res)
        if steps % 50 == 0 and t > steady_min_t and (len(t_hist) >= 20):
            tt = np.asarray(t_hist)
            idx = int(np.searchsorted(tt, t - steady_window))
            if len(tt) - idx >= 10:
                cdv = np.asarray(cd_hist[idx:])
                clv = np.asarray(cl_hist[idx:])
                tol = max(steady_tol, 1e-06 * abs(float(np.mean(cdv))))
                if np.ptp(cdv) < tol and np.ptp(clv) < tol:
                    steady = converged = True
                    break
        if verbose and steps % 500 == 0:
            cd_now = cd_hist[-1] if cd_hist else float('nan')
            cl_now = cl_hist[-1] if cl_hist else float('nan')
            print(f'  [{label}] step={steps} t={t:.2f} dt={dt_use:.2e} Cd={cd_now:.5f} Cl={cl_now:+.5f} res={res:.2e}', flush=True)
        if max_wall_time is not None and steps % 200 == 0 and (time.time() - t_wall0 > max_wall_time):
            break
    t_arr = np.asarray(t_hist)
    cd_arr = np.asarray(cd_hist)
    cl_arr = np.asarray(cl_hist)
    if len(t_arr) >= 8:
        t_start = float(t_arr[0] + 0.55 * (t_arr[-1] - t_arr[0]))
        avg = time_average_history(t_arr, cd_arr, cl_arr, t_start=t_start)
        st_info = strouhal_from_zero_crossings(t_arr[t_arr >= t_start], cl_arr[t_arr >= t_start])
        cd_mean, cl_mean, cl_rms = (avg['cd_mean'], avg['cl_mean'], avg['cl_rms'])
        St = float(st_info['St'])
        n_per = st_info['n_periods']
        unsteady = not steady and bool(np.ptp(cl_arr[t_arr >= t_start]) > 1e-06)
    else:
        t_start = t
        cd_mean = cl_mean = cl_rms = St = float('nan')
        n_per = 0
        unsteady = False
    cp_wall = cp_from_pressure(p[:, 0], Uinf)
    theta = np.arctan2(Y[:, 0] - cy, X[:, 0] - cx)
    return dict(u=u, v=v, p=p, omega=omega, psi=psi, Cp=cp_wall, Cd=float(cd_mean), Cl=float(cl_mean), Cd_inst=float(cd_arr[-1]) if cd_arr.size else float('nan'), Cl_inst=float(cl_arr[-1]) if cl_arr.size else float('nan'), converged=bool(converged), steady=bool(steady), unsteady=bool(unsteady), history=dict(t=t_arr, Cd=cd_arr, Cl=cl_arr, residual=np.asarray(res_hist)), Re=float(Re), alpha_deg=float(alpha_deg), n_steps=int(steps), dt=float(dt), t_end=float(t), L_ref=float(L_ref), mesh=(int(ni), int(nj)), X=X, Y=Y, theta=theta, cp_wall=cp_wall, p_wall=p[:, 0], omega_wall=omega[:, 0], St=St, n_periods=int(n_per), mean_window=(float(t_start), float(t)), Cd_mean=float(cd_mean), Cl_mean=float(cl_mean), Cl_rms=float(cl_rms), Gamma=float(Gamma), Fx_p=float(Fx_p), Fy_p=float(Fy_p), Fx_mu=float(Fx_mu), Fy_mu=float(Fy_mu), nu=float(nu), r_far=float(r_far), far_center=(cx, cy), n_refactor=int(n_refactor), wall_time=float(time.time() - t_wall0), upwind_blend=blend, implicit_diffusion=bool(implicit_diffusion), max_steps=int(max_steps))


__all__ = ['wall_geometry', 'wall_pressure_gradient', 'pressure_force', 'friction_force', 'force_coefficients', 'moment_coefficient', 'cp_from_pressure', 'circulation', 'strouhal_from_zero_crossings', 'period_from_history', 'time_average_history']


def wall_geometry(metrics, j: int=0) -> dict:
    x_xi = metrics.x_xi[:, j]
    y_xi = metrics.y_xi[:, j]
    return dict(x_xi=x_xi, y_xi=y_xi, ds=np.sqrt(x_xi ** 2 + y_xi ** 2), nx=y_xi / np.sqrt(x_xi ** 2 + y_xi ** 2), ny=-x_xi / np.sqrt(x_xi ** 2 + y_xi ** 2))


def wall_pressure_gradient(omega_wall: np.ndarray, metrics, re: float) -> np.ndarray:
    om = np.asarray(omega_wall, float)
    dxi = 0.5 * (np.roll(om, -1) - np.roll(om, 1))
    gamma = metrics.x_xi[:, 0] ** 2 + metrics.y_xi[:, 0] ** 2
    return -dxi / (np.sqrt(gamma) * re)


def pressure_force(p_wall: np.ndarray, metrics) -> np.ndarray:
    g = wall_geometry(metrics)
    f = -np.array([np.sum(p_wall * g['y_xi']), np.sum(p_wall * -g['x_xi'])])
    return f


def friction_force(omega_wall: np.ndarray, metrics, re: float) -> np.ndarray:
    g = wall_geometry(metrics)
    return np.array([np.sum(omega_wall * g['x_xi']), np.sum(omega_wall * g['y_xi'])]) / re


def force_coefficients(force: np.ndarray, alpha_deg: float=0.0) -> tuple[float, float]:
    a = np.deg2rad(alpha_deg)
    fx, fy = (float(force[0]), float(force[1]))
    cd = 2.0 * (fx * np.cos(a) + fy * np.sin(a))
    cl = 2.0 * (-fx * np.sin(a) + fy * np.cos(a))
    return (cd, cl)


def moment_coefficient(p_wall: np.ndarray, omega_wall: np.ndarray, X: np.ndarray, Y: np.ndarray, metrics, re: float, pivot=(0.25, 0.0)) -> float:
    g = wall_geometry(metrics)
    x = np.asarray(X, float)[:, 0] - pivot[0]
    y = np.asarray(Y, float)[:, 0] - pivot[1]
    dFx = -np.asarray(p_wall, float) * g['y_xi'] + np.asarray(omega_wall, float) * g['x_xi'] / re
    dFy = np.asarray(p_wall, float) * g['x_xi'] + np.asarray(omega_wall, float) * g['y_xi'] / re
    mz = np.sum(x * dFy - y * dFx)
    return float(-2.0 * mz)


def cp_from_pressure(p: np.ndarray, u_inf: float=1.0) -> np.ndarray:
    return 2.0 * np.asarray(p, float) / u_inf ** 2


def circulation(omega: np.ndarray, jac: np.ndarray) -> float:
    return float(np.sum(np.asarray(omega, float) * np.abs(jac)))


def _zero_crossings(t: np.ndarray, y: np.ndarray, up: bool=True) -> np.ndarray:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if up:
        idx = np.where((y[:-1] < 0.0) & (y[1:] >= 0.0))[0]
    else:
        idx = np.where((y[:-1] > 0.0) & (y[1:] <= 0.0))[0]
    if idx.size == 0:
        return np.array([])
    y0, y1 = (y[idx], y[idx + 1])
    frac = -y0 / np.where(np.abs(y1 - y0) > 0, y1 - y0, 1e-300)
    return t[idx] + frac * (t[idx + 1] - t[idx])


def period_from_history(t, y) -> dict:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y = y - np.mean(y)
    tc = _zero_crossings(t, y, up=True)
    if tc.size < 2:
        return dict(period=float('nan'), freq=0.0, n_periods=0, periods=np.array([]))
    periods = np.diff(tc)
    return dict(period=float(np.mean(periods)), freq=float(1.0 / np.mean(periods)), n_periods=int(periods.size), periods=periods)


def strouhal_from_zero_crossings(t, cl, u_inf: float=1.0, length: float=1.0) -> dict:
    info = period_from_history(t, cl)
    st = info['freq'] * length / u_inf
    info['St'] = float(st)
    return info


def time_average_history(t, cd, cl, *, t_start: float) -> dict:
    t = np.asarray(t, float)
    cd = np.asarray(cd, float)
    cl = np.asarray(cl, float)
    m = t >= t_start
    if np.count_nonzero(m) < 4:
        m = np.ones_like(t, bool)
    tt, cdd, cll = (t[m], cd[m], cl[m])
    per = period_from_history(tt, cll)
    return dict(cd_mean=float(np.mean(cdd)), cl_mean=float(np.mean(cll)), cl_rms=float(np.sqrt(np.mean((cll - np.mean(cll)) ** 2))), cd_rms=float(np.sqrt(np.mean((cdd - np.mean(cdd)) ** 2))), cl_amp=float(0.5 * (np.max(cll) - np.min(cll))), period=per['period'], St=per['freq'], n_periods=per['n_periods'], t_start=float(tt[0]), t_end=float(tt[-1]))




def read_airfoil(name):
    s = str(name).upper().replace('NACA', '').strip()
    if len(s) == 4 and s.isdigit():
        m = int(s[0]) / 100.0
        p = int(s[1]) / 10.0
        t = int(s[2:]) / 100.0
    else:
        m, p, t = 0.0, 0.4, 0.12
    beta = np.linspace(0.0, np.pi, 160)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 +
                    0.2843 * x ** 3 - 0.1015 * x ** 4)
    if m > 0.0:
        yc = np.where(x < p, m / p ** 2 * (2.0 * p * x - x ** 2),
                      m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2))
        dyc = np.where(x < p, 2.0 * m / p ** 2 * (p - x),
                       2.0 * m / (1.0 - p) ** 2 * (p - x))
        th = np.arctan(dyc)
        xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
        xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    else:
        xu, yu = x, yt
        xl, yl = x, -yt
    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    return np.vstack([upper, lower])


def cylinder_field(X, Y, R=1.0, V=1.0):
    return cylinder_analytic(X, Y, R=R, V=V)


def surface_geometry(name):
    pts = normalize(read_airfoil(name))
    upper, lower = split_surfaces(pts)
    upper = np.array(upper, float, copy=True)
    lower = np.array(lower, float, copy=True)
    upper[:, 1] -= upper[-1, 1] * upper[:, 0]
    lower[:, 1] -= lower[-1, 1] * lower[:, 0]
    return upper, lower


def closed_coords(upper, lower):
    coords = np.vstack([upper[::-1], lower[1:]])
    coords[-1] = coords[0]
    return coords


def _surface_normals(pts):
    x, y = pts[:, 0], pts[:, 1]
    tx, ty = np.gradient(x), np.gradient(y)
    norm = np.maximum(np.hypot(tx, ty), 1e-12)
    tx, ty = tx / norm, ty / norm
    n = np.column_stack([-ty, tx])
    if np.median(y) >= 0.0:
        if np.median(n[:, 1]) < 0.0:
            n = -n
    else:
        if np.median(n[:, 1]) > 0.0:
            n = -n
    return n


def _tangent(pts):
    x, y = pts[:, 0], pts[:, 1]
    tx, ty = np.gradient(x), np.gradient(y)
    norm = np.maximum(np.hypot(tx, ty), 1e-12)
    return np.column_stack([tx / norm, ty / norm])


def _displace_surface(pts, delta, s_delta, cap=0.008, taper_te=0.88):
    s = arclength(pts)
    d = np.interp(s, s_delta, delta)
    d = np.clip(d, 0.0, cap)
    d = _smooth_arclen(s, d, n_uniform=301, window=9)
    if d.size > 10:
        try:
            d = np.maximum(UnivariateSpline(s, d, s=len(s) * 1e-6)(s), 0.0)
        except Exception:
            pass
    if taper_te < 1.0:
        w = np.clip((1.0 - pts[:, 0]) / (1.0 - taper_te), 0.0, 1.0)
        d = d * w
    n = _surface_normals(pts)
    return np.column_stack([pts[:, 0], pts[:, 1] + d * n[:, 1]])


def _fill_nan_constant(y):
    y = np.array(y, float, copy=True)
    if np.all(np.isnan(y)):
        return np.zeros_like(y)
    idx = np.where(~np.isnan(y))[0]
    y[idx[-1] + 1:] = y[idx[-1]]
    y[:idx[0]] = y[idx[0]]
    return y


def _smooth(y, n=5):
    y = np.asarray(y, float)
    if y.size < n + 2:
        return y
    kernel = np.ones(n) / n
    pad = np.concatenate((y[:2][::-1], y, y[-2:][::-1]))
    out = np.convolve(pad, kernel, mode="valid")
    out[:2] = y[:2]
    out[-2:] = y[-2:]
    return out


def _friction_drag(bl, Ue):
    s = np.asarray(bl["s"], float)
    Cf = np.asarray(bl["Cf"], float)
    Ue = np.asarray(Ue, float)
    if s.size < 2:
        return 0.0
    good = np.isfinite(Cf) & np.isfinite(Ue)
    return float(np.trapezoid(np.where(good, Cf * Ue ** 2, 0.0), s))


def _displacement_drag(pts, inv_side, delta):
    s = arclength(pts)
    Cp = np.interp(s, inv_side["s"], _smooth(inv_side["Cp"]))
    dCp = np.gradient(Cp, s)
    tx = _tangent(pts)[:, 0]
    d = np.interp(s, s, _fill_nan_constant(delta))
    return float(np.trapezoid(d * dCp * tx, s))


def _taper(delta, pts, taper_te):
    if taper_te >= 1.0:
        return delta
    w = np.clip((1.0 - pts[:, 0]) / (1.0 - taper_te), 0.0, 1.0)
    return delta * w


def _side_from_panel(res, which):
    d = res[which]
    x = np.asarray(d["x"], float)
    y = np.asarray(d["y"], float)
    Cp = np.asarray(d["Cp"], float)
    Ue = np.abs(np.asarray(d["Ue"], float))
    if x[0] > x[-1]:
        x, y, Cp, Ue = x[::-1], y[::-1], Cp[::-1], Ue[::-1]
    s = arclength(np.column_stack([x, y]))
    return dict(s=s, x=x, y=y, Cp=Cp, Ue=Ue)


def viscous_case(alpha, name="NACA0012", Re=4.0e5, n_iter=3, n_panel=201,
                 relax=0.7, delta_cap=0.008, taper_te=0.88):
    base_upper, base_lower = surface_geometry(name)
    s_u = arclength(base_upper)
    s_l = arclength(base_lower)
    du = np.zeros_like(s_u)
    dl = np.zeros_like(s_l)
    du_prev = np.zeros_like(s_u)
    dl_prev = np.zeros_like(s_l)
    coords = None
    res = bl_u = bl_l = None
    upper = lower = None
    for it in range(n_iter):
        res = solve_panel(alpha, coords if coords is not None else name, n_panel)
        upper = _side_from_panel(res, "upper")
        lower = _side_from_panel(res, "lower")
        bl_u = solve_bl(upper["s"], upper["Ue"], Re, side="upper",
                        H_sep_lam=3.5, H_sep_turb=2.4, n_sub=8,
                        transition_at_separation=True, transition_window=0.02)
        bl_l = solve_bl(lower["s"], lower["Ue"], Re, side="lower",
                        H_sep_lam=3.5, H_sep_turb=2.4, n_sub=8,
                        transition_at_separation=True, transition_window=0.02)
        if it == n_iter - 1:
            break
        dsu = np.interp(s_u, upper["s"],
                        _fill_nan_constant(np.asarray(bl_u["delta_star"], float)))
        dsl = np.interp(s_l, lower["s"],
                        _fill_nan_constant(np.asarray(bl_l["delta_star"], float)))
        du = relax * dsu + (1.0 - relax) * du_prev
        dl = relax * dsl + (1.0 - relax) * dl_prev
        du_prev, dl_prev = du, dl
        up_d = _displace_surface(base_upper, du, s_u, cap=delta_cap,
                                 taper_te=taper_te)
        lo_d = _displace_surface(base_lower, dl, s_l, cap=delta_cap,
                                 taper_te=taper_te)
        up_d[0] = lo_d[0] = 0.5 * (up_d[0] + lo_d[0])
        up_d[-1] = lo_d[-1] = 0.5 * (up_d[-1] + lo_d[-1])
        coords = closed_coords(up_d, lo_d)
    upper = _side_from_panel(res, "upper")
    lower = _side_from_panel(res, "lower")
    inv = solve_panel(alpha, name, n_panel)
    inv_u = _side_from_panel(inv, "upper")
    inv_l = _side_from_panel(inv, "lower")
    dsu = np.clip(du, 0.0, delta_cap) if it > 0 else np.zeros_like(s_u)
    dsl = np.clip(dl, 0.0, delta_cap) if it > 0 else np.zeros_like(s_l)
    Cd_pres = (_displacement_drag(base_upper, inv_u,
                                  _taper(dsu, base_upper, taper_te)) +
               _displacement_drag(base_lower, inv_l,
                                  _taper(dsl, base_lower, taper_te)))
    Cd_fric = (_friction_drag(bl_u, upper["Ue"]) +
               _friction_drag(bl_l, lower["Ue"]))
    return dict(alpha=float(alpha), Re=float(Re), Cl=float(res["Cl"]),
                Cd=float(Cd_fric + Cd_pres), Cm=float(res["Cm"]),
                Cd_friction=float(Cd_fric), Cd_pressure=float(Cd_pres),
                Cl_inviscid=float(inv["Cl"]), Cm_inviscid=float(inv["Cm"]),
                x_tr_upper=float(np.interp(bl_u["x_tr"], upper["s"],
                                           upper["x"]))
                if np.isfinite(bl_u["x_tr"]) else float("nan"),
                x_tr_lower=float(np.interp(bl_l["x_tr"], lower["s"],
                                           lower["x"]))
                if np.isfinite(bl_l["x_tr"]) else float("nan"),
                upper=upper, lower=lower, bl_upper=bl_u, bl_lower=bl_l,
                Cp_panel=res["Cp"], x_panel=res["x"], y_panel=res["y"])


def polar(alpha_list, name="NACA0012", Re=4.0e5, **kw):
    rows = []
    for a in alpha_list:
        rows.append(viscous_case(a, name=name, Re=Re, **kw))
    return rows





def main():
    alphas = np.array([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    rows = polar(alphas, "NACA0012", 4.0e5)
    print("NACA0012  Re = 4e5")
    print(" alpha      Cl       Cd       Cm    x_tr_u  x_tr_l")
    for r in rows:
        print(f"{r['alpha']:6.1f} {r['Cl']:8.4f} {r['Cd']:8.5f} "
              f"{r['Cm']:8.4f} {r['x_tr_upper']:7.3f} {r['x_tr_lower']:7.3f}")

    cl = np.array([r["Cl"] for r in rows])
    cd = np.array([r["Cd"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].plot(alphas, cl, "o-", color="#1f4e79", lw=1.2, ms=3)
    axes[0].set_xlabel("alpha / deg")
    axes[0].set_ylabel("Cl")
    axes[0].grid(alpha=0.3)
    axes[1].plot(alphas, cd, "s-", color="#c00000", lw=1.2, ms=3)
    axes[1].set_xlabel("alpha / deg")
    axes[1].set_ylabel("Cd")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("polar.png", dpi=200)
    plt.close(fig)

    case = viscous_case(4.0, "NACA0012", 4.0e5)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for side, c in (("upper", "#c00000"), ("lower", "#1f4e79")):
        d = case[side]
        ax.plot(d["x"], d["Cp"], "-", color=c, lw=1.2, label=side)
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("airfoil_cp.png", dpi=200)
    plt.close(fig)

    x = np.linspace(-4.0, 4.0, 401)
    y = np.linspace(-4.0, 4.0, 401)
    X, Y = np.meshgrid(x, y)
    pot = cylinder_field(X, Y, R=1.0, V=1.0)
    Cp = np.ma.masked_where(X ** 2 + Y ** 2 < 1.0, pot["Cp"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lv = np.linspace(-3.0, 1.0, 31)
    cf = ax.contourf(X, Y, Cp, levels=lv, cmap="RdYlBu_r", extend="both")
    fig.colorbar(cf, ax=ax, label="Cp")
    ax.add_patch(plt.Circle((0, 0), 1.0, color="w", ec="k", zorder=5))
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    fig.tight_layout()
    fig.savefig("cylinder_potential.png", dpi=200)
    plt.close(fig)

    Xc, Yc = cylinder_o_grid(120, 60, 25.0, 1.04, 0.45)
    out = solve_ns(Xc, Yc, 40.0, t_end=60.0)
    print(f"cylinder Re=40  Cd={out['Cd']:.4f}  Cl={out['Cl']:+.4f}  "
          f"steps={out['n_steps']}")
    cp = 2.0 * out["p"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    m = (Xc < -4.0) | (Xc > 7.0) | (np.abs(Yc) > 5.0)
    cf = axes[0].contourf(Xc, Yc, np.ma.masked_where(m, cp),
                          levels=np.linspace(-2.4, 1.0, 31),
                          cmap="RdYlBu_r", extend="both")
    fig.colorbar(cf, ax=axes[0], label="Cp")
    axes[0].add_patch(plt.Circle((0, 0), 1.0, color="w", ec="k", zorder=5))
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-3, 5)
    axes[0].set_ylim(-3, 3)
    ps = np.ma.masked_where(m, out["psi"])
    axes[1].contour(Xc, Yc, ps, levels=31, colors="k", linewidths=0.5)
    axes[1].add_patch(plt.Circle((0, 0), 1.0, color="w", ec="k", zorder=5))
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-3, 5)
    axes[1].set_ylim(-3, 3)
    fig.tight_layout()
    fig.savefig("cylinder_ns.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
