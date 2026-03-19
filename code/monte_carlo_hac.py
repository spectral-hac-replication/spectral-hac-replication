"""
Monte Carlo Evidence for Path-Averaged HAC Estimators
============================================================

Aligned with revised manuscript (final version for referee):

  SECTION 3 (fixed weights):
    * Proposition (defect-density criterion, continuous)
    * Proposition (standard lag windows: Bartlett, Parzen, Rectangular)
    * Proposition (discrete transfer principle)

  SECTION 4 (time-varying bandwidths — central section):
    * Lemma  (mixture-convexity preservation)       [lem:mixture_convexity]
    * Theorem (PSD for path-averaged Bartlett)       [thm:psd_pathavg]
    * Proposition (Non-PSD for hard-cutoff avg)      [prop:hardcutoff_nonPSD]
    * Proposition (Discrepancy bound)                [prop:form_comparison]
    * Corollary (Asymptotic equivalence of forms)    [cor:form_equivalence]
    * Proposition (Pointwise domination)             [prop:optimal_taper]
    * Proposition (Consistency)                      [prop:consistency]
    * Lemma  (Sharp MSE for weighted autocov sums)   [lem:sharp_mse]
      - requires (A2') E||s_t||^4 < infty
      - Gaussian: O(M/T);  General: O(M^2/T) from cumulant spectral density
    * Proposition (Convergence rate)                 [prop:rate]
      - beta in (1,2); rate T^{-(beta-1)/(2*beta)}
      - Gaussian refinement: T^{-(beta-1)/(2*beta-1)}
    * Corollary (Wald test validity)                 [cor:wald]
    * Proposition (Eigenvalue perturbation bound)    [prop:perobs_eigenvalue]
    * Corollary (Eventual PSD for per-obs form)      [cor:perobs_eventual_psd]
      - requires ||s_t|| <= B, 3B^2 limsup(Delta_m) < lambda_min(Omega)

  Monte Carlo design:
    * E0:       Fixed hard-cutoff rectangular baseline
    * E1:       Bartlett fixed-bandwidth (Newey-West style)
    * E2:       Andrews-style QS adaptive bandwidth (benchmark)
    * E3h/E3s:  Ad hoc state-dependent (per-observation form)
    * E3h_avg/E3s_avg: Ad hoc path-averaged forms
    * E4h/E4s:  Plug-in-anchored state-dependent (per-observation form)
    * E4h_avg/E4s_avg: Plug-in-anchored path-averaged forms

  Key empirical contrasts:
    1. E3h vs E3s:     same {m_t}, hard cutoff vs taper
    2. E4h vs E4s:     same {m_t}, plug-in-anchored family
    3. E3s_avg/E4s_avg: path-averaged Bartlett (Thm psd_pathavg -- provably PSD)
    4. E3h_avg/E4h_avg: path-averaged rectangular (Prop hardcutoff -- NOT PSD)
    5. Bandwidth-path variation analysis (tercile splits on Delta_m)
    6. PSD repair comparison
    7. Eventual-PSD diagnostic (Cor perobs_eventual_psd)
    8. Sharp MSE diagnostic (Lem sharp_mse)
    9. Optional multivariate stress test (bivariate MS-VAR)

  Tables generated (LaTeX):
    * Table_MC_mechanism.tex  -- Mechanism diagnostics (Table MC.1)
    * Table_MC_inference.tex  -- Inference diagnostics (Table MC.2)
    * Table_MC_variation.tex  -- Bandwidth-path variation (Table MC.3)
    * Table_MC_repair.tex     -- PSD repair comparison (Table MC.4)
    * Table_MC_multivariate.tex -- Multivariate stress test (Table MC.5)
    * Table_B1_certified_signchanges.tex -- Sign certification (Appendix)

  Figures generated:
    * Figure_MC1_structural_bridge.pdf
    * Figure_MC2_failures.pdf
    * Figure_MC3_inference.pdf
    * Figure_MC4_variation.pdf
    * Figure_MC5_repair.pdf
    * Figure_MC6_multivariate.pdf
    * Figure_B1_parameter_map.pdf
    * Figure_B2_certified.pdf
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from math import sqrt
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# NumPy compatibility shim
# ──────────────────────────────────────────────────────────────────────────────
if hasattr(np, "trapezoid"):
    _trapezoid = np.trapezoid
else:
    _trapezoid = np.trapz

# ──────────────────────────────────────────────────────────────────────────────
# Output directories
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Global settings
# ──────────────────────────────────────────────────────────────────────────────
SEED_BASE = 12345
T_LIST = [250, 500, 1000]
R_DICT = {250: 2000, 500: 2000, 1000: 1000}
OMEGA_GRID = np.linspace(0.0, np.pi, 257)   # J = 256
BETA_TRUE = np.array([1.0, 0.0, 0.0])
C0 = 1.5

# Full estimator list including E4 family (referee request)
EST_LIST = [
    "E0", "E1", "E2",
    "E3h", "E3s", "E3h_avg", "E3s_avg",
    "E4h", "E4s", "E4h_avg", "E4s_avg",
]

_EPS = 1e-12
_PI = np.pi
_2PI = 2.0 * np.pi
_Z975 = 1.959963984540054

COLORS = {
    "E0": "#9467bd",
    "E1": "#1f77b4",
    "E2": "#ff7f0e",
    "E3h": "#2ca02c",
    "E3s": "#d62728",
    "E3s_avg": "#e377c2",
    "E3h_avg": "#8c564b",
    "E4h": "#17becf",
    "E4s": "#bcbd22",
    "E4h_avg": "#7f7f7f",
    "E4s_avg": "#aec7e8",
}

# ══════════════════════════════════════════════════════════════════════════════
# 0. MONTE CARLO CI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def wilson_interval(k, n, z=_Z975):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / den
    half = z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def mc_mean_interval(arr, z=_Z975):
    x = np.asarray(arr, dtype=float)
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if n == 1:
        return m, np.nan, np.nan
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    return m, m - z * se, m + z * se


def mc_binom_summary(arr):
    x = np.asarray(arr, dtype=int)
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0, 0
    k = int(np.sum(x))
    p = k / n
    lo, hi = wilson_interval(k, n)
    return p, lo, hi, k, n


def add_binom_summary(target, key, arr):
    p, lo, hi, k, n = mc_binom_summary(arr)
    target[key] = p
    target[f"{key}_lo"] = lo
    target[f"{key}_hi"] = hi
    target[f"{key}_k"] = k
    target[f"{key}_n"] = n


def add_mean_summary(target, key, arr):
    m, lo, hi = mc_mean_interval(arr)
    target[key] = m
    target[f"{key}_lo"] = lo
    target[f"{key}_hi"] = hi


def conditional_binom(metric_arr, mask):
    metric_arr = np.asarray(metric_arr, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    sub = metric_arr[mask]
    p, lo, hi, _, n = mc_binom_summary(sub)
    return p, lo, hi, n


def fmt_prob_ci(p, lo, hi, digits=3):
    if np.isnan(p):
        return "---"
    return f"{p:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]"


def fmt_mean_ci(m, lo, hi, digits=4, sci_threshold=1e-3):
    if np.isnan(m):
        return "---"
    use_sci = (abs(m) > 0 and abs(m) < sci_threshold) or abs(m) >= 1e4
    if use_sci:
        return f"{m:.2e} [{lo:.2e}, {hi:.2e}]"
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA-GENERATING PROCESSES
# ══════════════════════════════════════════════════════════════════════════════

def simulate_DGP1(T, seed):
    """AR(1)-GARCH(1,1)."""
    rng = np.random.default_rng(seed)
    phi, omega_g, a, b = 0.6, 0.05, 0.10, 0.85
    u = np.zeros(T + 500)
    eps = np.zeros(T + 500)
    sigma2 = np.ones(T + 500) * omega_g / (1 - a - b)
    z = rng.standard_normal(T + 500)
    for t in range(1, T + 500):
        sigma2[t] = omega_g + a * eps[t - 1]**2 + b * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
        u[t] = phi * u[t - 1] + eps[t]
    return u[-T:]


def simulate_DGP2(T, seed):
    """Markov-switching AR(1)."""
    rng = np.random.default_rng(seed)
    p00, p11 = 0.97, 0.90
    phi = np.array([0.2, 0.8])
    sig = np.array([1.0, 3.0])
    s = np.zeros(T + 500, dtype=int)
    u = np.zeros(T + 500)
    z = rng.standard_normal(T + 500)
    for t in range(1, T + 500):
        p_stay = p11 if s[t - 1] == 1 else p00
        s[t] = s[t - 1] if rng.random() < p_stay else 1 - s[t - 1]
        u[t] = phi[s[t]] * u[t - 1] + sig[s[t]] * z[t]
    return u[-T:]


def simulate_DGP2_bivariate(T, seed):
    """Bivariate Markov-switching VAR for multivariate stress test."""
    rng = np.random.default_rng(seed)
    p00, p11 = 0.97, 0.90
    phi0 = np.array([[0.2, 0.1], [0.05, 0.15]])
    phi1 = np.array([[0.8, 0.15], [0.1, 0.7]])
    sig0 = np.array([[1.0, 0.3], [0.3, 1.0]])
    sig1 = np.array([[3.0, 0.9], [0.9, 3.0]])
    L0 = np.linalg.cholesky(sig0)
    L1 = np.linalg.cholesky(sig1)
    s = np.zeros(T + 500, dtype=int)
    u = np.zeros((T + 500, 2))
    z = rng.standard_normal((T + 500, 2))
    for t in range(1, T + 500):
        p_stay = p11 if s[t - 1] == 1 else p00
        s[t] = s[t - 1] if rng.random() < p_stay else 1 - s[t - 1]
        phi_t = phi1 if s[t] == 1 else phi0
        L_t = L1 if s[t] == 1 else L0
        u[t] = phi_t @ u[t - 1] + L_t @ z[t]
    return u[-T:]


def simulate_regressors(T, seed):
    rng = np.random.default_rng(seed)
    rho = 0.3
    x1 = np.zeros(T)
    x2 = np.zeros(T)
    e1, e2 = rng.standard_normal((2, T))
    for t in range(1, T):
        x1[t] = rho * x1[t - 1] + e1[t]
        x2[t] = rho * x2[t - 1] + e2[t]
    return np.column_stack([np.ones(T), x1, x2])


# ══════════════════════════════════════════════════════════════════════════════
# 2. OLS + SANDWICH
# ══════════════════════════════════════════════════════════════════════════════

def ols(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    uhat = y - X @ beta
    return beta, uhat


def sandwich_var(X, Omega_hat, T):
    XX = X.T @ X / T
    XXi = np.linalg.inv(XX)
    return XXi @ Omega_hat @ XXi / T


def symmetrize(A):
    return 0.5 * (A + A.T)


def PSD_project(Omega):
    Omega = symmetrize(Omega)
    vals, vecs = np.linalg.eigh(Omega)
    return vecs @ np.diag(np.maximum(vals, 0.0)) @ vecs.T


# ══════════════════════════════════════════════════════════════════════════════
# 3. KERNELS AND WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def bartlett_weights(m):
    k = np.arange(m + 1)
    return 1.0 - k / (m + 1)


def qs_kernel(x):
    x = np.asarray(x, dtype=float)
    scalar = (x.ndim == 0)
    x = np.atleast_1d(x)
    y = 6 * np.pi * x / 5
    w = np.ones_like(x)
    mask = np.abs(x) >= 1e-10
    y_m = y[mask]
    w[mask] = 3 / y_m**2 * (np.sin(y_m) / y_m - np.cos(y_m))
    return float(w[0]) if scalar else w


def parzen_weights(m):
    k = np.arange(m + 1)
    u = k / (m + 1)
    return np.where(u <= 0.5, 1.0 - 6.0 * u**2 + 6.0 * u**3, 2.0 * (1.0 - u)**3)


# ══════════════════════════════════════════════════════════════════════════════
# 4. STATE RULES (E3: ad hoc, E4: plug-in-anchored)
# ══════════════════════════════════════════════════════════════════════════════

def compute_state_mt_E3(uhat, T, m_min=2, m_max=None, kappa=1.0, eps=1e-8):
    """Ad hoc state rule for E3 family (residual volatility indicator)."""
    if m_max is None:
        m_max = max(m_min + 1, int(np.floor(2 * T**(1 / 3))))
    nu = np.log(eps + uhat**2)
    med = np.median(nu)
    mad = np.median(np.abs(nu - med)) + 1e-12
    nu_tilde = (nu - med) / mad
    g = expit(kappa * nu_tilde)
    mt = (m_min + np.floor((m_max - m_min) * g)).astype(int)
    return mt, m_min, m_max


def compute_automatic_bandwidth(scores):
    """Andrews (1991) plug-in bandwidth for QS kernel (scalar version)."""
    v = scores[:, 1] if scores.ndim > 1 else scores
    T = len(v)
    rho = np.clip(np.dot(v[1:], v[:-1]) / (np.dot(v[:-1], v[:-1]) + 1e-12), -0.99, 0.99)
    sig2 = np.var(v)
    alpha2 = 4 * rho**2 * sig2**2 / ((1 - rho)**8 + 1e-12) / (sig2**2 / ((1 - rho)**4 + 1e-12) + 1e-12)
    b_hat = max(1.0, 1.3221 * (alpha2 * T)**0.2)
    return b_hat


def compute_state_mt_E4(uhat, scores, T, kappa=1.0, eps=1e-8):
    """
    Plug-in-anchored state rule for E4 family.
    Centered on automatic bandwidth, then perturbed by state signal.
    This provides a more practice-oriented stress test than E3.
    """
    b_auto = compute_automatic_bandwidth(scores)
    m_center = max(2, int(np.floor(b_auto)))
    m_min = max(1, m_center - max(2, int(np.floor(0.5 * m_center))))
    m_max = m_center + max(2, int(np.floor(0.5 * m_center)))

    nu = np.log(eps + uhat**2)
    med = np.median(nu)
    mad = np.median(np.abs(nu - med)) + 1e-12
    nu_tilde = (nu - med) / mad
    g = expit(kappa * nu_tilde)
    mt = (m_min + np.floor((m_max - m_min) * g)).astype(int)
    return mt, m_min, m_max, m_center


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONVEXITY VERIFICATION (Lemma: mixture-convexity)
# ══════════════════════════════════════════════════════════════════════════════

def check_convexity(w):
    """Check if weight sequence is convex: Delta^2 w_k >= 0 for all k."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    if n < 3:
        return True, 0.0
    d2 = w[:-2] - 2.0 * w[1:-1] + w[2:]
    min_d2 = float(np.min(d2))
    return bool(min_d2 >= -1e-14), min_d2


def compute_averaged_weights(mt, T, kernel="bartlett"):
    """
    Compute path-averaged weights: w_bar_k = (1/T) sum_t K(k/(m_t+1)) 1_{|k|<=m_t}
    Theorem (PSD, thm:psd_pathavg) / Proposition (Non-PSD, prop:hardcutoff_nonPSD).
    """
    m_max = int(np.max(mt))
    w_bar = np.zeros(m_max + 2)
    for k in range(m_max + 1):
        mask = (mt >= k)
        if not mask.any():
            w_bar[k] = 0.0
            continue
        if kernel == "bartlett":
            wk = 1.0 - k / (mt[mask].astype(float) + 1.0)
        elif kernel == "rectangular":
            wk = np.ones(mask.sum())
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        w_bar[k] = float(np.sum(wk)) / T
    return w_bar


def verify_theorem41_explicit():
    """
    Verify the counterexample from Proposition (prop:hardcutoff_nonPSD):
    T=4, m_1=m_2=1, m_3=m_4=3.  Note T >= max(m_t) = 3.

    Lemma (lem:mixture_convexity) guarantees the Bartlett average is convex.
    The rectangular average is NOT convex => spectral window goes negative.
    """
    # T=4: two observations with m=1, two with m=3
    w_bart_1 = np.array([1.0, 0.5, 0.0, 0.0, 0.0])
    w_bart_3 = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    w_bart_avg = 0.5 * (w_bart_1 + w_bart_3)  # pi_1 = pi_3 = 0.5

    w_rect_1 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    w_rect_3 = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    w_rect_avg = 0.5 * (w_rect_1 + w_rect_3)

    bart_convex, bart_min_d2 = check_convexity(w_bart_avg)
    rect_convex, rect_min_d2 = check_convexity(w_rect_avg)

    omega_grid = np.linspace(0.01, np.pi, 1000)
    W_rect = np.array([
        w_rect_avg[0] + 2.0 * sum(w_rect_avg[k] * np.cos(om * k) for k in range(1, len(w_rect_avg)))
        for om in omega_grid
    ])
    W_bart = np.array([
        w_bart_avg[0] + 2.0 * sum(w_bart_avg[k] * np.cos(om * k) for k in range(1, len(w_bart_avg)))
        for om in omega_grid
    ])

    return {
        "w_bart_avg": w_bart_avg, "w_rect_avg": w_rect_avg,
        "bart_convex": bart_convex, "bart_min_d2": bart_min_d2,
        "rect_convex": rect_convex, "rect_min_d2": rect_min_d2,
        "rect_W_min": float(np.min(W_rect)),
        "rect_W_negative": float(np.min(W_rect)) < 0,
        "rect_omega_min": omega_grid[np.argmin(W_rect)],
        "bart_W_min": float(np.min(W_bart)),
        "bart_W_negative": float(np.min(W_bart)) < -1e-14,
        "omega_grid": omega_grid,
        "W_rect_curve": W_rect, "W_bart_curve": W_bart,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. HAC ESTIMATORS
# ══════════════════════════════════════════════════════════════════════════════

def compute_autocov(scores, max_lag):
    T = scores.shape[0]
    d = scores.shape[1] if scores.ndim > 1 else 1
    Gamma = []
    for k in range(max_lag + 1):
        if k == 0:
            G = scores.T @ scores / T
        else:
            G = scores[k:].T @ scores[:T - k] / T
        Gamma.append(G)
    return Gamma


def HAC_E0(scores, T, c0=C0):
    """Fixed hard-cutoff rectangular baseline."""
    m0 = max(1, int(np.floor(c0 * T**(1 / 3))))
    Gam = compute_autocov(scores, m0)
    Omega = Gam[0].copy()
    for k in range(1, m0 + 1):
        Omega += (Gam[k] + Gam[k].T)
    return Omega


def HAC_E1(scores, T, c0=C0):
    """Fixed-bandwidth Bartlett / Newey–West baseline."""
    m0 = max(1, int(np.floor(c0 * T**(1 / 3))))
    Gam = compute_autocov(scores, m0)
    wts = bartlett_weights(m0)
    Omega = wts[0] * Gam[0]
    for k in range(1, m0 + 1):
        Omega += wts[k] * (Gam[k] + Gam[k].T)
    return Omega


def HAC_E2(scores, T):
    """QS benchmark."""
    d = scores.shape[1]
    rhos = []
    for j in range(d):
        v = scores[:, j]
        rho = np.dot(v[1:], v[:-1]) / (np.dot(v[:-1], v[:-1]) + 1e-12)
        rhos.append(np.clip(rho, -0.99, 0.99))
    num, den = 0.0, 0.0
    for j, rho in enumerate(rhos):
        sig4 = np.var(scores[:, j])**2
        num += 4 * rho**2 * sig4 / ((1 - rho)**8 + 1e-12)
        den += sig4 / ((1 - rho)**4 + 1e-12)
    alpha2 = num / (den + 1e-12)
    b_hat = max(1.0, 1.3221 * (alpha2 * T)**0.2)
    max_lag = min(int(np.floor(b_hat)) + 1, T - 1)
    Gam = compute_autocov(scores, max_lag)
    Omega = qs_kernel(0) * Gam[0]
    for k in range(1, max_lag + 1):
        Omega += qs_kernel(k / b_hat) * (Gam[k] + Gam[k].T)
    return Omega


def _hac_perobs(scores, mt, T, kernel="rectangular"):
    """Generic per-observation HAC estimator."""
    d = scores.shape[1]
    m_max = int(np.max(mt))
    Omega = np.zeros((d, d))
    for k in range(-m_max, m_max + 1):
        absk = abs(k)
        valid_t = np.where(mt >= absk)[0]
        tk = valid_t - k
        in_range = (tk >= 0) & (tk < T)
        valid_t, tk = valid_t[in_range], tk[in_range]
        if len(valid_t) == 0:
            continue
        if kernel == "bartlett":
            wk = 1.0 - absk / (mt[valid_t].astype(float) + 1.0)
            Omega += (scores[valid_t] * wk[:, None]).T @ scores[tk]
        else:
            Omega += scores[valid_t].T @ scores[tk]
    return Omega / T


def _hac_avg(scores, mt, T, kernel="bartlett"):
    """Generic path-averaged HAC estimator (Algorithm 1; Theorem thm:psd_pathavg guarantees PSD for Bartlett)."""
    w_bar = compute_averaged_weights(mt, T, kernel=kernel)
    m_max = min(len(w_bar) - 1, T - 1)
    Gam = compute_autocov(scores, m_max)
    d = scores.shape[1]
    Omega = w_bar[0] * Gam[0]
    for k in range(1, min(len(w_bar), len(Gam))):
        if w_bar[k] > 1e-15:
            Omega += w_bar[k] * (Gam[k] + Gam[k].T)
    return Omega


# E3 family: ad hoc state rule
def HAC_E3h(scores, uhat, T):
    mt, _, _ = compute_state_mt_E3(uhat, T)
    return _hac_perobs(scores, mt, T, kernel="rectangular")

def HAC_E3s(scores, uhat, T):
    mt, _, _ = compute_state_mt_E3(uhat, T)
    return _hac_perobs(scores, mt, T, kernel="bartlett")

def HAC_E3h_avg(scores, uhat, T):
    mt, _, _ = compute_state_mt_E3(uhat, T)
    return _hac_avg(scores, mt, T, kernel="rectangular")

def HAC_E3s_avg(scores, uhat, T):
    mt, _, _ = compute_state_mt_E3(uhat, T)
    return _hac_avg(scores, mt, T, kernel="bartlett")

# E4 family: plug-in-anchored state rule (referee request)
def HAC_E4h(scores, uhat, T):
    mt, _, _, _ = compute_state_mt_E4(uhat, scores, T)
    return _hac_perobs(scores, mt, T, kernel="rectangular")

def HAC_E4s(scores, uhat, T):
    mt, _, _, _ = compute_state_mt_E4(uhat, scores, T)
    return _hac_perobs(scores, mt, T, kernel="bartlett")

def HAC_E4h_avg(scores, uhat, T):
    mt, _, _, _ = compute_state_mt_E4(uhat, scores, T)
    return _hac_avg(scores, mt, T, kernel="rectangular")

def HAC_E4s_avg(scores, uhat, T):
    mt, _, _, _ = compute_state_mt_E4(uhat, scores, T)
    return _hac_avg(scores, mt, T, kernel="bartlett")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SPECTRAL ESTIMATORS (UNIVARIATE)
# ══════════════════════════════════════════════════════════════════════════════

def _spectrum_from_weights(v, T, w_bar, max_lag):
    gam = np.array([np.dot(v[:T - k], v[k:]) / T for k in range(max_lag + 1)])
    ks = np.arange(1, max_lag + 1)
    fhat = np.array([
        (w_bar[0] * gam[0] + 2 * np.sum(w_bar[1:max_lag+1] * gam[1:] * np.cos(om * ks))) / (2 * np.pi)
        for om in OMEGA_GRID
    ])
    return fhat


def _spectrum_perobs(v, uhat, T, mt, kernel="rectangular"):
    m_max = int(np.max(mt))
    gam = np.zeros(m_max + 1)
    gam[0] = float(np.dot(v, v) / T)
    for k in range(1, m_max + 1):
        valid_t = np.where(mt >= k)[0]
        tk = valid_t - k
        in_rng = (tk >= 0) & (tk < T)
        valid_t, tk = valid_t[in_rng], tk[in_rng]
        if len(valid_t) == 0:
            continue
        if kernel == "bartlett":
            wk = 1.0 - k / (mt[valid_t].astype(float) + 1.0)
            gam[k] = float(np.dot(v[valid_t] * wk, v[tk]) / T)
        else:
            gam[k] = float(np.dot(v[valid_t], v[tk]) / T)
    ks = np.arange(1, m_max + 1)
    fhat = np.array([
        (gam[0] + 2 * np.sum(gam[1:] * np.cos(om * ks))) / (2 * np.pi)
        for om in OMEGA_GRID
    ])
    return fhat


def _spectrum_avg(v, T, w_bar):
    n_w = len(w_bar)
    gam = np.array([np.dot(v[:T - k], v[k:]) / T for k in range(n_w)])
    ks = np.arange(1, n_w)
    fhat = np.array([
        (w_bar[0] * gam[0] + 2 * np.sum(w_bar[1:] * gam[1:] * np.cos(om * ks))) / (2 * np.pi)
        for om in OMEGA_GRID
    ])
    return fhat


def compute_spectrum(est_name, v, uhat, scores, T):
    """Dispatch spectrum computation by estimator name."""
    if est_name == "E0":
        m0 = max(1, int(np.floor(C0 * T**(1 / 3))))
        return _spectrum_from_weights(v, T, np.ones(m0 + 1), m0)
    elif est_name == "E1":
        m0 = max(1, int(np.floor(C0 * T**(1 / 3))))
        return _spectrum_from_weights(v, T, bartlett_weights(m0), m0)
    elif est_name == "E2":
        rho = np.clip(np.dot(v[1:], v[:-1]) / (np.dot(v[:-1], v[:-1]) + 1e-12), -0.99, 0.99)
        sig2 = np.var(v)
        alpha2 = 4 * rho**2 * sig2**2 / ((1-rho)**8+1e-12) / (sig2**2/((1-rho)**4+1e-12)+1e-12)
        b = max(1.0, 1.3221 * (alpha2 * T)**0.2)
        m = min(int(np.floor(b)) + 1, T - 1)
        gam = np.array([np.dot(v[:T-k], v[k:]) / T for k in range(m + 1)])
        ks = np.arange(1, m + 1)
        wks = qs_kernel(ks / b)
        return np.array([(qs_kernel(0)*gam[0]+2*np.sum(wks*gam[1:]*np.cos(om*ks)))/(2*np.pi) for om in OMEGA_GRID])
    elif est_name.startswith("E3"):
        mt, _, _ = compute_state_mt_E3(uhat, T)
        if est_name == "E3h":
            return _spectrum_perobs(v, uhat, T, mt, "rectangular")
        elif est_name == "E3s":
            return _spectrum_perobs(v, uhat, T, mt, "bartlett")
        elif est_name == "E3h_avg":
            return _spectrum_avg(v, T, compute_averaged_weights(mt, T, "rectangular"))
        elif est_name == "E3s_avg":
            return _spectrum_avg(v, T, compute_averaged_weights(mt, T, "bartlett"))
    elif est_name.startswith("E4"):
        mt, _, _, _ = compute_state_mt_E4(uhat, scores, T)
        if est_name == "E4h":
            return _spectrum_perobs(v, uhat, T, mt, "rectangular")
        elif est_name == "E4s":
            return _spectrum_perobs(v, uhat, T, mt, "bartlett")
        elif est_name == "E4h_avg":
            return _spectrum_avg(v, T, compute_averaged_weights(mt, T, "rectangular"))
        elif est_name == "E4s_avg":
            return _spectrum_avg(v, T, compute_averaged_weights(mt, T, "bartlett"))
    raise ValueError(f"Unknown estimator: {est_name}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _init_store():
    return {
        "lam_min": [], "neg_eig": [], "fmin": [], "neg_spec": [],
        "cov95": [], "size5": [], "power_loc": [], "inadm_se": [],
        "cov95_psd": [], "size5_psd": [],
        "_lam_min": [], "_fmin": [],
        "neg_eig_mass": [], "dist_to_psd": [],
        "spec_neg_mass": [], "neg_spec_flag": [], "neg_eig_flag": [],
        "size5_flag": [], "cov95_flag": [], "power_flag": [],
        # Section 4 diagnostics (Lemma mixture-convexity, Prop discrepancy)
        "convex_flag": [], "min_delta2": [],
        "discrepancy_to_avg": [],
        # Bandwidth path diagnostics
        "delta_m": [], "mean_m": [], "sd_m": [], "selector_base": [],
        # Eventual PSD diagnostic (Cor perobs_eventual_psd)
        "eventual_psd_cond": [],
        # Sharp MSE diagnostic (Lem sharp_mse): sum of squared avg weights
        "V_T_sq": [],
        # Fourth moment diagnostic (A2')
        "fourth_moment": [],
    }


def _get_mt_and_info(est_name, uhat, scores, T):
    """Return (mt, selector_base) for any estimator."""
    if est_name.startswith("E3"):
        mt, _, _ = compute_state_mt_E3(uhat, T)
        return mt, np.nan
    elif est_name.startswith("E4"):
        mt, _, _, m_center = compute_state_mt_E4(uhat, scores, T)
        return mt, float(m_center)
    return None, np.nan


def run_monte_carlo():
    results = {}

    for dgp_name in ["DGP1", "DGP2"]:
        simulate_dgp = simulate_DGP1 if dgp_name == "DGP1" else simulate_DGP2
        dgp_id = 1 if dgp_name == "DGP1" else 2
        results[dgp_name] = {}

        for T in T_LIST:
            R = R_DICT[T]
            print(f"\n{'=' * 72}")
            print(f"  {dgp_name}  |  T={T}  |  R={R}")
            print(f"{'=' * 72}")

            store = {est: _init_store() for est in EST_LIST}

            for r in range(R):
                if (r + 1) % 500 == 0:
                    print(f"    rep {r + 1}/{R}")

                seed_u = SEED_BASE + 10_000 * dgp_id + 100 * (T_LIST.index(T) + 1) + r
                seed_x = seed_u + 50_000

                u = simulate_dgp(T, seed_u)
                X = simulate_regressors(T, seed_x)
                y = X @ BETA_TRUE + u
                beta_hat, uhat = ols(y, X)
                scores = X * uhat[:, None]

                # Local alternative
                delta = 1.0
                y_alt = X @ np.array([1.0, delta / np.sqrt(T), 0.0]) + u
                bhat_alt, uhat_alt = ols(y_alt, X)
                scores_alt = X * uhat_alt[:, None]

                # Compute bandwidth paths
                mt_E3, _, _ = compute_state_mt_E3(uhat, T)
                mt_E4, _, _, m_center_E4 = compute_state_mt_E4(uhat, scores, T)
                mt_E3_alt, _, _ = compute_state_mt_E3(uhat_alt, T)
                mt_E4_alt, _, _, _ = compute_state_mt_E4(uhat_alt, scores_alt, T)

                # Averaged weights for convexity checks (Lemma mixture-convexity)
                w_bart_E3 = compute_averaged_weights(mt_E3, T, "bartlett")
                w_rect_E3 = compute_averaged_weights(mt_E3, T, "rectangular")
                w_bart_E4 = compute_averaged_weights(mt_E4, T, "bartlett")
                w_rect_E4 = compute_averaged_weights(mt_E4, T, "rectangular")

                bart_E3_conv, bart_E3_d2 = check_convexity(w_bart_E3)
                rect_E3_conv, rect_E3_d2 = check_convexity(w_rect_E3)
                bart_E4_conv, bart_E4_d2 = check_convexity(w_bart_E4)
                rect_E4_conv, rect_E4_d2 = check_convexity(w_rect_E4)

                # Build all estimators
                estimators = {
                    "E0":       (HAC_E0(scores, T),              HAC_E0(scores_alt, T)),
                    "E1":       (HAC_E1(scores, T),              HAC_E1(scores_alt, T)),
                    "E2":       (HAC_E2(scores, T),              HAC_E2(scores_alt, T)),
                    "E3h":      (HAC_E3h(scores, uhat, T),       HAC_E3h(scores_alt, uhat_alt, T)),
                    "E3s":      (HAC_E3s(scores, uhat, T),       HAC_E3s(scores_alt, uhat_alt, T)),
                    "E3h_avg":  (HAC_E3h_avg(scores, uhat, T),   HAC_E3h_avg(scores_alt, uhat_alt, T)),
                    "E3s_avg":  (HAC_E3s_avg(scores, uhat, T),   HAC_E3s_avg(scores_alt, uhat_alt, T)),
                    "E4h":      (HAC_E4h(scores, uhat, T),       HAC_E4h(scores_alt, uhat_alt, T)),
                    "E4s":      (HAC_E4s(scores, uhat, T),       HAC_E4s(scores_alt, uhat_alt, T)),
                    "E4h_avg":  (HAC_E4h_avg(scores, uhat, T),   HAC_E4h_avg(scores_alt, uhat_alt, T)),
                    "E4s_avg":  (HAC_E4s_avg(scores, uhat, T),   HAC_E4s_avg(scores_alt, uhat_alt, T)),
                }

                # Discrepancies (Proposition form_comparison)
                disc = {}
                for fam, mt_fam in [("E3", mt_E3), ("E4", mt_E4)]:
                    for kk in ["h", "s"]:
                        key_po = f"{fam}{kk}"
                        key_avg = f"{fam}{kk}_avg"
                        Om_po = symmetrize(estimators[key_po][0])
                        Om_av = symmetrize(estimators[key_avg][0])
                        disc[key_po] = float(np.linalg.norm(Om_po - Om_av, ord='fro'))
                        disc[key_avg] = disc[key_po]

                for est_name, (Om, Om_alt) in estimators.items():
                    s = store[est_name]
                    Om = symmetrize(Om)
                    Om_alt = symmetrize(Om_alt)

                    vals = np.linalg.eigvalsh(Om)
                    lmin = float(vals.min())
                    s["lam_min"].append(lmin)
                    s["neg_eig"].append(int(lmin < 0))
                    s["_lam_min"].append(lmin)
                    s["neg_eig_mass"].append(float(np.sum(np.maximum(-vals, 0.0))))
                    s["neg_eig_flag"].append(int(lmin < 0))

                    # Spectrum
                    fhat = compute_spectrum(est_name, u, uhat, scores, T)
                    fmin = float(fhat.min())
                    s["fmin"].append(fmin)
                    s["neg_spec"].append(int(fmin < 0))
                    s["_fmin"].append(fmin)
                    s["neg_spec_flag"].append(int(fmin < 0))
                    s["spec_neg_mass"].append(float(_trapezoid(np.maximum(-fhat, 0.0), OMEGA_GRID)))

                    # Inference
                    Vb = sandwich_var(X, Om, T)
                    diag_ok = (Vb[1, 1] > 0)
                    s["inadm_se"].append(int(not diag_ok))
                    tstat = beta_hat[1] / np.sqrt(max(Vb[1, 1], 1e-30))
                    s["size5"].append(int(abs(tstat) > 1.96))
                    ci_lo = beta_hat[1] - 1.96 * np.sqrt(max(Vb[1, 1], 0.0))
                    ci_hi = beta_hat[1] + 1.96 * np.sqrt(max(Vb[1, 1], 0.0))
                    s["cov95"].append(int(ci_lo <= 0.0 <= ci_hi))

                    Vb_alt = sandwich_var(X, Om_alt, T)
                    ts_alt = bhat_alt[1] / np.sqrt(max(Vb_alt[1, 1], 1e-30))
                    s["power_loc"].append(int(abs(ts_alt) > 1.96))

                    s["size5_flag"].append(s["size5"][-1])
                    s["cov95_flag"].append(s["cov95"][-1])
                    s["power_flag"].append(s["power_loc"][-1])

                    # PSD repair
                    Om_psd = PSD_project(Om)
                    Vb_psd = sandwich_var(X, Om_psd, T)
                    tstat_psd = beta_hat[1] / np.sqrt(max(Vb_psd[1, 1], 1e-30))
                    s["size5_psd"].append(int(abs(tstat_psd) > 1.96))
                    ci_lo_p = beta_hat[1] - 1.96 * np.sqrt(max(Vb_psd[1, 1], 0.0))
                    ci_hi_p = beta_hat[1] + 1.96 * np.sqrt(max(Vb_psd[1, 1], 0.0))
                    s["cov95_psd"].append(int(ci_lo_p <= 0.0 <= ci_hi_p))
                    s["dist_to_psd"].append(float(np.linalg.norm(Om - Om_psd, ord="fro")))

                    # Fourth moment diagnostic (A2')
                    s["fourth_moment"].append(float(np.mean(np.sum(scores**2, axis=1)**2)))

                    # Section 4 diagnostics (Lemma mixture-convexity, Prop discrepancy)
                    if est_name in ["E3s_avg", "E3h_avg", "E3s", "E3h"]:
                        mt_used = mt_E3
                        if "s_avg" in est_name:
                            s["convex_flag"].append(int(bart_E3_conv))
                            s["min_delta2"].append(bart_E3_d2)
                        elif "h_avg" in est_name:
                            s["convex_flag"].append(int(rect_E3_conv))
                            s["min_delta2"].append(rect_E3_d2)
                        else:
                            s["convex_flag"].append(np.nan)
                            s["min_delta2"].append(np.nan)
                        s["discrepancy_to_avg"].append(disc.get(est_name, np.nan))
                        s["delta_m"].append(float(np.max(mt_used) - np.min(mt_used)))
                        s["mean_m"].append(float(np.mean(mt_used)))
                        s["sd_m"].append(float(np.std(mt_used)))
                        s["selector_base"].append(np.nan)
                        # Eventual PSD (Cor perobs_eventual_psd): 3*B^2*Delta_m vs lam_min(Omega_avg)
                        B_sq = float(np.max(np.sum(scores**2, axis=1)))
                        delta_m_val = float(np.max(mt_used) - np.min(mt_used))
                        lmin_avg = float(np.linalg.eigvalsh(symmetrize(estimators.get(est_name.replace("h","h_avg").replace("s","s_avg") if "_avg" not in est_name else est_name, (Om,Om))[0])).min())
                        s["eventual_psd_cond"].append(float(3*B_sq*delta_m_val) if delta_m_val > 0 else 0.0)
                        # Sharp MSE (Lem sharp_mse): sum of squared avg weights
                        w_avg_key = "bartlett" if "s" in est_name else "rectangular"
                        w_avg_tmp = compute_averaged_weights(mt_used, T, w_avg_key)
                        s["V_T_sq"].append(float(np.sum(w_avg_tmp**2)))
                    elif est_name in ["E4s_avg", "E4h_avg", "E4s", "E4h"]:
                        mt_used = mt_E4
                        if "s_avg" in est_name:
                            s["convex_flag"].append(int(bart_E4_conv))
                            s["min_delta2"].append(bart_E4_d2)
                        elif "h_avg" in est_name:
                            s["convex_flag"].append(int(rect_E4_conv))
                            s["min_delta2"].append(rect_E4_d2)
                        else:
                            s["convex_flag"].append(np.nan)
                            s["min_delta2"].append(np.nan)
                        s["discrepancy_to_avg"].append(disc.get(est_name, np.nan))
                        s["delta_m"].append(float(np.max(mt_used) - np.min(mt_used)))
                        s["mean_m"].append(float(np.mean(mt_used)))
                        s["sd_m"].append(float(np.std(mt_used)))
                        s["selector_base"].append(float(m_center_E4))
                        B_sq = float(np.max(np.sum(scores**2, axis=1)))
                        delta_m_val = float(np.max(mt_used) - np.min(mt_used))
                        s["eventual_psd_cond"].append(float(3*B_sq*delta_m_val) if delta_m_val > 0 else 0.0)
                        w_avg_key = "bartlett" if "s" in est_name else "rectangular"
                        w_avg_tmp = compute_averaged_weights(mt_used, T, w_avg_key)
                        s["V_T_sq"].append(float(np.sum(w_avg_tmp**2)))
                    else:
                        s["convex_flag"].append(np.nan)
                        s["min_delta2"].append(np.nan)
                        s["discrepancy_to_avg"].append(np.nan)
                        s["delta_m"].append(np.nan)
                        s["mean_m"].append(np.nan)
                        s["sd_m"].append(np.nan)
                        s["selector_base"].append(np.nan)
                        s["eventual_psd_cond"].append(np.nan)
                        s["V_T_sq"].append(np.nan)

            # ── Summarize ──
            summary = {}
            for est_name in EST_LIST:
                sv = store[est_name]
                neg_spec = np.array(sv["neg_spec_flag"], dtype=bool)
                neg_eig = np.array(sv["neg_eig_flag"], dtype=bool)
                size_arr = np.array(sv["size5_flag"], dtype=int)
                cov_arr = np.array(sv["cov95_flag"], dtype=int)
                power_arr = np.array(sv["power_flag"], dtype=int)

                out = {}
                add_binom_summary(out, "Pr(lmin<0)", sv["neg_eig"])
                add_binom_summary(out, "Pr(fmin<0)", sv["neg_spec"])
                add_binom_summary(out, "Coverage95", sv["cov95"])
                add_binom_summary(out, "Size5", sv["size5"])
                add_binom_summary(out, "Power(local)", sv["power_loc"])
                add_binom_summary(out, "Coverage95_PSD", sv["cov95_psd"])
                add_binom_summary(out, "Size5_PSD", sv["size5_psd"])

                # NetPower = mean(power - size) per replication
                net_power = np.array(sv["power_flag"], dtype=float) - np.array(sv["size5_flag"], dtype=float)
                add_mean_summary(out, "NetPower", net_power)

                add_mean_summary(out, "E[lmin]", sv["lam_min"])
                add_mean_summary(out, "E[fmin]", sv["fmin"])
                add_mean_summary(out, "E[neg_eig_mass]", sv["neg_eig_mass"])
                add_mean_summary(out, "E[dist_to_psd]", sv["dist_to_psd"])
                add_mean_summary(out, "E[spec_neg_mass]", sv["spec_neg_mass"])

                # Convexity
                convex_flags = [x for x in sv["convex_flag"] if not np.isnan(x)]
                out["convexity_rate"] = float(np.mean(convex_flags)) if convex_flags else np.nan

                disc_vals = [x for x in sv["discrepancy_to_avg"] if not np.isnan(x)]
                if disc_vals:
                    add_mean_summary(out, "E[disc_to_avg]", disc_vals)
                else:
                    out["E[disc_to_avg]"] = np.nan

                # Bandwidth path stats
                dm = [x for x in sv["delta_m"] if not np.isnan(x)]
                mm = [x for x in sv["mean_m"] if not np.isnan(x)]
                sdm = [x for x in sv["sd_m"] if not np.isnan(x)]
                sb = [x for x in sv["selector_base"] if not np.isnan(x)]
                out["E[delta_m]"] = float(np.mean(dm)) if dm else np.nan
                out["E[mean_m]"] = float(np.mean(mm)) if mm else np.nan
                out["E[sd_m]"] = float(np.mean(sdm)) if sdm else np.nan
                out["selector_base"] = float(np.mean(sb)) if sb else np.nan

                # Eventual PSD diagnostic
                epc = [x for x in sv["eventual_psd_cond"] if not np.isnan(x)]
                out["E[eventual_psd_cond]"] = float(np.mean(epc)) if epc else np.nan

                # Sharp MSE: sum of squared weights (Lem sharp_mse)
                vtsq = [x for x in sv["V_T_sq"] if not np.isnan(x)]
                out["E[sum_w2]"] = float(np.mean(vtsq)) if vtsq else np.nan

                # Fourth moment (A2')
                fm = sv["fourth_moment"]
                out["E[fourth_moment]"] = float(np.mean(fm)) if fm else np.nan

                # Tercile analysis on delta_m
                dm_arr = np.array(dm) if dm else np.array([])
                if len(dm_arr) > 10:
                    t1, t2 = np.percentile(dm_arr, [33.33, 66.67])
                    low_mask = dm_arr <= t1
                    high_mask = dm_arr >= t2
                    neg_spec_dm = np.array(sv["neg_spec_flag"][:len(dm_arr)], dtype=int)
                    neg_eig_dm = np.array(sv["neg_eig_flag"][:len(dm_arr)], dtype=int)
                    out["Pr(fmin<0)|low_dm"] = float(np.mean(neg_spec_dm[low_mask])) if low_mask.sum() > 0 else np.nan
                    out["Pr(fmin<0)|high_dm"] = float(np.mean(neg_spec_dm[high_mask])) if high_mask.sum() > 0 else np.nan
                    out["Pr(lmin<0)|low_dm"] = float(np.mean(neg_eig_dm[low_mask])) if low_mask.sum() > 0 else np.nan
                    out["Pr(lmin<0)|high_dm"] = float(np.mean(neg_eig_dm[high_mask])) if high_mask.sum() > 0 else np.nan
                else:
                    for key in ["Pr(fmin<0)|low_dm", "Pr(fmin<0)|high_dm", "Pr(lmin<0)|low_dm", "Pr(lmin<0)|high_dm"]:
                        out[key] = np.nan

                # Conditional inference
                p, lo, hi, nsub = conditional_binom(size_arr, neg_spec)
                out["Size5|neg_spec"] = p
                out["Size5|neg_spec_lo"] = lo
                out["Size5|neg_spec_hi"] = hi
                p, lo, hi, _ = conditional_binom(cov_arr, neg_spec)
                out["Cov95|neg_spec"] = p
                out["Cov95|neg_spec_lo"] = lo
                out["Cov95|neg_spec_hi"] = hi

                out["_lam_min"] = np.array(sv["_lam_min"], dtype=float)
                out["_fmin"] = np.array(sv["_fmin"], dtype=float)

                summary[est_name] = out

            results[dgp_name][T] = summary

            # Print
            print(f"\n  {'Est':<10} {'Pr(lmin<0)':>12} {'Pr(f<0)':>10} {'Cov95':>8} {'Size5':>8}")
            print("  " + "-" * 55)
            for est_name in EST_LIST:
                sv = summary[est_name]
                print(f"  {est_name:<10} {sv['Pr(lmin<0)']:>12.4f} {sv['Pr(fmin<0)']:>10.4f} "
                      f"{sv['Coverage95']:>8.4f} {sv['Size5']:>8.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 9. MULTIVARIATE STRESS TEST (referee request)
# ══════════════════════════════════════════════════════════════════════════════

def run_multivariate_mc():
    """
    Bivariate Markov-switching VAR stress test.
    Uses DGP2 only, joint Wald test for both slope coefficients.
    """
    print("\n" + "=" * 72)
    print("  MULTIVARIATE STRESS TEST (DGP2, bivariate)")
    print("=" * 72)

    results_mv = {}
    est_mv = ["E1", "E3h", "E3s", "E3h_avg", "E3s_avg", "E4h", "E4s", "E4h_avg", "E4s_avg"]

    for T in T_LIST:
        R = R_DICT[T]
        print(f"\n  T={T}, R={R}")

        store_mv = {e: {"lam_min": [], "neg_eig": [], "size_joint": [], "power_joint": [],
                        "size_joint_psd": [], "dist_psd": []} for e in est_mv}

        for r in range(R):
            if (r + 1) % 500 == 0:
                print(f"    rep {r + 1}/{R}")
            seed = SEED_BASE + 90_000 + 100 * (T_LIST.index(T) + 1) + r
            rng = np.random.default_rng(seed)

            u_biv = simulate_DGP2_bivariate(T, seed)
            X_reg = simulate_regressors(T, seed + 60_000)

            # Two equations
            y1 = X_reg @ BETA_TRUE + u_biv[:, 0]
            y2 = X_reg @ BETA_TRUE + u_biv[:, 1]
            b1, u1 = ols(y1, X_reg)
            b2, u2 = ols(y2, X_reg)

            # Stack scores: (T, 6)
            scores_full = np.column_stack([X_reg * u1[:, None], X_reg * u2[:, None]])
            uhat_avg = 0.5 * (np.abs(u1) + np.abs(u2))

            # Local alt
            delta = 1.0
            y1_alt = X_reg @ np.array([1.0, delta/np.sqrt(T), 0.0]) + u_biv[:, 0]
            y2_alt = X_reg @ np.array([1.0, 0.0, delta/np.sqrt(T)]) + u_biv[:, 1]
            b1a, u1a = ols(y1_alt, X_reg)
            b2a, u2a = ols(y2_alt, X_reg)
            scores_alt = np.column_stack([X_reg * u1a[:, None], X_reg * u2a[:, None]])
            uhat_avg_alt = 0.5 * (np.abs(u1a) + np.abs(u2a))

            mt_E3, _, _ = compute_state_mt_E3(uhat_avg, T)
            mt_E4, _, _, _ = compute_state_mt_E4(uhat_avg, scores_full[:, :3], T)
            mt_E3a, _, _ = compute_state_mt_E3(uhat_avg_alt, T)
            mt_E4a, _, _, _ = compute_state_mt_E4(uhat_avg_alt, scores_alt[:, :3], T)

            def build_om(est, sc, uh, mt3, mt4):
                if est == "E1":
                    return HAC_E1(sc, T)
                elif est == "E3h":
                    return _hac_perobs(sc, mt3, T, "rectangular")
                elif est == "E3s":
                    return _hac_perobs(sc, mt3, T, "bartlett")
                elif est == "E3h_avg":
                    return _hac_avg(sc, mt3, T, "rectangular")
                elif est == "E3s_avg":
                    return _hac_avg(sc, mt3, T, "bartlett")
                elif est == "E4h":
                    return _hac_perobs(sc, mt4, T, "rectangular")
                elif est == "E4s":
                    return _hac_perobs(sc, mt4, T, "bartlett")
                elif est == "E4h_avg":
                    return _hac_avg(sc, mt4, T, "rectangular")
                elif est == "E4s_avg":
                    return _hac_avg(sc, mt4, T, "bartlett")

            for est in est_mv:
                Om = symmetrize(build_om(est, scores_full, uhat_avg, mt_E3, mt_E4))
                Om_alt = symmetrize(build_om(est, scores_alt, uhat_avg_alt, mt_E3a, mt_E4a))

                vals = np.linalg.eigvalsh(Om)
                lmin = float(vals.min())
                store_mv[est]["lam_min"].append(lmin)
                store_mv[est]["neg_eig"].append(int(lmin < 0))

                # Joint Wald: test beta_1^(1) = 0 and beta_1^(2) = 0
                # Indices 1 and 4 in the 6x6 covariance
                XX = X_reg.T @ X_reg / T
                try:
                    XXi = np.linalg.inv(XX)
                    V_full = np.kron(np.eye(2), XXi) @ Om @ np.kron(np.eye(2), XXi) / T
                    R_mat = np.zeros((2, 6))
                    R_mat[0, 1] = 1.0
                    R_mat[1, 4] = 1.0
                    bhat_full = np.concatenate([b1, b2])
                    diff = R_mat @ bhat_full
                    V_sub = R_mat @ V_full @ R_mat.T
                    wald = float(diff @ np.linalg.solve(V_sub + 1e-12 * np.eye(2), diff))
                    store_mv[est]["size_joint"].append(int(wald > 5.991))  # chi2(2), 5%

                    bhat_alt_full = np.concatenate([b1a, b2a])
                    V_alt = np.kron(np.eye(2), XXi) @ Om_alt @ np.kron(np.eye(2), XXi) / T
                    V_sub_a = R_mat @ V_alt @ R_mat.T
                    diff_a = R_mat @ bhat_alt_full
                    wald_a = float(diff_a @ np.linalg.solve(V_sub_a + 1e-12*np.eye(2), diff_a))
                    store_mv[est]["power_joint"].append(int(wald_a > 5.991))
                except:
                    store_mv[est]["size_joint"].append(1)
                    store_mv[est]["power_joint"].append(1)

                # PSD repair
                Om_psd = PSD_project(Om)
                store_mv[est]["dist_psd"].append(float(np.linalg.norm(Om - Om_psd, ord="fro")))
                try:
                    V_psd = np.kron(np.eye(2), XXi) @ Om_psd @ np.kron(np.eye(2), XXi) / T
                    V_sub_p = R_mat @ V_psd @ R_mat.T
                    wald_p = float(diff @ np.linalg.solve(V_sub_p + 1e-12*np.eye(2), diff))
                    store_mv[est]["size_joint_psd"].append(int(wald_p > 5.991))
                except:
                    store_mv[est]["size_joint_psd"].append(1)

        # Summarize
        summary_mv = {}
        for est in est_mv:
            sv = store_mv[est]
            out = {}
            add_binom_summary(out, "Pr(lmin<0)", sv["neg_eig"])
            add_binom_summary(out, "Size5_joint", sv["size_joint"])
            add_binom_summary(out, "Power_joint", sv["power_joint"])
            add_binom_summary(out, "Size5_joint_PSD", sv["size_joint_psd"])
            net = np.array(sv["power_joint"], dtype=float) - np.array(sv["size_joint"], dtype=float)
            add_mean_summary(out, "NetPower_joint", net)
            add_mean_summary(out, "E[dist_psd]", sv["dist_psd"])
            summary_mv[est] = out
        results_mv[T] = summary_mv

    return results_mv


# ══════════════════════════════════════════════════════════════════════════════
# 10. LATEX TABLE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def df_to_latex(df, path, caption, label, col_rename=None, float_fmt="%.4g", escape=False, table_env=True, fontsize="footnotesize"):
    df_out = df.copy()
    if col_rename:
        df_out = df_out.rename(columns=col_rename)
    tab = df_out.to_latex(index=False, escape=escape, float_format=float_fmt, longtable=False)
    if table_env:
        tex = (f"% Requires \\usepackage{{booktabs,graphicx}}\n"
               f"\\begin{{table}}[!htbp]\n\\centering\n\\{fontsize}\n"
               f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
               f"\\resizebox{{\\textwidth}}{{!}}{{\n{tab}}}\n\\end{{table}}\n")
    else:
        tex = tab
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  LaTeX: {path.name}")


def make_table_mechanism(results):
    """Table MC.1: Mechanism diagnostics."""
    est_show = ["E1", "E3h", "E3s", "E3h_avg", "E3s_avg", "E4h", "E4s", "E4h_avg", "E4s_avg"]
    rows = []
    for dgp in ["DGP1", "DGP2"]:
        for T in T_LIST:
            for est in est_show:
                sv = results[dgp][T][est]
                rows.append({
                    "DGP": dgp, "$T$": T, "Est.": est,
                    r"$\Pr(\hat f<0)$": sv["Pr(fmin<0)"],
                    r"$\Pr(\hat\lambda<0)$": sv["Pr(lmin<0)"],
                    r"$\bar f_{\min}$": sv["E[fmin]"],
                    r"$\bar\lambda_{\min}$": sv["E[lmin]"],
                    r"Neg.\ eig.": sv["E[neg_eig_mass]"],
                    r"Neg.\ spec.": sv["E[spec_neg_mass]"],
                    "Disc.": sv.get("E[disc_to_avg]", np.nan),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_MC_mechanism.csv", index=False)
    df_to_latex(df, OUT / "Table_MC_mechanism.tex",
                caption=r"Mechanism table: hard cutoff vs taper under shared paths. Path-averaged Bartlett is provably PSD (Theorem~\ref{thm:psd_pathavg}); hard-cutoff average can fail PSD (Proposition~\ref{prop:hardcutoff_nonPSD}). Discrepancy controlled by $\Delta_m$ (Proposition~\ref{prop:form_comparison}).",
                label="tab:mc_mechanism", escape=False)
    return df


def make_table_inference(results):
    """Table MC.2: Inference diagnostics."""
    est_show = ["E1", "E3h", "E3s", "E3h_avg", "E3s_avg", "E4h", "E4s", "E4h_avg", "E4s_avg"]
    rows = []
    for dgp in ["DGP1", "DGP2"]:
        for T in T_LIST:
            for est in est_show:
                sv = results[dgp][T][est]
                rows.append({
                    "DGP": dgp, "$T$": T, "Est.": est,
                    "Cov95": sv["Coverage95"],
                    "Size5": sv["Size5"],
                    "Power": sv["Power(local)"],
                    "Net": sv["NetPower"],
                    r"Size$^P$": sv["Size5_PSD"],
                    r"Cov$^P$": sv["Coverage95_PSD"],
                    r"Size$|\hat f\!<\!0$": fmt_prob_ci(sv["Size5|neg_spec"], sv["Size5|neg_spec_lo"], sv["Size5|neg_spec_hi"]),
                    r"Cov$|\hat f\!<\!0$": fmt_prob_ci(sv["Cov95|neg_spec"], sv["Cov95|neg_spec_lo"], sv["Cov95|neg_spec_hi"]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_MC_inference.csv", index=False)
    df_to_latex(df, OUT / "Table_MC_inference.tex",
                caption=r"Inference table. Corollary~\ref{cor:wald} guarantees valid Wald tests for the path-averaged Bartlett form. NetPower = mean(power $-$ size), showing that apparent power gains from hard cutoff reflect size distortion.",
                label="tab:mc_inference", escape=False)
    return df


def make_table_variation(results):
    """Table MC.3: Bandwidth-path variation diagnostics."""
    est_show = ["E3h", "E3s", "E3h_avg", "E3s_avg", "E4h", "E4s", "E4h_avg", "E4s_avg"]
    rows = []
    for dgp in ["DGP1", "DGP2"]:
        for T in T_LIST:
            for est in est_show:
                sv = results[dgp][T][est]
                rows.append({
                    "DGP": dgp, "$T$": T, "Est.": est,
                    r"$\bar\Delta_m$": sv.get("E[delta_m]", np.nan),
                    r"$\bar m$": sv.get("E[mean_m]", np.nan),
                    r"sd$(m_t)$": sv.get("E[sd_m]", np.nan),
                    "Base": sv.get("selector_base", np.nan),
                    r"$\sum w^2$": sv.get("E[sum_w2]", np.nan),
                    r"$3B^2\!\Delta$": sv.get("E[eventual_psd_cond]", np.nan),
                    r"$\hat f\!<\!0|$lo": sv.get("Pr(fmin<0)|low_dm", np.nan),
                    r"$\hat f\!<\!0|$hi": sv.get("Pr(fmin<0)|high_dm", np.nan),
                    r"$\hat\lambda\!<\!0|$lo": sv.get("Pr(lmin<0)|low_dm", np.nan),
                    r"$\hat\lambda\!<\!0|$hi": sv.get("Pr(lmin<0)|high_dm", np.nan),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_MC_variation.csv", index=False)
    df_to_latex(df, OUT / "Table_MC_variation.tex",
                caption=r"Bandwidth-path variation diagnostics. Failure rates by tercile of $\Delta_m$; consistent with Proposition~\ref{prop:form_comparison} (discrepancy $\le 3B^2\Delta_m$) and Corollary~\ref{cor:perobs_eventual_psd} (eventual PSD when $3B^2\Delta_m^* < \lambda_{\min}(\Omega)$).",
                label="tab:mc_variation", escape=False)
    return df


def make_table_repair(results):
    """Table MC.4: PSD repair comparison."""
    est_show = ["E0", "E3h", "E3h_avg", "E4h", "E4h_avg"]
    rows = []
    for dgp in ["DGP1", "DGP2"]:
        for T in T_LIST:
            for est in est_show:
                sv = results[dgp][T][est]
                rows.append({
                    "DGP": dgp, "$T$": T, "Est.": est,
                    "Size": sv["Size5"],
                    r"Size$^P$": sv["Size5_PSD"],
                    r"$\Delta$S": sv["Size5_PSD"] - sv["Size5"],
                    "Cov": sv["Coverage95"],
                    r"Cov$^P$": sv["Coverage95_PSD"],
                    r"$\Delta$C": sv["Coverage95_PSD"] - sv["Coverage95"],
                    "Dist.": sv["E[dist_to_psd]"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_MC_repair.csv", index=False)
    df_to_latex(df, OUT / "Table_MC_repair.tex",
                caption=r"Ex-post PSD repair comparison. The table shows how much size and coverage change after projecting the covariance estimate onto the PSD cone.",
                label="tab:mc_repair", escape=False)
    return df


def make_table_multivariate(results_mv):
    """Table MC.5: Multivariate stress test."""
    est_show = ["E1", "E3h", "E3s", "E3h_avg", "E3s_avg", "E4h", "E4s", "E4h_avg", "E4s_avg"]
    rows = []
    for T in T_LIST:
        for est in est_show:
            sv = results_mv[T][est]
            rows.append({
                "$T$": T, "Est.": est,
                r"$\Pr(\hat\lambda\!<\!0)$": sv["Pr(lmin<0)"],
                "Size": sv["Size5_joint"],
                "Power": sv["Power_joint"],
                "Net": sv["NetPower_joint"],
                r"Size$^P$": sv["Size5_joint_PSD"],
                "Dist.": sv["E[dist_psd]"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_MC_multivariate.csv", index=False)
    df_to_latex(df, OUT / "Table_MC_multivariate.tex",
                caption=r"Optional multivariate stress test using a bivariate Markov-switching VAR and a joint Wald test. Included to show how PSD failures matter in a genuinely multivariate setting.",
                label="tab:mc_multivar", escape=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 11. FIGURE GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def bartlett_sine_transform(t_grid):
    t = np.asarray(t_grid, dtype=float)
    out = np.zeros_like(t)
    mask = np.abs(t) > 1e-10
    out[mask] = (1.0 - np.cos(t[mask])) / t[mask]
    return out

def rectangular_sine_transform(t_grid):
    return np.sin(np.asarray(t_grid, dtype=float))

def parzen_sine_transform(t_grid):
    t = np.asarray(t_grid, dtype=float)
    x = np.linspace(0.0, 1.0, 2001)
    kappa_p = np.where(x < 0.5, 12.0 * x - 18.0 * x**2, 6.0 * (1.0 - x)**2)
    kappa_p[0] = 0.0
    out = np.array([_trapezoid(kappa_p * np.sin(ti * x), x) for ti in t])
    out[np.abs(out) < 1e-12] = 0.0
    return out

def fejer_window(m, omega_grid):
    omega = np.asarray(omega_grid, dtype=float)
    out = np.empty_like(omega)
    den = np.sin(omega / 2.0)
    mask = np.abs(den) > 1e-12
    out[mask] = np.sin((m + 1.0) * omega[mask] / 2.0)**2 / ((m + 1.0) * den[mask]**2)
    out[~mask] = m + 1.0
    return out

def dirichlet_window(m, omega_grid):
    omega = np.asarray(omega_grid, dtype=float)
    out = np.empty_like(omega)
    den = np.sin(omega / 2.0)
    mask = np.abs(den) > 1e-12
    out[mask] = np.sin((m + 0.5) * omega[mask]) / den[mask]
    out[~mask] = 2.0 * m + 1.0
    return out


def fig_MC1_structural_bridge(results):
    """Figure MC.1: Structural bridge from theory to MC."""
    t_grid = np.linspace(0.01, 4 * np.pi, 400)
    om_grid = np.linspace(0.0, np.pi, 400)
    m_demo = 10

    S_B = bartlett_sine_transform(t_grid)
    S_R = rectangular_sine_transform(t_grid)
    S_P = parzen_sine_transform(t_grid)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    x_dense = np.linspace(0, 1.5, 500)
    kappa_B = np.where((x_dense > 0) & (x_dense < 1), 1.0, 0.0)
    ax.fill_between(x_dense, kappa_B, 0, alpha=0.35, color="steelblue", label=r"$\kappa_B = \mathbf{1}_{(0,1)}$")
    ax.plot(x_dense, kappa_B, color="steelblue", linewidth=1.5)
    ax.annotate("", xy=(1.0, 1.2), xytext=(1.0, 0.0), arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5))
    ax.scatter([1.0], [1.2], color="crimson", s=80, zorder=5, label=r"$\kappa_R = \delta(u-1)$")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("$u$"); ax.set_ylabel(r"$\kappa_K(u)$")
    ax.set_title("(A) Defect densities: Bartlett vs Rectangular", fontsize=10); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(t_grid, S_B, color="steelblue", lw=2, label=r"Bartlett: $(1-\cos t)/t \geq 0$")
    ax.plot(t_grid, S_R, color="crimson", lw=2, ls="--", label=r"Rectangular: $\sin t$")
    ax.plot(t_grid, S_P, color="darkorange", lw=1.5, ls="-.", label="Parzen: positive")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$S_K(t)$")
    ax.set_title("(B) Sine transforms of defect densities", fontsize=10); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(om_grid/np.pi, fejer_window(m_demo, om_grid), color="steelblue", lw=2, label=f"Bartlett/Fejér, m={m_demo}")
    L_rect = dirichlet_window(m_demo, om_grid)
    ax.plot(om_grid/np.pi, L_rect, color="crimson", lw=2, ls="--", label=f"Rectangular/Dirichlet, m={m_demo}")
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(om_grid/np.pi, np.minimum(L_rect, 0), 0, color="crimson", alpha=0.15)
    ax.set_xlabel(r"$\omega/\pi$"); ax.set_ylabel("Spectral window")
    ax.set_title("(C) Discrete windows: Fejér vs Dirichlet", fontsize=10); ax.legend(fontsize=8)

    ax = axes[1, 1]
    show_ests = ["E3h", "E3s", "E3h_avg", "E3s_avg", "E4h", "E4s"]
    x = np.arange(len(T_LIST))
    width = 0.12
    for k, est in enumerate(show_ests):
        vals = [results["DGP2"][T][est]["Pr(fmin<0)"] for T in T_LIST]
        ax.bar(x + k*width, vals, width, label=est, color=COLORS[est], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + 2.5*width); ax.set_xticklabels([f"$T={T}$" for T in T_LIST])
    ax.set_ylabel(r"$\Pr(\min_\omega \hat{f}(\omega) < 0)$")
    ax.set_title("(D) DGP2: E3/E4 families", fontsize=10); ax.legend(fontsize=6, ncol=2); ax.set_ylim(bottom=0)

    fig.suptitle("Structural Bridge: Prop 3.2 (fixed) + Cor 4.3 / Prop 4.5 (path-averaged)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC1_structural_bridge.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC1_structural_bridge.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.1 saved.")


def fig_MC2_failures(results):
    """Figure MC.2: Failure frequencies and severity."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    est_show = ["E1", "E3h", "E3s", "E4h", "E4s", "E3s_avg", "E4s_avg"]
    for col, dgp in enumerate(["DGP1", "DGP2"]):
        ax = axes[0, col]
        x = np.arange(len(T_LIST)); width = 0.11
        for k, est in enumerate(est_show):
            vals = [results[dgp][T][est]["Pr(fmin<0)"] for T in T_LIST]
            ax.bar(x + k*width, vals, width, label=est, color=COLORS[est], alpha=0.85)
        ax.set_xticks(x + 3*width); ax.set_xticklabels([f"T={T}" for T in T_LIST])
        ax.set_ylabel(r"$\Pr(\hat f_{\min}<0)$"); ax.set_title(f"{dgp} — Spectral failures"); ax.legend(fontsize=5, ncol=2); ax.set_ylim(bottom=0)

        ax = axes[1, col]
        for k, est in enumerate(est_show):
            vals = [results[dgp][T][est]["Pr(lmin<0)"] for T in T_LIST]
            ax.bar(x + k*width, vals, width, label=est, color=COLORS[est], alpha=0.85)
        ax.set_xticks(x + 3*width); ax.set_xticklabels([f"T={T}" for T in T_LIST])
        ax.set_ylabel(r"$\Pr(\hat\lambda_{\min}<0)$"); ax.set_title(f"{dgp} — PSD failures"); ax.legend(fontsize=5, ncol=2); ax.set_ylim(bottom=0)
    fig.suptitle("Failure frequencies and severity diagnostics", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC2_failures.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC2_failures.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.2 saved.")


def fig_MC3_inference(results):
    """Figure MC.3: Coverage, size, size-adjusted power."""
    metrics = ["Coverage95", "Size5", "Power(local)"]
    labels = ["95% CI Coverage", "Size at 5%", "Power (local)"]
    est_show = ["E1", "E3h", "E3s", "E4h", "E4s", "E3s_avg", "E4s_avg"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, dgp in enumerate(["DGP1", "DGP2"]):
        for col, (met, lbl) in enumerate(zip(metrics, labels)):
            ax = axes[row, col]; x = np.arange(len(T_LIST)); width = 0.11
            for k, est in enumerate(est_show):
                vals = [results[dgp][T][est][met] for T in T_LIST]
                ax.bar(x + k*width, vals, width, label=est, color=COLORS[est], alpha=0.85)
            if met == "Coverage95": ax.axhline(0.95, color="black", lw=1, ls="--")
            if met == "Size5": ax.axhline(0.05, color="red", lw=1, ls="--")
            ax.set_xticks(x + 3*width); ax.set_xticklabels([f"T={T}" for T in T_LIST])
            ax.set_title(f"{dgp} — {lbl}"); ax.legend(fontsize=5, ncol=2); ax.set_ylim(0, 1)
    fig.suptitle("Inference diagnostics: coverage, size, and size-adjusted power", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC3_inference.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC3_inference.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.3 saved.")


def fig_MC4_variation(results):
    """Figure MC.4: Bandwidth-path variation and failure rates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, dgp in zip(axes, ["DGP1", "DGP2"]):
        for est, color, marker in [("E3h", COLORS["E3h"], "o"), ("E4h", COLORS["E4h"], "s")]:
            lo = [results[dgp][T][est].get("Pr(fmin<0)|low_dm", 0) for T in T_LIST]
            hi = [results[dgp][T][est].get("Pr(fmin<0)|high_dm", 0) for T in T_LIST]
            lo = [x if not np.isnan(x) else 0 for x in lo]
            hi = [x if not np.isnan(x) else 0 for x in hi]
            ax.plot(T_LIST, lo, f"{marker}--", color=color, alpha=0.6, label=f"{est} low $\\Delta_m$")
            ax.plot(T_LIST, hi, f"{marker}-", color=color, lw=2, label=f"{est} high $\\Delta_m$")
        ax.set_xlabel("$T$"); ax.set_ylabel(r"$\Pr(\hat f_{\min}<0)$")
        ax.set_title(dgp); ax.legend(fontsize=7); ax.set_ylim(bottom=0)
    fig.suptitle("Bandwidth-path variation and failure rates", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC4_variation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC4_variation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.4 saved.")


def fig_MC5_repair(results):
    """Figure MC.5: Ex-post PSD repair."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    est_show = ["E0", "E3h", "E4h", "E3h_avg", "E4h_avg"]
    for ax, dgp in zip(axes, ["DGP1", "DGP2"]):
        x = np.arange(len(T_LIST)); width = 0.15
        for k, est in enumerate(est_show):
            delta_size = [results[dgp][T][est]["Size5_PSD"] - results[dgp][T][est]["Size5"] for T in T_LIST]
            ax.bar(x + k*width, delta_size, width, label=est, color=COLORS.get(est, "gray"), alpha=0.85)
        ax.set_xticks(x + 2*width); ax.set_xticklabels([f"T={T}" for T in T_LIST])
        ax.set_ylabel(r"$\Delta$ Size (after $-$ before PSD repair)")
        ax.set_title(dgp); ax.legend(fontsize=7); ax.axhline(0, color="black", lw=0.8)
    fig.suptitle("Ex-post PSD repair in DGP2", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC5_repair.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC5_repair.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.5 saved.")


def fig_MC6_multivariate(results_mv):
    """Figure MC.6: Multivariate stress test."""
    est_show = ["E1", "E3h", "E3s", "E4h", "E4s", "E3s_avg", "E4s_avg"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(T_LIST)); width = 0.11

    ax = axes[0]
    for k, est in enumerate(est_show):
        vals = [results_mv[T][est]["Pr(lmin<0)"] for T in T_LIST]
        ax.bar(x + k*width, vals, width, label=est, color=COLORS[est], alpha=0.85)
    ax.set_xticks(x + 3*width); ax.set_xticklabels([f"T={T}" for T in T_LIST])
    ax.set_ylabel(r"$\Pr(\hat\lambda_{\min}<0)$"); ax.set_title("PSD failures (multivariate)")
    ax.legend(fontsize=5, ncol=2); ax.set_ylim(bottom=0)

    ax = axes[1]
    for k, est in enumerate(est_show):
        vals = [results_mv[T][est]["Size5_joint"] for T in T_LIST]
        ax.bar(x + k*width, vals, width, label=est, color=COLORS[est], alpha=0.85)
    ax.axhline(0.05, color="red", lw=1, ls="--")
    ax.set_xticks(x + 3*width); ax.set_xticklabels([f"T={T}" for T in T_LIST])
    ax.set_ylabel("Joint Size at 5%"); ax.set_title("Joint Wald size (multivariate)")
    ax.legend(fontsize=5, ncol=2); ax.set_ylim(0, 0.3)

    fig.suptitle("Optional multivariate stress test", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_MC6_multivariate.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_MC6_multivariate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure MC.6 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 12. CONTINUOUS FAMILY H_{alpha,beta} + CERTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def w_weight(a):
    a = np.asarray(a, dtype=float)
    z = a / _2PI
    return (1.0 / (8.0 * _PI)) * np.log(np.maximum(z, 1.0 + 1e-16)) * (z ** 1.5)


def gauss_legendre_integrate(f, a0, a1, n=64):
    xs, ws = np.polynomial.legendre.leggauss(n)
    mid = 0.5 * (a0 + a1); half = 0.5 * (a1 - a0)
    a = mid + half * xs
    return half * np.sum(ws * f(a))


def W_approx_paper(alpha, beta, t_grid, x_max=3.0, n_x=260, tol_a=1e-14, n_quad=64):
    x_grid = np.linspace(0.0, x_max, n_x)
    dx = x_grid[1] - x_grid[0]
    wx = np.ones_like(x_grid); wx[0] = 0.5; wx[-1] = 0.5; wx *= dx
    I0 = np.zeros_like(x_grid); I1 = np.zeros_like(x_grid); I2 = np.zeros_like(x_grid)
    for i, x in enumerate(x_grid):
        cx = np.cosh(x)
        a0 = _2PI * np.exp(abs(x))
        tail = max(5.0, np.log(1.0 / tol_a) / max(cx, 1.0))
        a1 = a0 + tail
        def base(a): return w_weight(a) * np.exp(-a * cx)
        I0[i] = gauss_legendre_integrate(base, a0, a1, n=n_quad)
        I1[i] = gauss_legendre_integrate(lambda a: a * base(a), a0, a1, n=n_quad)
        I2[i] = gauss_legendre_integrate(lambda a: a**2 * base(a), a0, a1, n=n_quad)
    cosh_x = np.cosh(x_grid)
    Hx = I2 + beta * I0 - alpha * cosh_x * I1
    t_grid = np.asarray(t_grid, dtype=float)
    C = np.cos(np.outer(t_grid, x_grid))
    W_vals = 2.0 * (C @ (Hx * wx))
    return W_vals, {"x_grid": x_grid}


def run_parameter_map(N_alpha=30, N_beta=30, t_max=30.0, dt=0.1,
                      x_max=3.0, n_x=220, tol_a=1e-14, n_quad=48,
                      eps_scr=1e-10, K_cert=10):
    print("\n  PARAMETER MAP FOR H_{alpha,beta}")
    alpha_grid = np.linspace(0.05, 2*np.pi - 0.05, N_alpha)
    beta_grid = np.logspace(-1, 1, N_beta)
    t_grid = np.arange(0.0, t_max + 1e-12, dt)
    screen_map = np.full((N_alpha, N_beta), np.inf, dtype=float)
    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            W_vals, _ = W_approx_paper(alpha, beta, t_grid, x_max=x_max, n_x=n_x, tol_a=tol_a, n_quad=n_quad)
            screen_map[i, j] = float(np.min(W_vals))
    refined_pts = []
    neg_idx = list(zip(*np.where(screen_map < -eps_scr)))
    if neg_idx:
        sorted_idx = sorted(neg_idx, key=lambda ij: screen_map[ij[0], ij[1]])
        for (i, j) in sorted_idx[:min(K_cert, len(sorted_idx))]:
            alpha, beta = float(alpha_grid[i]), float(beta_grid[j])
            refined_pts.append((alpha, beta, screen_map[i, j], True, 0.01))
    return alpha_grid, beta_grid, screen_map, refined_pts


def fig_B1_parameter_map(alpha_grid, beta_grid, screen_map, refined_pts):
    fig, ax = plt.subplots(figsize=(9, 7))
    vmin, vmax = float(np.min(screen_map)), float(np.max(screen_map))
    im = ax.pcolormesh(beta_grid, alpha_grid, screen_map, cmap="RdBu_r",
                       norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=max(1e-6, vmax)))
    plt.colorbar(im, ax=ax, label=r"$\min_t W_{\alpha,\beta}(t)$")
    ax.contour(beta_grid, alpha_grid, screen_map, levels=[0], colors="black", linewidths=1.5)
    ax.scatter([9.0], [6.0], c="red", s=120, marker="*", zorder=6, edgecolors="black", label=r"$(\alpha,\beta)=(6,9)$")
    ax.set_xscale("log"); ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\alpha$")
    ax.set_title("Continuous-family screening for sign changes"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_B1_parameter_map.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_B1_parameter_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure B.1 saved.")


def fig_B2_certified():
    """Certified W curves near (6,9)."""
    cases = [(6.0, 9.0), (5.7, 9.0), (6.0, 8.0)]
    t_plot = np.linspace(15, 25, 600)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (a, b) in zip(axes, cases):
        W, _ = W_approx_paper(a, b, t_plot, x_max=3.5, n_x=330, tol_a=1e-16, n_quad=64)
        ax.plot(t_plot, W, color="steelblue", lw=1.5)
        ax.axhline(0, color="red", lw=1, ls="--")
        ax.set_xlabel("$t$"); ax.set_ylabel(r"$W_{\alpha,\beta}(t)$")
        ax.set_title(rf"$\alpha={a}, \beta={b}$", fontsize=9)
    fig.suptitle("Pointwise sign certification near $(6,9)$", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_B2_certified.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_B2_certified.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure B.2 saved.")


def C_ab(alpha, beta):
    return (1.0 / (8*np.pi)) * ((2*np.pi)**(-1.5)) * (1 + alpha/(2*np.pi) + beta/(2*np.pi)**2)

def eps_A_majorant(X, A, alpha, beta):
    C = C_ab(alpha, beta)
    poly = A**4 + 4*A**3 + 12*A**2 + 24*A + 24
    return float(2 * X * C * np.exp(-A) * poly)

def eps_X_majorant(X, alpha, beta):
    C = C_ab(alpha, beta)
    eX = np.exp(X)
    bracket = (4*np.pi)**4/8 * eX + (4*np.pi)**3 + 6*(4*np.pi)**2*np.exp(-X) + 24*(4*np.pi)*np.exp(-2*X) + 48*np.exp(-3*X)
    return float(C / np.pi * np.exp(-np.pi * np.exp(2*X)) * bracket)


def run_certification():
    """Pointwise sign certification (Appendix C)."""
    print("\n  POINTWISE SIGN CERTIFICATION")
    candidates = [
        ("near(6,9)", 6.0, 9.0, 19.88),
        ("near(6,9)", 5.7, 9.0, 19.88),
        ("near(6,9)", 6.0, 8.0, 19.88),
        ("strong-neg", 6.233, 0.1, 14.9),
        ("strong-neg", 6.233, 0.1172, 14.9),
    ]
    rows = []
    for group, a, b, t0 in candidates:
        W, _ = W_approx_paper(a, b, np.array([t0]), x_max=3.5, n_x=600, tol_a=1e-16, n_quad=96)
        W_hat = float(W[0])
        epsA = eps_A_majorant(3.0, 200.0, a, b)
        epsX = eps_X_majorant(3.0, a, b)
        epsTot = epsA + epsX + 1e-10
        sign = "NEG" if W_hat + epsTot < 0 else ("POS" if W_hat - epsTot > 0 else "NA")
        verdict = "NEGATIVE CERTIFIED" if sign == "NEG" else ("POSITIVE CERTIFIED" if sign == "POS" else "INCONCLUSIVE")
        rows.append({"Group": group, "alpha": a, "beta": b, "t0": t0,
                     r"$\widehat{W}(t_0)$": f"{W_hat:.2e}", r"$\varepsilon_{\mathrm{tot}}$": f"{epsTot:.2e}",
                     "Sign": sign, "Verdict": verdict})
        print(f"  {group}: alpha={a:.3f}, beta={b:.4f}, t0={t0:.2f} => {verdict}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Table_B1_certified_signchanges.csv", index=False)
    df_to_latex(df, OUT / "Table_B1_certified_signchanges.tex",
                caption=r"Pointwise sign certification via explicit deterministic error bounds.",
                label="tab:B1_certified", escape=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 13. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  MONTE CARLO: Path-Averaged HAC Estimation")
    print("  Section 3: fixed-weight theory")
    print("  Section 4: Lemma (mixture-convexity), Theorem (PSD),")
    print("             Lemma (sharp MSE), Prop (rate), Cor (Wald),")
    print("             Cor (eventual PSD per-obs)")
    print("  Estimators: E0-E4 families (per-obs + path-averaged)")
    print("=" * 72)

    # Verify counterexample: Prop (hardcutoff_nonPSD), T=4
    print("\n[0] Verifying counterexample (Prop hardcutoff_nonPSD, T=4)...")
    ver = verify_theorem41_explicit()
    print(f"    Bartlett avg convex: {ver['bart_convex']} (min Delta^2 = {ver['bart_min_d2']:.6f})")
    print(f"    Rectangular avg convex: {ver['rect_convex']} (min Delta^2 = {ver['rect_min_d2']:.6f})")
    print(f"    Rect avg min spectral window: {ver['rect_W_min']:.4f} (negative: {ver['rect_W_negative']})")

    print("\n[1] Running Monte Carlo ...")
    results = run_monte_carlo()

    print("\n[2] Generating tables ...")
    make_table_mechanism(results)
    make_table_inference(results)
    make_table_variation(results)
    make_table_repair(results)

    print("\n[3] Generating MC figures ...")
    fig_MC1_structural_bridge(results)
    fig_MC2_failures(results)
    fig_MC3_inference(results)
    fig_MC4_variation(results)
    fig_MC5_repair(results)

    print("\n[4] Running multivariate stress test ...")
    results_mv = run_multivariate_mc()
    make_table_multivariate(results_mv)
    fig_MC6_multivariate(results_mv)

    print("\n[5] Continuous-family parameter map ...")
    alpha_grid, beta_grid, screen_map, refined_pts = run_parameter_map()
    fig_B1_parameter_map(alpha_grid, beta_grid, screen_map, refined_pts)

    print("\n[6] Certified curves and sign certification ...")
    fig_B2_certified()
    cert_df = run_certification()

    print("\n" + "=" * 72)
    print("  ALL DONE. Key outputs in ./outputs/:")
    print("    Tables: Table_MC_mechanism, Table_MC_inference, Table_MC_variation,")
    print("            Table_MC_repair, Table_MC_multivariate, Table_B1_certified")
    print("    Figures: Figure_MC1–MC6, Figure_B1–B2")
    print("=" * 72)