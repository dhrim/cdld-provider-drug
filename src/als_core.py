"""간호사 x 약물 — 핵심 적합 루틴. 연구계획 4·5절.

참조선  y = mu + gamma[유닛] + gamma[근무조] + alpha[간호사] + beta[약물]
            + x'theta + delta_U[유닛,약물] + delta_S[근무조,약물]     ← U0 필수

delta[간호사,약물] 세 팔
  Z : u_i' Z_j          약물 = 알려진 분류체계 (35차원)
  V : u_i' v_j          약물 = 발견된 잠재 (랭크 r)
  H : Z 적합 후 잔차에 V   분류체계 위에 잠재를 얹음
"""
import numpy as np
from scipy import sparse

N_DEMEAN, N_ALS, SEED = 10, 15, 42
RANKS = [1, 2, 3, 5, 10, 20]
RIDGES = [1., 10., 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
WEIGHTS = ["count", "reliability"]


def gmean(v, c, n):
    s = np.bincount(c, weights=v, minlength=n); k = np.bincount(c, minlength=n)
    return np.divide(s, k, out=np.zeros(n), where=k > 0)


def fit_baseline(y, codes, sizes, X, train, ridge_x=10.0):
    """U0 포함 가법 참조선. 누적합을 유지해 요인 수에 선형이 되게 한다.
       (매번 나머지 합을 다시 만들면 요인^2 x 행 이 되어 못 쓴다.)"""
    yt = y[train]; Xt = X[train] if X.shape[1] else None
    ct = [c[train] for c in codes]
    g = yt.mean()
    eff = [np.zeros(s) for s in sizes]
    contrib = [np.zeros(len(yt)) for _ in codes]      # eff[p][ct[p]]
    total = np.zeros(len(yt))                          # sum of contrib
    th = np.zeros(X.shape[1]) if X.shape[1] else None
    xfit = np.zeros(len(yt))
    G = (Xt.T @ Xt + ridge_x * np.eye(X.shape[1])) if X.shape[1] else None
    for _ in range(N_DEMEAN):
        for p in range(len(codes)):
            total -= contrib[p]
            eff[p] = gmean(yt - g - total - xfit, ct[p], sizes[p])
            contrib[p] = eff[p][ct[p]]
            total += contrib[p]
        if X.shape[1]:
            th = np.linalg.solve(G, Xt.T @ (yt - g - total))
            xfit = Xt @ th

    def predict(mask):
        out = g + sum(e[c[mask]] for e, c in zip(eff, codes))
        return out + (X[mask] @ th if X.shape[1] else 0)
    return predict


def dense_pair(ci, cj, r_c, w, n_i, n_j):
    """(n_i x n_j) 밀집 가중·가중잔차 행렬. 1129 x 204 라 사소하다.
       이렇게 하면 ALS 한 단계가 BLAS 행렬곱 하나가 된다."""
    W = np.zeros((n_i, n_j)); R = np.zeros((n_i, n_j))
    W[ci, cj] = w
    R[ci, cj] = w * r_c
    return W, R


def _solve_side(W, R, F, ridge):
    """min sum_ij W_ij (r_ij - u_i'F_j)^2 + ridge|u_i|^2 를 모든 i 에 대해."""
    k = F.shape[1]
    FF = (F[:, :, None] * F[:, None, :]).reshape(len(F), k * k)   # (n_j, k*k)
    A = (W @ FF).reshape(len(W), k, k) + ridge * np.eye(k)
    b = R @ F
    return np.linalg.solve(A, b[:, :, None])[:, :, 0]


def fit_factor(W, R, F, ridge):
    """u_i' F_j 형태. F 는 고정 표현(Z) 또는 현재 V. 반환 U (n_i, k)."""
    return _solve_side(W, R, F, ridge)


def fit_low_rank(W, R, rank, ridge, seed=SEED, n_iter=N_ALS):
    n_i, n_j = W.shape
    rng = np.random.default_rng(seed)
    V = rng.normal(scale=.1, size=(n_j, rank))
    Wt, Rt = W.T.copy(), R.T.copy()
    for _ in range(n_iter):
        U = _solve_side(W, R, V, ridge)
        V = _solve_side(Wt, Rt, U, ridge)
        nm = np.linalg.norm(V, axis=0)
        V = V / np.where(nm > 1e-12, nm, 1.0)
    U = _solve_side(W, R, V, ridge)
    return U, V


def predict_cells(U, F, ci_all, cj_all):
    return (U[ci_all] * F[cj_all]).sum(1)


def unrestricted(r_c, w, resid_var):
    """셀별 축소 평균. 미관측 셀에서는 호출자가 0 을 준다."""
    between = np.average(r_c ** 2, weights=w)
    s = resid_var / between if between > 0 else np.inf
    return (w * r_c) / (w + s) if np.isfinite(s) else np.zeros(len(r_c))
