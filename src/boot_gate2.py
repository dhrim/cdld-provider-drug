"""Fit gate 부트스트랩 — native C (재학습분) vs CDLD-A2 C. 사전 문서 §7."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import paths
CD = paths.WORK + os.sep; DELTA = 0.0025; B = 2000


def cat(paths):
    ids, ps = [], []
    for p in paths:
        z = np.load(p); ids.append(z['te'].astype(np.int64)); ps.append(z['p'])
    return np.concatenate(ids), np.concatenate(ps)


def main():
    d = data.prep(None, ward=False); y = d['y'].astype(float); fold = d['fold']
    i1, p1 = cat([CD + f'ckpt/pred_C_f{f}.npz' for f in range(5)])          # native
    i2, p2 = cat([CD + f'ck_a2/pred_base_real_f{f}_C.npz' for f in range(5)])  # CDLD-A2
    o1, o2 = np.argsort(i1), np.argsort(i2)
    idx = i1[o1]; assert np.array_equal(idx, i2[o2])
    p1, p2 = p1[o1], p2[o2]; yv = y[idx]
    ybar = np.array([y[fold != f].mean() for f in range(5)])
    ref = ybar[fold[idx]]
    e1 = (yv - p1) ** 2; e2 = (yv - p2) ** 2; et = (yv - ref) ** 2
    pt = pd.factorize(data.frame().subject_id)[0][idx]
    order = np.argsort(pt, kind='stable'); pts = pt[order]
    bnd = np.flatnonzero(np.r_[True, pts[1:] != pts[:-1], True])
    st = bnd[:-1]
    S1 = np.add.reduceat(e1[order], st); S2 = np.add.reduceat(e2[order], st)
    ST = np.add.reduceat(et[order], st); n = len(st)
    point = (S1.sum() - S2.sum()) / ST.sum()
    rng = np.random.default_rng(11); out = np.empty(B)
    for b in range(B):
        s = rng.integers(0, n, n)
        out[b] = (S1[s].sum() - S2[s].sum()) / ST[s].sum()
    lo, hi = np.percentile(out, [2.5, 97.5])
    bss1 = 1 - S1.sum() / ST.sum(); bss2 = 1 - S2.sum() / ST.sum()
    res = dict(native_C_BSS=float(bss1), cdldA2_C_BSS=float(bss2),
               dBSS_point=float(point), ci_lo=float(lo), ci_hi=float(hi),
               delta=DELTA, gate_pass=bool(lo > -DELTA), B=B, n_patients=int(n),
               n_rows=int(len(idx)),
               note='conditional on the fitted models; evaluation-population sampling only')
    print(f'native C BSS {bss1:+.5f}   CDLD-A2 C BSS {bss2:+.5f}')
    print(f'dBSS (A2 - native) {point:+.5f}   95%CI [{lo:+.5f}, {hi:+.5f}]   margin -{DELTA}')
    print(f'환자 {n:,}명 · 행 {len(idx):,}   ->  {"관문 통과" if lo > -DELTA else "★관문 실패"}')
    json.dump(res, open(CD + 'diag/gate_C_a2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
