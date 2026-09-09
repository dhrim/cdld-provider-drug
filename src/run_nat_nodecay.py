"""CDLD-A 본실행 — base 설정, real + null x2, 환자 단위 5겹.

A 는 native 와 «공유»한다(계산 0). 기록된 pred_base_real_f{f}_A.npz 를 그대로 쓴다.
B, C 만 학습한다.

일정은 native 와 «동일» — B: 6에폭 / C: 3순환 x (ULD 1 + ILD 1) = 6에폭
Adam 5e-4, L2 1e-4, batch 8192, seed 42+fold.
"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, cdld_core as CC

import paths
CD = paths.WORK + os.sep
CK = CD + 'ck_natnd/'; os.makedirs(CK, exist_ok=True)
CKN = CD + 'ck/'
ARMS = ['real', 'shuffle0', 'shuffle1']
TAG = 'base'
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9


def auc(y, p):
    o = np.argsort(p); y = y[o]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    r = np.arange(1, len(y) + 1)
    return float((r[y > .5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def unit(d, D_cache, arm, fold, kind):
    key = f'{TAG}_{arm}_f{fold}_{kind}'
    fp = CK + key + '.json'
    if os.path.exists(fp): return None
    if arm not in D_cache:
        nu = d['nur'] if arm == 'real' else data.shuffle_nurse(d, int(arm[-1]), 'A')
        D_cache[arm] = data.tensors(d, nur_override=nu)
    D = D_cache[arm]
    tr = np.flatnonzero(d['fold'] != fold); te = np.flatnonzero(d['fold'] == fold)
    y = d['y']
    sizes = list(d['sizes']); ncov = d['cov'].shape[1]
    torch.manual_seed(4000 + fold * 10 + {'B': 1, 'C': 2}[kind])
    m = CC.Finder(sizes, ncov, n_nurse=int(d['n_nurse']), n_drug=int(d['n_drug']),
                  latent=True)      # native C 구조 그대로
    t0 = time.time()
    CC.fit(m, D, tr, te, cycles=(3 if kind == 'C' else 6),
           sub_epochs=1, bs=8192, seed=42 + fold, cyclic=(kind == 'C'), no_decay=('u_tab', 'v_tab'))
    p = CC.predict(m, D, te)
    res = dict(tag=TAG, model='native-nodecay', arm=arm, fold=int(fold), kind=kind,
               n_te=int(len(te)),
               sse=float(((y[te] - p) ** 2).sum()),
               sstot=float(((y[te] - y[tr].mean()) ** 2).sum()),
               auc=auc(y[te], p), secs=round(time.time() - t0, 1))
    json.dump(res, open(fp, 'w'))
    np.savez_compressed(CK + f'pred_{key}.npz', te=te.astype(np.int32),
                        p=p.astype(np.float32))
    torch.save(m.state_dict(), CK + f'w_{key}.pt')
    print(f'  ✓ natND {key}  BSS {1-res["sse"]/res["sstot"]:+.5f}  '
          f'AUC {res["auc"]:.4f}  {res["secs"]:.0f}s', flush=True)
    return res


def share_A():
    """native 의 A 를 그대로 CDLD-A 의 A 로 «공유»한다 (계산 0)."""
    for f in range(5):
        src = CKN + f'{TAG}_real_f{f}_A.json'
        dst = CK + f'{TAG}_real_f{f}_A.json'
        if os.path.exists(dst): continue
        r = json.load(open(src)); r['model'] = 'CDLD-A'; r['shared_from'] = 'native CDLD A'
        json.dump(r, open(dst, 'w'))
        ps, pd = CKN + f'pred_{TAG}_real_f{f}_A.npz', CK + f'pred_{TAG}_real_f{f}_A.npz'
        if os.path.exists(ps) and not os.path.exists(pd):
            import shutil; shutil.copy(ps, pd)
    print('  A 공유 완료 (5겹, 계산 0)', flush=True)


def main():
    d = data.prep(None, ward=False)
    share_A()
    D_cache = {}
    # B 단계는 잠재표가 없어 감쇠 변경의 영향이 «구조적으로» 없다 -> ck_a 결과 공유
    import shutil
    for arm in ARMS:
        for f in range(5):
            for ext in ('json',):
                s0, d0 = CD+f'ck/{TAG}_{arm}_f{f}_B.{ext}', CK+f'{TAG}_{arm}_f{f}_B.{ext}'
                if os.path.exists(s0) and not os.path.exists(d0): shutil.copy(s0, d0)
            s0, d0 = CD+f'ck/pred_{TAG}_{arm}_f{f}_B.npz', CK+f'pred_{TAG}_{arm}_f{f}_B.npz'
            if os.path.exists(s0) and not os.path.exists(d0): shutil.copy(s0, d0)
    queue = [(arm, f, 'C') for arm in ARMS for f in range(5)]
    t0 = time.time(); done = 0
    for arm, f, k in queue:
        if os.path.exists(CK + f'{TAG}_{arm}_f{f}_{k}.json'): continue
        if time.time() - t0 + {'B': 350., 'C': 400.}[k] > BUDGET: break
        unit(d, D_cache, arm, f, k); done += 1
    left = sum(1 for a, f, k in queue if not os.path.exists(CK + f'{TAG}_{a}_f{f}_{k}.json'))
    print(f'\n[CDLD-A] 이번 호출 {done}건 · 남은 {left}건 / 전체 {len(queue)}건', flush=True)


if __name__ == '__main__':
    main()
