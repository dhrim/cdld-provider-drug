"""CDLD 판 본측정 — 사전명시 절차를 «그대로» 옮긴다. 추정기만 ALS -> CDLD.

  A : 근무조 + 약물 + 진단챕터 + 공변량 + ★약물 잠재 64   (간호사 없음)
  B : A + 간호사 주효과(1차원)
  C : B + ★간호사 잠재 64 — 약물 잠재와 «순환 발견»(ULD <-> ILD)

  ★A·B 에도 약물 잠재를 둔다. 세 단계에서 «약물 쪽 표현을 동일»하게 만들어야
    C-B 가 «간호사 쪽만»의 증분이 된다. 이 교정 전에는 섞기 판에서도 C-B 가
    양수였다 — 섞기 귀무(S)가 교란된 대조를 잡아냈다.

  겹    환자 단위 5겹 (원판과 동일한 난수 구성)
  귀무  입원 단위 간호사 배정 치환 2회 (원판과 동일)
  지표  R² = 1 - SSE/SS_tot (원판과 동일), AUC 보조

학습 일정 (정본 CDLD 를 이 자료 규모에 맞춰 고정. 시험부는 보지 않고 정함)
  LATENT 64 · Adam 5e-4 · L2 1e-4 · batch 8192
  C : 3 순환 x (ULD 1에폭 + ILD 1에폭) = 6 에폭
  A·B : 4 에폭

작업 단위마다 체크포인트를 남긴다. 예산 초과 시 중단하고 다음 호출에서 이어간다.
"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, cdld_core as CC

import paths
CD = paths.WORK + os.sep
CK = CD + 'ck/'; os.makedirs(CK, exist_ok=True)
ARMS = ['real', 'shuffle0', 'shuffle1', 'shuffle2']   # ★귀무 3회 (검토 의견 ⑥)
CYCLES, SUB, EP_AB, BS = 3, 1, 6, 8192
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 540.0
WARD = len(sys.argv) > 2 and sys.argv[2] == 'ward'
TAG = 'wardX' if WARD else 'base'


def auc(y, p):
    o = np.argsort(p); y = y[o]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    r = np.empty(len(y)); r[np.arange(len(y))] = np.arange(1, len(y) + 1)
    return float((r[y > .5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def unit(d, D_cache, arm, fold, kind):
    """한 작업 단위 = (팔, 겹, 모형). 결과 json 저장."""
    key = f'{TAG}_{arm}_f{fold}_{kind}'
    fp = CK + key + '.json'
    if os.path.exists(fp): return None
    if arm not in D_cache:
        nu = d['nur'] if arm == 'real' else data.shuffle_nurse(d, int(arm[-1]), 'A')
        D_cache[arm] = data.tensors(d, nur_override=nu)
    D = D_cache[arm]
    tr = np.flatnonzero(d['fold'] != fold); te = np.flatnonzero(d['fold'] == fold)
    y = d['y']
    kw = dict(n_nurse=int(d['n_nurse']), n_drug=int(d['n_drug']))
    sizes = list(d['sizes']); ncov = d['cov'].shape[1]
    wk = dict(n_ward=int(d['n_ward'])) if WARD else {}
    if kind == 'A':   m = CC.Finder(sizes, ncov, n_drug=kw['n_drug'], drug_latent=True, **wk)
    elif kind == 'B': m = CC.Finder(sizes, ncov, drug_latent=True, **kw, **wk)
    else:             m = CC.Finder(sizes, ncov, latent=True, **kw, **wk)
    t0 = time.time()
    CC.fit(m, D, tr, te, cycles=(CYCLES if kind == 'C' else EP_AB),
           sub_epochs=(SUB if kind == 'C' else 1), bs=BS, seed=42 + fold,
           cyclic=(kind == 'C'))
    p = CC.predict(m, D, te)
    res = dict(tag=TAG, arm=arm, fold=int(fold), kind=kind,
               n_te=int(len(te)),
               sse=float(((y[te] - p) ** 2).sum()),
               sstot=float(((y[te] - y[tr].mean()) ** 2).sum()),
               auc=auc(y[te], p), secs=round(time.time() - t0, 1))
    json.dump(res, open(fp, 'w'))
    np.savez_compressed(CK + f'pred_{key}.npz', te=te.astype(np.int32),
                        p=p.astype(np.float32))      # 환자 단위 부트스트랩 CI 용
    if kind == 'C' and arm == 'real':
        np.savez(CK + f'uv_{TAG}_f{fold}.npz',
                 U=m.u_tab.weight.detach().numpy(), V=m.v_tab.weight.detach().numpy(),
                 nurse_eff=m.n_eff.weight.detach().numpy().ravel())
    print(f'  ✓ {key}  R2 {1-res["sse"]/res["sstot"]:+.4f}  AUC {res["auc"]:.4f}  {res["secs"]:.0f}s', flush=True)
    return res


def main():
    e = data.frame() if not os.path.exists(CD + f'.prep_{TAG}.npz') else None
    d = data.prep(e, ward=WARD)
    D_cache = {}
    queue = []
    for f in range(5):
        queue.append(('real', f, 'A'))          # A 는 간호사가 없어 팔과 무관 — 한 번만
    for kind in ('B', 'C'):            # 싼 것부터 — 호출당 묶음 효율
        for arm in ARMS:
            for f in range(5):
                queue.append((arm, f, kind))
    t0 = time.time(); done = 0
    for arm, f, k in queue:
        if os.path.exists(CK + f'{TAG}_{arm}_f{f}_{k}.json'): continue
        est = {'A': 340., 'B': 350., 'C': 470.}[k]      # 단위별 예상 소요
        if time.time() - t0 + est > BUDGET:
            break
        unit(d, D_cache, arm, f, k); done += 1
    left = sum(1 for a, f, k in queue if not os.path.exists(CK + f'{TAG}_{a}_f{f}_{k}.json'))
    print(f'\n[{TAG}] 이번 호출 {done}건 · 남은 {left}건 / 전체 {len(queue)}건', flush=True)


if __name__ == '__main__':
    main()
