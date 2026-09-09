"""D3 집중도 지표용 «진단용 복제 계열» 생성.

사전 고정: 20260908_D3_출력량_사전고정.md (sha256 c10c9ae2…)
           20260908_D3_부록1_출처정책_사전고정.md (sha256 dc067157…)

정책 — 원 실행이 아니라 «선언된 규칙을 따르는 하나의 복제 계열» 안에서 전부 같게 한다.
    초기화  torch.manual_seed(BASE[kind] + fold)  «모형 생성 직전»
            BASE = {A:2000, B:1000, C:1500}.  ★팔에 무관 — 같은 겹이면 같은 시드
    일정    run_cdld.py 와 동일
            A: cycles=6 sub_epochs=1 cyclic=False
            C: cycles=3 sub_epochs=1 cyclic=True
            공통 bs=8192  fit seed=42+fold
    자료    data.prep(None, ward=False) · SEED=42 5겹 · .prep_base.npz

    $ python3 run_conc.py <초 예산>
작업단위마다 체크포인트를 남기므로 같은 명령을 다시 띄우면 이어진다.
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, cdld_core as CC

CD = os.path.dirname(os.path.abspath(__file__)) + '/'
OUT = CD + 'ck_conc/'; os.makedirs(OUT, exist_ok=True)
BASE_SEED = {'A': 2000, 'B': 1000, 'C': 1500}
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 540.0

# 작업 목록 — A 는 팔 불변이므로 real 한 번만. C 는 귀무 3팔.
UNITS = [('real', f, 'A') for f in range(5)] + \
        [(a, f, 'C') for a in ('shuffle0', 'shuffle1', 'shuffle2') for f in range(5)]


def unit(d, D_cache, arm, fold, kind):
    fp = OUT + f'{arm}_f{fold}_{kind}.json'
    if os.path.exists(fp):
        return None
    if arm not in D_cache:
        nu = d['nur'] if arm == 'real' else data.shuffle_nurse(d, int(arm[-1]), 'A')
        D_cache[arm] = data.tensors(d, nur_override=nu)
    D = D_cache[arm]
    tr = np.flatnonzero(d['fold'] != fold); te = np.flatnonzero(d['fold'] == fold)
    y = d['y']; sizes = list(d['sizes']); ncov = d['cov'].shape[1]
    torch.manual_seed(BASE_SEED[kind] + fold)          # ★선언된 초기화 정책
    if kind == 'A':
        m = CC.Finder(sizes, ncov, n_drug=int(d['n_drug']), drug_latent=True)
        cyc, sub, cyclic = 6, 1, False
    else:
        m = CC.Finder(sizes, ncov, latent=True,
                      n_nurse=int(d['n_nurse']), n_drug=int(d['n_drug']))
        cyc, sub, cyclic = 3, 1, True
    t0 = time.time()
    CC.fit(m, D, tr, te, cycles=cyc, sub_epochs=sub, bs=8192, seed=42 + fold, cyclic=cyclic)
    p = CC.predict(m, D, te)
    sse = float(((y[te] - p) ** 2).sum()); sstot = float(((y[te] - y[tr].mean()) ** 2).sum())
    bss = 1 - sse / sstot
    # 기록된 원 실행 값과의 일치를 함께 남긴다 (§4.13 형식)
    op = CD + f'ck/base_{arm}_f{fold}_{kind}.json'
    obss = None
    if os.path.exists(op):
        o = json.load(open(op)); obss = 1 - o['sse'] / o['sstot']
    elif kind == 'A':                                   # A 는 real 기록만 존재 (팔 불변)
        o = json.load(open(CD + f'ck/base_real_f{fold}_A.json')); obss = 1 - o['sse'] / o['sstot']
    np.savez_compressed(OUT + f'pred_{arm}_f{fold}_{kind}.npz',
                        te=te.astype(np.int32), p=p.astype(np.float32))
    rec = dict(arm=arm, fold=fold, kind=kind, n_te=int(len(te)), sse=sse, sstot=sstot,
               bss=bss, bss_original=obss,
               diff=(None if obss is None else bss - obss),
               init_seed=BASE_SEED[kind] + fold, fit_seed=42 + fold,
               cycles=cyc, sub_epochs=sub, cyclic=cyclic,
               secs=round(time.time() - t0, 1))
    json.dump(rec, open(fp, 'w'))
    d_ = rec['diff']
    print(f"{arm:9} f{fold} {kind}  BSS {bss:+.5f}  기록 "
          f"{'—' if obss is None else f'{obss:+.5f}'}  차이 "
          f"{'—' if d_ is None else f'{d_:+.5f}'}  {rec['secs']:.0f}s", flush=True)
    return rec


def main():
    t0 = time.time()
    d = data.prep(None, ward=False)
    D_cache = {}
    done = sum(os.path.exists(OUT + f'{a}_f{f}_{k}.json') for a, f, k in UNITS)
    print(f'[conc] 시작 — 총 {len(UNITS)}단위 · 완료 {done}단위 · 예산 {BUDGET:.0f}s', flush=True)
    for arm, fold, kind in UNITS:
        if time.time() - t0 > BUDGET:
            print('[conc] 예산 소진 — 이어서 실행하면 계속된다', flush=True); break
        unit(d, D_cache, arm, fold, kind)
    done = sum(os.path.exists(OUT + f'{a}_f{f}_{k}.json') for a, f, k in UNITS)
    print(f'[conc] {done}/{len(UNITS)} 단위 완료  ({time.time()-t0:.0f}s)', flush=True)
    if done == len(UNITS):
        print('=== 전 단위 완료 ===', flush=True)


if __name__ == '__main__':
    main()
