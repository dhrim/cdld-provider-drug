"""native B 재학습 — D 진단용 가중치 확보.

본실험은 state_dict 를 저장하지 않았다(지표 json + 예측 npz + uv 만 저장).
D 진단은 counterfactual 순전파가 필요하므로 가중치가 있어야 한다.

★재현성 한계(기록) : 본실험의 모형 초기화는 «호출 안에서 앞선 단위들이 소비한»
전역 torch RNG 상태에 의존했다(fit() 이 manual_seed 를 모형 생성 «뒤»에 건다).
예산 분할 호출로 실행됐으므로 그 상태를 사후에 복원할 수 없다. 따라서 이 재학습은
비트 동일 재현이 아니라 «같은 절차·같은 일정·같은 겹의 진단용 복제»다.
재학습 B 의 BSS 가 기록값과 겹간 SD(0.0041) 안에 들어오는지 확인하고,
D 진단은 «재학습된 이 모형»의 함수 속성으로 보고한다.
γ* 계산에는 기록된 본실험 값을 쓴다.
"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, cdld_core as CC

import paths
CD = paths.WORK + os.sep; CK = CD + 'ck/'
CKB = CD + 'ckpt/'; os.makedirs(CKB, exist_ok=True)


def main(fold):
    d = data.prep(None, ward=False)
    D = data.tensors(d)
    tr = np.flatnonzero(d['fold'] != fold); te = np.flatnonzero(d['fold'] == fold)
    y = d['y']
    sizes = list(d['sizes']); ncov = d['cov'].shape[1]
    torch.manual_seed(1500 + fold)                      # 진단 복제용 고정 시드
    m = CC.Finder(sizes, ncov, n_nurse=int(d['n_nurse']),
                  n_drug=int(d['n_drug']), latent=True)
    t0 = time.time()
    CC.fit(m, D, tr, te, cycles=3, sub_epochs=1, bs=8192, seed=42 + fold, cyclic=True)
    p = CC.predict(m, D, te)
    sse = float(((y[te] - p) ** 2).sum())
    sstot = float(((y[te] - y[tr].mean()) ** 2).sum())
    bss = 1 - sse / sstot
    orig = json.load(open(CK + f'base_real_f{fold}_C.json'))
    obss = 1 - orig['sse'] / orig['sstot']
    np.savez_compressed(CKB + f'pred_C_f{fold}.npz', te=te.astype(np.int32),
                        p=p.astype(np.float32))
    torch.save(dict(state=m.state_dict(), sizes=sizes, ncov=ncov,
                    n_nurse=int(d['n_nurse']), n_drug=int(d['n_drug'])),
               CKB + f'C_f{fold}.pt')
    rec = dict(fold=fold, bss_retrain=bss, bss_original=obss,
               diff=bss - obss, secs=round(time.time() - t0, 1))
    json.dump(rec, open(CKB + f'C_f{fold}.json', 'w'))
    print(f'fold {fold}  재학습 BSS {bss:+.5f}  기록 {obss:+.5f}  차이 {bss-obss:+.5f}  '
          f'{rec["secs"]:.0f}s', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
