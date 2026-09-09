"""CDLD 판 자료 준비 — 사전명시 설계를 그대로 옮긴다(추정기만 교체)."""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
W = paths.DATA + os.sep       # MIMIC 파생 프레임 위치
CD = paths.WORK + os.sep      # 체크포인트·캐시 출력
N_FOLDS, SEED = 5, 42


def frame():
    fp = W + '.frame_ward.pkl'
    if not os.path.exists(fp):
        paths.need_data(f'derived frame {fp}')
    return pd.read_pickle(fp)


def prep(e, ward=False, cache=True):
    """codes, cov, nur, drg, y, fold, 메타 반환. ward=True 면 careunit 코드 추가."""
    tag = 'ward' if ward else 'base'
    cp = CD + f'.prep_{tag}.npz'
    if cache and os.path.exists(cp):
        d = np.load(cp, allow_pickle=True)
        return {k: d[k] for k in d.files}
    if e is None:
        # 호출부는 대부분 prep(None) 로 «캐시만» 쓴다. 캐시가 없으면 여기서 멈춰야
        # 하며, 예전처럼 None 을 계속 들고 가 AttributeError 로 죽으면 안 된다.
        paths.need_data(f'prepared design cache {cp} (or pass a built frame to prep())')

    nur, nl = pd.factorize(e.enter_provider_id)
    drg, dl = pd.factorize(e.drug)
    code_cols = ['shift', 'drug', 'dx']
    codes, sizes = [], []
    for c in code_cols:
        v, lv = pd.factorize(e[c].astype(str))
        codes.append(v.astype(np.int64)); sizes.append(len(lv))
    codes = np.column_stack(codes)

    num = ['age', 'n_dx', 'n_drug_adm', 'n_event_adm', 'icu', 'icu_los']
    X = np.column_stack([e[c].fillna(e[c].median()).to_numpy(float) for c in num])
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    for c in ['gender', 'admission_type', 'insurance']:
        X = np.hstack([X, pd.get_dummies(e[c].astype(str)).to_numpy(float)[:, 1:]])

    # ★환자 단위 5겹 — 사전명시 원판과 «동일한» 구성
    pt, pl = pd.factorize(e.subject_id)
    fold = np.random.default_rng(SEED).integers(0, N_FOLDS, len(pl))[pt].astype(np.int64)

    extra = {}
    if ward:
        wv, wl = pd.factorize(e['careunit'].astype(str))
        extra = dict(ward=wv.astype(np.int64), n_ward=len(wl))
    out = dict(**extra, codes=codes, cov=X.astype(np.float32), nur=nur.astype(np.int64),
               drg=drg.astype(np.int64), y=e.omit.to_numpy(np.float32),
               fold=fold, sizes=np.array(sizes), n_nurse=len(nl), n_drug=len(dl),
               hadm=pd.factorize(e.hadm_id.fillna(-1))[0].astype(np.int64),
               nurse_labels=np.asarray(nl, dtype=object),
               drug_labels=np.asarray(dl, dtype=object))
    if cache:
        np.savez(cp, **out)
    return out


def tensors(d, nur_override=None):
    n = nur_override if nur_override is not None else d['nur']
    w = torch.from_numpy(d['ward']) if 'ward' in d else None
    return (torch.from_numpy(d['codes']), torch.from_numpy(d['cov']),
            torch.from_numpy(n), torch.from_numpy(d['drg']),
            torch.from_numpy(d['y']), w)


def shuffle_nurse(d, k, design='A'):
    """★귀무 — 제공자 «정체성»만 파괴하고 설계 구조는 보존한다.

    구판(v1)은 `nh[hadm] = nur` 의 팬시 인덱싱 때문에 입원마다 «마지막 한 명»만
    남아, 158,422개 입원 전부가 제공자 1명으로 접혔다. 실제 자료는 입원당
    평균 3.02명(중앙 2, 최대 635)이고 행의 90.7%가 다제공자 입원에 속한다.
    구판은 다른 귀무였다.

    design='A' (주판) 입원별 «제공자 라벨 재배정»
        각 입원의 서로 다른 제공자 k명을 전역 제공자 풀에서 빈도 가중으로
        비복원 추출한 k명으로 «갈아끼운다». 어느 행이 그중 누구에게 가는지의
        패턴(다중도)은 그대로 둔다.
        보존 : 입원당 행 수 · 입원당 제공자 수 · 입원 내 배정 패턴 · 제공자 빈도 분포
        파괴 : 제공자–환자/약물 결합

    design='B' (민감도) 크기 층화 «입원 블록» 교환
        (행 수, 제공자 수)가 같은 입원끼리 배정 벡터를 통째로 교환한다.
        구조를 «정확히» 보존하나 층이 작으면 교환 상대가 없다(그런 입원은 유지).
    """
    rng = np.random.default_rng(70 + k)
    nur, hadm = d['nur'], d['hadm']
    n_nurse = int(d['n_nurse'])
    order = np.argsort(hadm, kind='stable')
    h = hadm[order]
    bnd = np.flatnonzero(np.r_[True, h[1:] != h[:-1], True])
    starts, ends = bnd[:-1], bnd[1:]
    out = nur.copy()

    if design == 'A':
        w = np.bincount(nur, minlength=n_nurse).astype(float)
        w = w / w.sum()
        pool = np.arange(n_nurse)
        for s0, s1 in zip(starts, ends):
            idx = order[s0:s1]
            v = nur[idx]
            uq, inv = np.unique(v, return_inverse=True)   # 입원 내 배정 패턴 보존
            new = rng.choice(pool, size=len(uq), replace=False, p=w)
            out[idx] = new[inv]
        return out

    # design == 'B'
    key = {}
    for a, (s0, s1) in enumerate(zip(starts, ends)):
        v = nur[order[s0:s1]]
        key.setdefault((s1 - s0, len(np.unique(v))), []).append(a)
    for members in key.values():
        if len(members) < 2:
            continue
        m = np.array(members)
        perm = m[rng.permutation(len(m))]
        src = [nur[order[starts[b]:ends[b]]].copy() for b in perm]
        for a, vec in zip(m, src):
            out[order[starts[a]:ends[a]]] = vec
    return out
