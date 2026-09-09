"""CDLD — Cyclic Dual Latent Discovery. 간호사 × 약물 투약 누락.

정본(`cdld_tutorial_fixed.ipynb` build_finder_model)의 구조를 그대로 옮긴다.
프레임워크만 Keras -> PyTorch (컨테이너에 TF 없음, CPU 2코어).

  스트림   Dense(96|64|96, swish, L2 1e-4) -> BatchNorm -> Dropout
  결합     concat -> BN -> 256 -> BN -> Drop.3 -> 128 -> BN -> Drop.3
                        -> 64 -> BN -> Drop.2 -> Dense(1, sigmoid)
  손실     BCE          최적화 Adam(5e-4)        LATENT_SIZE 64
  순환     ULD 턴(간호사 잠재만 학습, 약물 잠재 동결)
        <-> ILD 턴(약물 잠재만 학습, 간호사 잠재 동결)
           턴마다 «상대 표를 동기화»한 뒤 학습한다. 이것이 CDLD 다.
"""
import numpy as np, torch, torch.nn as nn, time

LATENT = 64
LR = 5e-4
L2 = 1e-4
DROP_S, DROP_D = 0.2, 0.3
torch.set_num_threads(2)


def swish_block(i, o, p):
    return nn.Sequential(nn.Linear(i, o), nn.SiLU(), nn.BatchNorm1d(o), nn.Dropout(p))


class Finder(nn.Module):
    """A/B/C 공통 골격.

    add_sizes : 가법 코드(교대·약물·진단챕터·병동…) 각각 1차원 임베딩  = A
    n_nurse   : 주어지면 간호사 1차원 임베딩 추가                      = B
    latent    : True 면 간호사 64 + 약물 64 이중 잠재 추가              = C
    """

    def __init__(self, add_sizes, n_cov, n_nurse=None, n_drug=None, latent=False,
                 drug_latent=False, n_ward=None):
        super().__init__()
        self.add = nn.ModuleList([nn.Embedding(s, 1) for s in add_sizes])
        for e in self.add:
            nn.init.zeros_(e.weight)
        self.cov = swish_block(n_cov, 32, 0.1)
        d = 32 + len(add_sizes)

        self.n_eff = None
        if n_nurse is not None:
            self.n_eff = nn.Embedding(n_nurse, 1)
            nn.init.zeros_(self.n_eff.weight)
            d += 1

        self.latent = latent
        self.drug_latent = drug_latent and not latent
        if latent:
            # ★이중 잠재 표 — CDLD 가 발견하는 것
            self.u_tab = nn.Embedding(n_nurse, LATENT)   # ULD
            self.v_tab = nn.Embedding(n_drug, LATENT)    # ILD
            nn.init.normal_(self.u_tab.weight, std=0.1)
            nn.init.normal_(self.v_tab.weight, std=0.1)
            self.u_str = swish_block(LATENT, 96, DROP_S)
            self.v_str = swish_block(LATENT, 96, DROP_S)
            d += 96 + 96
        elif self.drug_latent:
            # ★A·B 에도 «약물 잠재»를 둔다. 그래야 C-B 가 «간호사 쪽만»의 증분이 된다.
            #   (이것을 빼면 C-B 가 약물 쪽 추가 용량까지 담아 섞기 판에서도 양수가 된다)
            self.v_tab = nn.Embedding(n_drug, LATENT)
            nn.init.normal_(self.v_tab.weight, std=0.1)
            self.v_str = swish_block(LATENT, 96, DROP_S)
            d += 96

        # ★병동 관문 — 병동 잠재를 두어 «병동 x 약물» 상호작용이 표현되게 한다
        #   (ALS 판의 무제약 축소 셀평균 delta_W[careunit, drug] 에 대응)
        self.n_ward = n_ward
        if n_ward is not None:
            self.w_tab = nn.Embedding(n_ward, LATENT)
            nn.init.normal_(self.w_tab.weight, std=0.1)
            self.w_str = swish_block(LATENT, 96, DROP_S)
            d += 96

        self.head = nn.Sequential(
            nn.BatchNorm1d(d),
            swish_block(d, 256, DROP_D),
            swish_block(256, 128, DROP_D),
            swish_block(128, 64, DROP_S),
            nn.Linear(64, 1))

    def forward(self, codes, cov, nur, drg, wrd=None):
        parts = [e(codes[:, k]).squeeze(1, ) for k, e in enumerate(self.add)]
        z = [torch.stack(parts, 1), self.cov(cov)]
        if self.n_eff is not None:
            z.append(self.n_eff(nur))
        if self.latent:
            z.append(self.u_str(self.u_tab(nur)))
            z.append(self.v_str(self.v_tab(drg)))
        elif self.drug_latent:
            z.append(self.v_str(self.v_tab(drg)))
        if self.n_ward is not None:
            z.append(self.w_str(self.w_tab(wrd)))
        return self.head(torch.cat(z, 1)).squeeze(1)

    # --- CDLD 순환용 ---
    def set_phase(self, phase):
        """phase 'U' : 간호사 잠재만 학습 · 'I' : 약물 잠재만 학습 · None : 둘 다"""
        if not self.latent:
            return
        self.u_tab.weight.requires_grad_(phase in (None, 'U'))
        self.v_tab.weight.requires_grad_(phase in (None, 'I'))


def _batches(n, bs, rng=None):
    idx = np.arange(n) if rng is None else rng.permutation(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def fit(model, D, tr, va, cycles, sub_epochs, bs=8192, seed=42, log=None, cyclic=True,
        no_decay=(), latent_wd=None):
    """cyclic=True 이면 CDLD 순환. False 면 통상 학습(A·B 단계).

    no_decay : 가중치 감쇠(L2)를 «제외»할 파라미터 이름 접두사 튜플.
      기본값 () 이면 종전과 «완전히 동일»하게 동작한다(본실험 재현성 보존).
      직접 로짓에 들어가는 이중선형 잠재표에 Adam 의 L2 를 걸면, 자료 기울기가
      작을 때 감쇠항이 기울기를 지배하고 Adam 의 정규화 때문에 매 스텝 lr 만큼
      0 쪽으로 밀린다 -> 표가 붕괴한다. (프로젝트 내 선례: interp_fit.py)
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    codes, cov, nur, drg, y, wrd = D
    lossf = nn.BCEWithLogitsLoss()
    dev_hist = []

    def run_epochs(k, tag):
        LT = ('u_tab', 'v_tab', 'v2_tab')
        if latent_wd is not None:
            lt = [p for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith(LT)]
            ot = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith(LT)]
            opt = torch.optim.Adam([dict(params=ot, weight_decay=L2),
                                    dict(params=lt, weight_decay=latent_wd)], lr=LR)
        elif no_decay:
            nd = [p for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith(no_decay)]
            wd = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith(no_decay)]
            opt = torch.optim.Adam([dict(params=wd, weight_decay=L2),
                                    dict(params=nd, weight_decay=0.0)], lr=LR)
        else:
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                   lr=LR, weight_decay=L2)
        for ep in range(k):
            model.train()
            tot = 0.0
            for b in _batches(len(tr), bs, rng):
                j = tr[b]
                opt.zero_grad()
                out = model(codes[j], cov[j], nur[j], drg[j], None if wrd is None else wrd[j])
                ls = lossf(out, y[j])
                ls.backward()
                opt.step()
                tot += ls.item() * len(j)
            if log:
                log(f'      {tag} ep{ep+1}/{k} loss {tot/len(tr):.5f}')

    t0 = time.time()
    if not (cyclic and model.latent):
        run_epochs(cycles * sub_epochs, 'std')
    else:
        for c in range(cycles):
            model.set_phase('U'); run_epochs(sub_epochs, f'ULD c{c+1}')
            model.set_phase('I'); run_epochs(sub_epochs, f'ILD c{c+1}')
        model.set_phase(None)
    if log:
        log(f'      학습 {time.time()-t0:.0f}s')
    return dev_hist


@torch.no_grad()
def predict(model, D, idx, bs=65536):
    codes, cov, nur, drg, y, wrd = D
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        out[s:s + bs] = torch.sigmoid(model(codes[j], cov[j], nur[j], drg[j], None if wrd is None else wrd[j])).numpy()
    return out


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
