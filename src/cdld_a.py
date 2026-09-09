"""CDLD-A — 분해 정합 제약을 건 CDLD 변종 (decomposition-aligned CDLD).

CDLD 의 «정의적 기제»(두 잠재표 · 교대 ULD/ILD 턴 · 턴마다 상대표 동기화)는
그대로 두고, readout 만 구조화한다.

    eta_A = f(x)
    eta_B = f(x) + alpha_i
    eta_C = f(x) + alpha_i + u_i . v_j

f(x) 는 native CDLD 의 A 와 «완전히 동일한» 심층망이다
(가법코드 임베딩 + 공변량 스트림 + 약물 잠재 스트림 -> 비선형 head -> 스칼라).
약물 «주효과»는 f(x) 안에 남고, v2 는 «상호작용 전용» 별도 약물 임베딩이다.
ALS 의 구조(가법 기저에 약물 주효과, 저랭크 항에 별도 v)와 동일하다.

★순환 대상 — CDLD-A 에서 교대로 도는 두 표는 (u, v2), 즉 «상호작용 잠재 둘»이다.
  native 에서는 v_tab 이 주효과와 상호작용 두 역할을 겸해 ULD 턴에 함께 얼었지만,
  CDLD-A 에서는 주효과가 f(x) 안에 있으므로 baseline 은 두 턴 모두 학습된다.
  이것이 「A 를 공유하고 baseline 을 공통으로 둔다」와 양립하는 유일한 배치다.
  (구현 결정으로 사전 문서 §5 에 기록)
"""
import numpy as np, torch, torch.nn as nn, time
import cdld_core as CC

LATENT = CC.LATENT


class AlignedFinder(nn.Module):
    """kind: 'A' | 'B' | 'C'"""

    def __init__(self, add_sizes, n_cov, n_nurse, n_drug, kind, n_ward=None):
        super().__init__()
        self.kind = kind
        # --- f(x) : native A 와 동일 구조 (병동 설정이면 병동 잠재도 A 에 있다) ---
        self.base = CC.Finder(add_sizes, n_cov, n_drug=n_drug, drug_latent=True,
                              n_ward=n_ward)
        if kind in ('B', 'C'):
            self.alpha = nn.Embedding(n_nurse, 1)
            nn.init.zeros_(self.alpha.weight)
        if kind == 'C':
            self.u_tab = nn.Embedding(n_nurse, LATENT)    # ULD
            self.v2_tab = nn.Embedding(n_drug, LATENT)    # ILD (상호작용 전용)
            nn.init.normal_(self.u_tab.weight, std=0.1)
            nn.init.normal_(self.v2_tab.weight, std=0.1)
        self.latent = (kind == 'C')

    def forward(self, codes, cov, nur, drg, wrd=None):
        eta = self.base(codes, cov, nur, drg, wrd)
        if self.kind in ('B', 'C'):
            eta = eta + self.alpha(nur).squeeze(1)
        if self.kind == 'C':
            eta = eta + (self.u_tab(nur) * self.v2_tab(drg)).sum(1)
        return eta

    def set_phase(self, phase):
        if self.kind != 'C':
            return
        self.u_tab.weight.requires_grad_(phase in (None, 'U'))
        self.v2_tab.weight.requires_grad_(phase in (None, 'I'))


# ---------------- 단위시험 ----------------

def _dummy(n_nurse=7, n_drug=5, n_cov=6, sizes=(3, 5, 4)):
    torch.manual_seed(0)
    return list(sizes), n_cov, n_nurse, n_drug


def _grid_inputs(sizes, n_cov, n_nurse, n_drug, x_seed=0):
    """고정 context x 에서 (제공자 x 약물) 격자 입력을 만든다."""
    g = torch.Generator().manual_seed(x_seed)
    codes_x = torch.stack([torch.randint(0, s, (1,), generator=g) for s in sizes], 1)
    cov_x = torch.randn(1, n_cov, generator=g)
    I, J = n_nurse, n_drug
    nur = torch.arange(I).repeat_interleave(J)
    drg = torch.arange(J).repeat(I)
    codes = codes_x.repeat(I * J, 1)
    cov = cov_x.repeat(I * J, 1)
    return codes, cov, nur, drg, I, J


@torch.no_grad()
def eta_grid(model, sizes, n_cov, n_nurse, n_drug, x_seed=0):
    model.eval()
    codes, cov, nur, drg, I, J = _grid_inputs(sizes, n_cov, n_nurse, n_drug, x_seed)
    return model(codes, cov, nur, drg).reshape(I, J)


def d_stat(E):
    """E[i,j] 격자에서 이차 차분 D(i,i',j,j') 전체를 만든다."""
    # D = E[i,j] - E[i',j] - E[i,j'] + E[i',j']
    d1 = E[:, None, :, None] - E[None, :, :, None]      # i,i',j
    return d1 - d1.transpose(2, 3).diagonal(dim1=0, dim2=0) if False else \
        (E[:, None, :, None] - E[None, :, :, None]
         - E[:, None, None, :] + E[None, :, None, :])


def anova2(E, w_i=None, w_j=None):
    """격자 위 함수의 이원 분산분해. 가중 평균 기준."""
    E = np.asarray(E, float)
    I, J = E.shape
    wi = np.ones(I) / I if w_i is None else np.asarray(w_i, float) / np.sum(w_i)
    wj = np.ones(J) / J if w_j is None else np.asarray(w_j, float) / np.sum(w_j)
    mu = (wi[:, None] * wj[None, :] * E).sum()
    a = (wj[None, :] * E).sum(1) - mu
    b = (wi[:, None] * E).sum(0) - mu
    c = E - mu - a[:, None] - b[None, :]
    Va = (wi * a ** 2).sum(); Vb = (wj * b ** 2).sum()
    Vc = (wi[:, None] * wj[None, :] * c ** 2).sum()
    tot = Va + Vb + Vc
    return dict(V_provider=Va, V_drug=Vb, V_inter=Vc,
                frac_inter=(Vc / tot if tot > 0 else float('nan')),
                frac_provider=(Va / tot if tot > 0 else float('nan')),
                frac_drug=(Vb / tot if tot > 0 else float('nan')))


def unit_tests(verbose=True):
    sizes, n_cov, n_nurse, n_drug = _dummy()
    out = {}

    # --- U1 : f(x) 가 native A 와 동일 구조 ---
    a_native = CC.Finder(sizes, n_cov, n_drug=n_drug, drug_latent=True)
    a_align = AlignedFinder(sizes, n_cov, n_nurse, n_drug, 'A')
    sn = {k: tuple(v.shape) for k, v in a_native.state_dict().items()}
    sa = {k[len('base.'):]: tuple(v.shape) for k, v in a_align.state_dict().items()
          if k.startswith('base.')}
    out['U1_arch_identical'] = (sn == sa)
    out['U1_n_params_native'] = sum(p.numel() for p in a_native.parameters())
    out['U1_n_params_aligned_base'] = sum(p.numel() for n, p in a_align.named_parameters()
                                          if n.startswith('base.'))

    # --- U2 : CDLD-A 의 B 는 제공자에 대해 «정확히» 가법 ---
    mB = AlignedFinder(sizes, n_cov, n_nurse, n_drug, 'B')
    with torch.no_grad():
        mB.alpha.weight.normal_(0, 0.5)
    E = eta_grid(mB, sizes, n_cov, n_nurse, n_drug).numpy()
    D = d_stat(torch.tensor(E)).numpy()
    out['U2_maxabs_D_B'] = float(np.abs(D).max())
    out['U2_pass'] = bool(np.abs(D).max() < 1e-4)

    # --- U3 : CDLD-A 의 C 에서 D 가 (u_i-u_i')·(v_j-v_j') 와 일치 ---
    mC = AlignedFinder(sizes, n_cov, n_nurse, n_drug, 'C')
    with torch.no_grad():
        mC.alpha.weight.normal_(0, 0.5)
        mC.u_tab.weight.normal_(0, 0.3); mC.v2_tab.weight.normal_(0, 0.3)
    E = eta_grid(mC, sizes, n_cov, n_nurse, n_drug).numpy()
    D = d_stat(torch.tensor(E)).numpy()
    U = mC.u_tab.weight.detach().numpy(); V = mC.v2_tab.weight.detach().numpy()
    dU = U[:, None, :] - U[None, :, :]                 # i,i',L
    dV = V[:, None, :] - V[None, :, :]                 # j,j',L
    D_ref = np.einsum('abL,cdL->abcd', dU, dV)
    out['U3_maxabs_err'] = float(np.abs(D - D_ref).max())
    out['U3_pass'] = bool(np.abs(D - D_ref).max() < 1e-4)

    # --- U4 : native B 는 «일반적으로» 가법이 아니다 (진단, pass/fail 아님) ---
    mN = CC.Finder(sizes, n_cov, n_nurse=n_nurse, n_drug=n_drug, drug_latent=True)
    with torch.no_grad():
        mN.n_eff.weight.normal_(0, 0.5)
    E = eta_grid(mN, sizes, n_cov, n_nurse, n_drug).numpy()
    D = d_stat(torch.tensor(E)).numpy()
    out['U4_native_B_maxabs_D'] = float(np.abs(D).max())
    out['U4_native_B_sd_D'] = float(D.std())

    # --- U5 : anova2 정합 (가법 격자면 frac_inter == 0) ---
    Ea = np.arange(7)[:, None] + np.arange(5)[None, :] * 0.5
    out['U5_additive_frac_inter'] = anova2(Ea)['frac_inter']
    out['U5_pass'] = bool(abs(anova2(Ea)['frac_inter']) < 1e-12)

    if verbose:
        for k, v in out.items():
            print(f'  {k:32s} {v}')
    return out


if __name__ == '__main__':
    print('=== CDLD-A 단위시험 ===')
    r = unit_tests()
    ok = r['U1_arch_identical'] and r['U2_pass'] and r['U3_pass'] and r['U5_pass']
    print('\n판정:', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)
