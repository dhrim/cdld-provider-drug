"""★해석 전용 장기 적합 — 전체 자료. 표본외 판정과 «무관»한 별도 적합.

판정 수치(사다리·병동관문)는 이 적합을 쓰지 않는다.
목적은 잠재의 «방향»을 해석 가능한 수준까지 학습시키는 것뿐이다.
  순환 8회 x (ULD 2에폭 + ILD 2에폭) = 32 에폭
  잠재 표에는 weight decay 를 걸지 않는다 (노름 붕괴 방지)
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import data, cdld_core as CC

CD=paths.WORK + os.sep; OUT=CD+'uv_interp.npz'; ST=CD+'.interp_state.pt'
CYCLES, SUB, BS = 8, 2, 8192
BUDGET=float(sys.argv[1]) if len(sys.argv)>1 else 500.

d=data.prep(None); D=data.tensors(d)
tr=np.arange(len(d['y']))
m=CC.Finder(list(d['sizes']), d['cov'].shape[1], n_nurse=int(d['n_nurse']),
            n_drug=int(d['n_drug']), latent=True)
done=0
if os.path.exists(ST):
    s=torch.load(ST, weights_only=False); m.load_state_dict(s['sd']); done=s['done']
print(f'시작 — 이미 끝난 턴 {done}/{CYCLES*2}', flush=True)

codes,cov,nur,drg,y,_=D
lossf=torch.nn.BCEWithLogitsLoss()
t0=time.time(); rng=np.random.default_rng(42+done)
torch.manual_seed(42+done)
while done < CYCLES*2:
    if time.time()-t0 + 170 > BUDGET: break
    phase = 'U' if done%2==0 else 'I'
    m.set_phase(phase)
    tab = m.u_tab if phase=='U' else m.v_tab
    # ★잠재 표는 weight decay 제외
    ps=[p for n,p in m.named_parameters() if p.requires_grad and 'tab' not in n]
    opt=torch.optim.Adam([{'params':ps,'weight_decay':CC.L2},
                          {'params':[tab.weight],'weight_decay':0.0}], lr=CC.LR)
    for ep in range(SUB):
        m.train(); tot=0.
        for b in CC._batches(len(tr), BS, rng):
            j=tr[b]; opt.zero_grad()
            ls=lossf(m(codes[j],cov[j],nur[j],drg[j]), y[j]); ls.backward(); opt.step()
            tot+=ls.item()*len(j)
    done+=1
    nu=float(np.median(np.linalg.norm(m.u_tab.weight.detach().numpy(),axis=1)))
    nv=float(np.median(np.linalg.norm(m.v_tab.weight.detach().numpy(),axis=1)))
    print(f'  턴{done}/{CYCLES*2} ({phase}) loss {tot/len(tr):.5f} |u| {nu:.3f} |v| {nv:.3f} ({time.time()-t0:.0f}s)', flush=True)
    m.set_phase(None)
    torch.save({'sd':m.state_dict(),'done':done}, ST)
    np.savez(OUT, U=m.u_tab.weight.detach().numpy(), V=m.v_tab.weight.detach().numpy(),
             nurse_eff=m.n_eff.weight.detach().numpy().ravel(), turns=done,
             nurse_labels=d['nurse_labels'], drug_labels=d['drug_labels'])
print(f'\n턴 {done}/{CYCLES*2} 완료', flush=True)
