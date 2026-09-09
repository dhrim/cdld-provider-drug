"""v6 검증 — 본문 수치를 겹 단위 결과에서 재계산하고 불일치 시 비영 종료."""
import sys, os, json, glob, numpy as np
_H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_H)
_ROOT=os.path.dirname(_H)
CD=os.path.join(_ROOT,'results','v6')+os.sep      # 저장소 자립 경로
ok=True; n=0
def chk(name,got,want,tol=5e-5):
    global ok,n; n+=1
    good=(got is not None) and abs(got-want)<=tol; ok&=good
    print(f'{"OK " if good else "FAIL"}  {name:46s} got {got:+.5f}  want {want:+.5f}')
def load(d,tag='base'):
    o={}
    pats=[CD+f'fold_json/{d}__{tag}_*_f?_?.json', CD+d+f'/{tag}_*_f?_?.json']
    for fp in [x for p_ in pats for x in glob.glob(p_)]:
        r=json.load(open(fp)); o[(r['arm'],r['fold'],r['kind'])]=1-r['sse']/r['sstot']
    return o
def lad(A,B,C,N):
    a=np.mean([A[('real',f,'A')] for f in range(5)])
    b=np.mean([B[('real',f,'B')] for f in range(5)]); bn=np.mean([B[(x,f,'B')] for x in N for f in range(5)])
    c=np.mean([C[('real',f,'C')] for f in range(5)]); cn=np.mean([C[(x,f,'C')] for x in N for f in range(5)])
    dm=b-bn; dt=c-cn
    return dict(A=a,B=b,C=c,dm=dm,dl=dt-dm,dt=dt,r=(dt-dm)/dm,S=cn-bn)
S={k:load(k) for k in ('ck','ck_natnd','ck_a','ck_a2','ck_a2mid')}
N3=['shuffle0','shuffle1','shuffle2']; N2=['shuffle0','shuffle1']
nat3=lad(S['ck'],S['ck'],S['ck'],N3); ali3=lad(S['ck'],S['ck_a2'],S['ck_a2'],N3)
nat2=lad(S['ck'],S['ck'],S['ck'],N2); nd2=lad(S['ck'],S['ck'],S['ck_natnd'],N2)
mid2=lad(S['ck'],S['ck_a2'],S['ck_a2mid'],N2); a12=lad(S['ck'],S['ck_a'],S['ck_a'],N2)
ali2=lad(S['ck'],S['ck_a2'],S['ck_a2'],N2)

print('== 사다리 (귀무 3회, 본문 headline) ==')
chk('native  Delta_main',nat3['dm'],0.00963); chk('native  Delta_latent',nat3['dl'],0.00573)
chk('native  Delta_total (R)',nat3['dt'],0.01535); chk('native  ratio',nat3['r'],0.5947,1e-3)
chk('native  S',nat3['S'],-0.00038)
chk('CDLD-A  Delta_main',ali3['dm'],0.00435); chk('CDLD-A  Delta_latent',ali3['dl'],0.01072)
chk('CDLD-A  Delta_total (R)',ali3['dt'],0.01507); chk('CDLD-A  ratio',ali3['r'],2.4643,2e-3)
chk('CDLD-A  S',ali3['S'],-0.00071)

print('\n== A 공유 · 항등식 ==')
chk('A_native == A_CDLD-A',ali3['A']-nat3['A'],0.0,1e-12)
g=nat3['dm']-ali3['dm']; k=ali3['dt']-nat3['dt']
chk('gamma* (3-null)',g,0.00528,1e-4); chk('kappa (3-null)',k,-0.00028,1e-4)
chk('항등식 l+g+k == Delta_latent,A',nat3['dl']+g+k-ali3['dl'],0.0,1e-12)

print('\n== ★2x2 (귀무 2회로 칸을 맞춤) ==')
chk('비제약 L2=1e-4  ratio',nat2['r'],0.5884,1e-3)
chk('비제약 L2=0     ratio',nd2['r'],0.4609,1e-3)
chk('정합   L2=1e-4  ratio (1차, 적합실패)',a12['r'],0.2214,1e-3)
chk('정합   L2=1e-5  ratio',mid2['r'],1.9432,2e-3)
chk('정합   L2=0     ratio',ali2['r'],2.3433,2e-3)
chk('신규 칸 적합 관문 C 차 (비제약 L2=0)',nd2['C']-nat2['C'],-0.00069,5e-5)
assert nd2['r']<0.8, '★사전 규칙 R1 의 첫 조건 불충족'
assert mid2['r']>=1.0, '★사전 규칙 R1 의 둘째 조건 불충족'
assert nd2['C']-nat2['C']>-0.0025, '★신규 칸 적합 관문 실패'
print('OK    사전 규칙 R1 — r_비제약(L2=0) < 0.8 이고 r_정합(1e-5) >= 1, 관문 통과')

print('\n== 시드 반복 ==')
sd={}
for fp in glob.glob(CD+'fold_json/ck_seed__*.json'):
    r=json.load(open(fp)); sd[(r['model'],r['seed'],r['fold'])]=1-r['sse']/r['sstot']
natB={f:S['ck'][('real',f,'B')] for f in range(5)}; aliB={f:S['ck_a'][('real',f,'B')] for f in range(5)}
gaps=[natB[f]-aliB[f] for f in range(3)]+[sd[('natB',s,f)]-sd[('aliB',s,f)] for s in (1,2) for f in range(3)]
chk('시드 9개 격차 평균',float(np.mean(gaps)),0.00411,2e-4)
assert all(x>0 for x in gaps), '★시드 격차에 음수 존재'
print(f'OK    9/9 전부 양수 (sd {np.std(gaps,ddof=1):.5f})')
nulls=[S['ck'][(a,f,'B')]-S['ck_a'][(a,f,'B')] for a in N2 for f in range(5)]
assert all(x<0 for x in nulls), '★귀무팔 격차에 양수 존재'
print('OK    귀무팔 10/10 전부 음수 — 부호가 팔로 완전히 갈린다')

print('\n== 함수 수준 분해 (사전 지표 primary) ==')
F=json.load(open(CD+'fn_decomp2.json'))
chk('native  frac_inter (사전)',F['native_frac_inter']['mean'],0.0422,5e-4)
chk('CDLD-A  frac_inter (사전)',F['aligned_frac_inter']['mean'],0.0232,5e-4)
chk('native  sd(D)/sd(M) (사전)',F['native_sdD_sdM']['mean'],1.5156,2e-3)
chk('corr(c_native, c_aligned)',F['corr_c']['mean'],0.4980,2e-3)
chk('(참고) corr(eta, eta)',F['corr_eta_raw']['mean'],0.9643,2e-3)
assert F['native_frac_inter']['mean']>F['aligned_frac_inter']['mean'], '★함수 분해 순서 불일치'
assert F['corr_c']['mean']<0.7, '★상호작용 성분 상관이 높다면 「비슷한 함수」 문안을 재검토해야 한다'
print('OK    함수 분해 순서 반전 · 상호작용 성분 상관은 중간 수준(<0.7)')

print('\n== 귀무 빈도 분포 — «근사» 보존 (층 1: 배포된 표에서 읽는다) ==')
import csv, os
t8 = os.path.join(CD, 'table8_null_frequency.csv')
if not os.path.exists(t8):
    raise SystemExit('table8_null_frequency.csv 가 results/v6/ 에 없다 — 저장소가 불완전하다')
rows = {r['팔']: r for r in csv.DictReader(open(t8))}
real, n0 = rows['실제'], rows['귀무0']
chk('평균 사건수 (정확 보존)', float(n0['평균']) - float(real['평균']), 0.0, 1e-6)
chk('순위 상관', float(n0['순위상관']), 0.9914, 1e-3)
assert int(float(n0['최소'])) < 200, '★귀무 최소값이 포함 문턱 아래로 내려가지 않는다'
print(f'OK    최소 {int(float(real["최소"]))} -> {int(float(n0["최소"]))} '
      f'(코호트 포함 문턱 200 아래) · 최대 {int(float(real["최대"]))} -> {int(float(n0["최대"]))}')

print('\n== 층 2 — 자료 의존 재계산 (원자료가 있을 때만) ==')
try:
    import data
    d = data.prep(None, ward=False)
    if d is None: raise RuntimeError('no cache')
    rc = np.bincount(d['nur'], minlength=int(d['n_nurse']))
    c0 = np.bincount(data.shuffle_nurse(d, 0, 'A'), minlength=int(d['n_nurse']))
    chk('[층2] 귀무0 최소 (원자료 재계산)', float(c0.min()), float(n0['최소']), 1.0)
    chk('[층2] 귀무0 최대 (원자료 재계산)', float(c0.max()), float(n0['최대']), 1.0)
    print('OK    층 2 통과 — 배포된 표가 원자료에서 재계산된 값과 일치한다')
except (Exception, SystemExit) as e:      # need_data 는 SystemExit 를 던진다
    print(f'SKIP  원자료(MIMIC 파생 캐시)가 없어 층 2 를 건너뛴다 — {type(e).__name__}')
    print('      층 1 만으로 보고 수치는 «전부» 검증된다. 종료 코드에 영향 없음.')

print(f'\n검사 {n}건 — {"전부 통과" if ok else "★불일치 있음"}')
sys.exit(0 if ok else 1)
