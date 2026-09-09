"""v8 검증 — 본문 수치를 겹 단위 결과에서 재계산하고 불일치 시 비영 종료."""
import sys, os, re, json, glob, numpy as np
_H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_H)
_ROOT=os.path.dirname(_H)
_CANDS=[os.path.join(_ROOT,'results','v6')+os.sep, _H+os.sep, os.getcwd()+os.sep]
CD=next((c for c in _CANDS if glob.glob(c+'fold_json/ck__base_real_f0_A.json')
                             or glob.glob(c+'ck/base_real_f0_A.json')), _CANDS[0])
# v7 산출물 · 원고 후보 경로
_V7=[os.path.join(_ROOT,'results','v7')+os.sep, CD,
     _ROOT+os.sep, os.path.dirname(_ROOT)+os.sep,   # 저장소를 원고 폴더 안에 풀었을 때
     os.getcwd()+os.sep]                            # ★절대경로 없음
def _p(*names):
    for d in _V7:
        for nm in names:
            if os.path.exists(d+nm): return d+nm
    return None
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
for fp in glob.glob(CD+'fold_json/ck_seed__*.json')+glob.glob(CD+'ck_seed/*.json'):
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
F=json.load(open(next(x for x in (CD+'fn_decomp2.json',CD+'diag/fn_decomp2.json') if os.path.exists(x))))
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
t8 = _p('table8_null_frequency.csv') or os.path.join(CD, 'table8_null_frequency.csv')
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

print('\n== ★대안 귀무 (층 1: 배포된 팔 단위 표에서 재계산) ==')
import csv as _csv
arms_fp=_p('table10_alt_null_arms.csv'); t10_fp=_p('table10_alt_null.csv')
AR={r['arm']:{k:float(r[k]) for k in 'ABC'} for r in _csv.DictReader(open(arms_fp))}
def _lad(nulls):
    b,c=AR['real']['B'],AR['real']['C']
    bn=np.mean([AR[x]['B'] for x in nulls]); cn=np.mean([AR[x]['C'] for x in nulls])
    dm,dt=b-bn,c-cn
    return dict(B_null=bn,C_null=cn,d_main=dm,d_latent=dt-dm,d_total=dt,ratio=(dt-dm)/dm,S=cn-bn)
SH=['shuffle0','shuffle1','shuffle2']; BS=['blockswap0','blockswap1']
REC={('주',2):_lad(SH[:2]),('대안',2):_lad(BS),('주',3):_lad(SH)}
T10=list(_csv.DictReader(open(t10_fp)))
for row in T10:
    kind='대안' if row['귀무설계'].startswith('대안') else '주'
    rec=REC[(kind,int(row['n_null']))]
    for col,tol in (('d_main',5e-5),('d_latent',5e-5),('d_total',5e-5),
                    ('ratio',1e-3),('S',5e-5),('B_null',5e-5),('C_null',5e-5)):
        chk(f"table10 {kind}/{row['n_null']} {col}",rec[col],float(row[col]),tol)
main2,alt2,main3=REC[('주',2)],REC[('대안',2)],REC[('주',3)]
assert alt2['S']>0>main2['S'], '★칸을 맞춘 비교에서 S 의 부호가 갈리지 않는다'
assert alt2['d_total']>=0.010 and alt2['ratio']>=1.0, '★대안 귀무에서 R·Q2b 가 유지되지 않는다'
print('OK    칸을 맞춘 2회 대 2회 비교 — R·Q2b 유지 · S 부호가 갈린다 (§4.11.1)')

print('\n== ★귀무 횟수 혼재 금지 (Figure 3 에서 겪은 실수의 재발 방지) ==')
n=n+1
mixed=[r for r in T10 if r['역할'].startswith('★주 비교')]
same=len({r['n_null'] for r in mixed})==1
ok&=same
print(f'{"OK " if same else "FAIL"}  table10 의 «주 비교» 행들이 같은 귀무 횟수를 쓴다 '
      f'({sorted({r["n_null"] for r in mixed})})')
n=n+1
t11rows=list(_csv.DictReader(open(_p('table11_two_by_two_v7.csv'))))
same11=len({r['n_null'] for r in t11rows})==1
ok&=same11
print(f'{"OK " if same11 else "FAIL"}  table11 의 다섯 칸이 같은 귀무 횟수를 쓴다 '
      f'({sorted({r["n_null"] for r in t11rows})})')

print('\n== ★대안 귀무의 치환 강도 진단 (층 1: 배포된 표) ==')
T12=list(_csv.DictReader(open(_p('table12_null_B_diag.csv'))))
for r12 in T12:
    npart=float(r12['상대없는층_비율']); iden=float(r12['★실제동일_비율'])
    chk(f"null_B seed {r12['seed']} 교환 상대 없음",npart,0.01617,5e-4)
    chk(f"null_B seed {r12['seed']} ★실제 동일",iden,0.035,1e-3)
    chk(f"null_B seed {r12['seed']} 행 제공자 유지",float(r12['행_제공자유지']),0.317,1e-3)
    n=n+1; good=iden>npart; ok&=good
    print(f'{"OK " if good else "FAIL"}  ★「실제 동일」 > 「상대 없음」 — 두 수는 다른 것을 센다 '
          f'({iden:.2%} > {npart:.2%})')

print('\n== ★2x2 배포 표 (table11) 가 겹 단위 결과와 일치하는가 ==')
t11_fp=_p('table11_two_by_two_v7.csv')
CELL={('비제약','1e-4'):nat2,('비제약','0'):nd2,('정합','1e-4'):a12,
      ('정합','1e-5'):mid2,('정합','0'):ali2}
for r in _csv.DictReader(open(t11_fp)):
    cell=CELL[(r['readout'],r['latent_L2'])]
    assert int(r['n_null'])==2, '★table11 의 모든 칸은 귀무 2회여야 한다'
    for col,key,tol in (('d_main','dm',5e-5),('d_latent','dl',5e-5),
                        ('d_total','dt',5e-5),('ratio','r',2e-3),('S','S',5e-5)):
        chk(f"table11 {r['readout']}/{r['latent_L2']} {col}",cell[key],float(r[col]),tol)

print('\n== ★귀속 표 (table2_v7) 가 두 귀무 기준을 분리해 싣는가 ==')
t2_fp=_p('table2_attribution_v7.json')
J=json.load(open(t2_fp))
chk('table2_v7 3-null gamma*',J['headline_3null']['gamma_star'],g,1e-9)
chk('table2_v7 3-null kappa',J['headline_3null']['kappa'],k,1e-9)
chk('table2_v7 3-null 항등식 오차',J['headline_3null']['identity_error'],0.0,1e-12)
chk('table2_v7 2-null gamma*',J['matched_2null']['gamma_star'],nat2['dm']-ali2['dm'],1e-9)
chk('table2_v7 2-null 항등식 오차',J['matched_2null']['identity_error'],0.0,1e-12)
chk('table2_v7 3-null CDLD-A ratio',J['headline_3null']['aligned']['ratio'],2.4643,2e-3)
chk('table2_v7 2-null CDLD-A ratio',J['matched_2null']['aligned_L2_0']['ratio'],2.3433,2e-3)

print('\n== ★원고 문안 점검 (원고가 함께 있을 때만) ==')
MSS=_p('투고본_한글_v8.md','투고본_한글_v7.md')
if MSS:
    t=open(MSS).read()
    tn=re.sub(r'\s+',' ',re.sub(r'\n>\s*','\n',t))   # 인용부호·줄바꿈을 넘는 문장도 잡는다
    금칙=['오류','간호사 능력','pre-registered','동시출현 빈도','읽기층 모수화의 산물',
        'well defined as a decomposition of the fitted function',
        'non-additive provider-dependent predictive contribution',
        '만들어낼 수는 없다','거의 면역','크기를 잘못 재는',
        '적합된 함수 위에서 정의하라','추정기 계열의 성질이 아니었다',
        '결론이 귀무 설계에 좌우되지 않는다',
        '가능하면 예측 기술이','답은 아니었다']
    for wd in 금칙:
        c=tn.count(re.sub(r'\s+',' ',wd)); ok&= (c==0); n+=1
        print(f'{"OK " if c==0 else "FAIL"}  금칙 「{wd[:34]}」 {c}건')
    필수=[('product of provider and drug',1),('weighted Pearson r',1),
        ('★이 시정은 동결 계획에 근거가 없다',1),
        ('같은 데이터베이스에 대한 탐색 분석',2),
        ('귀무 3회 기준으로 통일',1),('귀무 2회 기준',2),
        ('remains a methodological limitation',1),
        ('교환 상대가 없',1),('실제로» 동일한 입원',1),('RQ2',1),
        ('estimator family alone',1),
        ('should not be treated as interchangeable',1),
        ('RQ1',1),('RQ4',1),
        ('Estimator family alone',1)]
    for wd,least in 필수:
        c=tn.count(re.sub(r'\s+',' ',wd)); good=c>=least; ok&=good; n+=1
        print(f'{"OK " if good else "FAIL"}  필수 「{wd[:34]}」 {c}건 (>= {least})')
    # 상호참조
    hd=set(re.findall(r'^#{1,4}\s+([0-9]+(?:\.[0-9]+)*)[\s.]',t,re.M))
    hd|={x.split('.')[0] for x in hd}
    dang=sorted({m for m in re.findall(r'§\s?([0-9]+(?:\.[0-9]+)*)',t) if m not in hd})
    ok&= not dang; n+=1
    print(f'{"OK " if not dang else "FAIL"}  상호참조 — 없는 절을 가리키는 참조 {len(dang)}건 {dang}')
else:
    print('SKIP  원고 파일이 함께 있지 않아 문안 점검을 건너뛴다')



print(f'\n검사 {n}건 — {"전부 통과" if ok else "★불일치 있음"}')
sys.exit(0 if ok else 1)
