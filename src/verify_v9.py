"""v9 검증 — 본문 수치를 겹 단위 결과에서 재계산하고 불일치 시 비영 종료."""
import sys, os, re, json, glob, numpy as np
_H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_H)
_ROOT=os.path.dirname(_H)
_CANDS=[os.path.join(_ROOT,'results','v6')+os.sep, _H+os.sep, os.getcwd()+os.sep]
CD=next((c for c in _CANDS if glob.glob(c+'fold_json/ck__base_real_f0_A.json')
                             or glob.glob(c+'ck/base_real_f0_A.json')), _CANDS[0])
# v7 산출물 · 원고 후보 경로
_V7=[os.path.join(_ROOT,'results','v9')+os.sep, os.path.join(_ROOT,'results','v7')+os.sep, CD,
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

print('\n== ★집중도 지표 (D3) — 사전 고정한 네 값만 있는가 ==')
t13=_p('table13_concentration.csv'); t14=_p('table14_replica_family.csv')
R13=list(_csv.DictReader(open(t13)))
n=n+1; good=len(R13)==2; ok&=good
print(f'{"OK " if good else "FAIL"}  표에 «두 줄»만 있다 (capture · observed rate) — {len(R13)}줄')
cap=[r for r in R13 if r['지표'].startswith('top-10% capture')][0]
chk('A  top-10% capture',      float(cap['A (제공자 없음)']),39.1,.05)
chk('C  top-10% capture',      float(cap['C (제공자 잠재 포함)']),40.1,.05)
chk('C 귀무 3팔 평균 capture',   float(cap['C 귀무 3팔 평균']),38.9,.05)
chk('real incremental gain',   float(cap['real incremental gain (C−A)']),1.0,.05)
chk('귀무 보정 incremental',    float(cap['provider-shuffle-adjusted incremental gain']),1.2,.05)
_j13=_p('table13_concentration.json')
if _j13:
    J13=json.load(open(_j13)); arms=[v['capture'] for v in J13['C_nulls'].values()]
else:
    arms=[float(cap[f'C {a}']) for a in ('shuffle0','shuffle1','shuffle2')]
n=n+1; good=(max(arms)-min(arms))<=0.10; ok&=good
print(f'{"OK " if good else "FAIL"}  귀무 3팔의 폭 {max(arms)-min(arms):.3f} pp (<= 0.10)')
n=n+1; good=float(cap['C 귀무 3팔 평균'])<float(cap['A (제공자 없음)']); ok&=good
print(f'{"OK " if good else "FAIL"}  ★귀무 C 가 A 보다 «낮다» — 귀무 보정 증분이 원 증분보다 크다')
R14=list(_csv.DictReader(open(t14)))
n=n+1; good=len(R14)==25; ok&=good
print(f'{"OK " if good else "FAIL"}  복제 계열 25단위가 전부 있다 — {len(R14)}단위')
import statistics as _st
for kind,arm,cnt in (('A','real',5),('C',None,20)):
    rr=[r for r in R14 if r['kind']==kind and (arm is None or r['arm']==arm)]
    d=[abs(float(r['diff'])) for r in rr if r['diff']]
    n=n+1; good=len(rr)==cnt and max(d)<0.0041/2; ok&=good
    print(f'{"OK " if good else "FAIL"}  {kind} {len(rr)}단위 · 기록 대비 최대|차이| {max(d):.5f} (겹간 SD 0.0041 의 절반 미만)')
seeds={('A',2000),('C',1500)}
n=n+1; good=all(int(r['init_seed'])-int(r['fold'])==dict(seeds)[r['kind']] for r in R14); ok&=good
print(f'{"OK " if good else "FAIL"}  ★초기화 시드가 선언된 정책(BASE[kind]+fold)을 따른다 · 팔에 무관')

print('\n== ★반전폭 — 본문 표 2 는 «복제 수준» 통계량이다 ==')
_T17=_p('table17_reversal_replicate.csv')
if _T17:
    _r17={r['통계']: r for r in _csv.DictReader(open(_T17))}
    for _lab,_o,_nl,_ratio in (('중앙값',0.78279,0.35431,2.2094),('75분위',1.11721,0.49866,2.2404),
                               ('90분위',1.54788,0.66312,2.3343),('95분위',2.01683,0.77817,2.5916),
                               ('99분위',3.07597,1.06029,2.9010),('평균',0.90159,0.39077,2.3072)):
        chk(f'반전폭 {_lab} 관측', float(_r17[_lab]['관측']), _o, 1e-4)
        chk(f'반전폭 {_lab} 귀무 복제평균', float(_r17[_lab]['귀무_복제평균']), _nl, 1e-4)
        chk(f'반전폭 {_lab} 배', float(_r17[_lab]['배']), _ratio, 1e-3)
    n=n+1; good=all(int(_r17[k]['넘지못한복제'])==0 for k,_,_,_ in
                    (('중앙값',0,0,0),('75분위',0,0,0),('90분위',0,0,0),
                     ('95분위',0,0,0),('99분위',0,0,0),('평균',0,0,0))); ok&=good
    print(f'{"OK " if good else "FAIL"}  ★여섯 통계량 전부에서 관측이 귀무 복제 20회를 «모두» 넘는다')
else:
    print('SKIP  table17_reversal_replicate.csv 가 없다')

print('\n== ★반전폭 — 배포된 out_09 과 표 17 이 같은 귀무에서 나왔나 ==')
_T9=_p('out_09_reversal_span.csv')
if _T9:
    _r9={r['통계']: r for r in _csv.DictReader(open(_T9))}
    for _lab,_o,_nl in (('중앙값',0.78279,0.35426),('75분위',1.11721,0.49862),
                        ('90분위',1.54788,0.66263),('95분위',2.01683,0.77902),
                        ('99분위',3.07597,1.05813),('평균',0.90159,0.39079),
                        ('최대',4.51045,1.70495)):
        chk(f'반전폭 {_lab} 관측', float(_r9[_lab]['관측']), _o, 1e-4)
        chk(f'반전폭 {_lab} 귀무', float(_r9[_lab]['귀무']), _nl, 1e-4)
else:
    print('SKIP  out_09_reversal_span.csv 가 없다')

    if _T17:
        n=n+1
        _d=[abs(float(_r9[k]['귀무'])-float(_r17[k]['귀무_pooled'])) for k in
            ('중앙값','75분위','90분위','95분위','99분위','평균')]
        good=max(_d)<1e-3; ok&=good
        print(f'{"OK " if good else "FAIL"}  ★out_09(pooled) 과 표 17 의 pooled 열이 «같은 귀무»다 '
              f'(최대차 {max(_d):.5f})')

print('\n== ★중앙값 몬테카를로 표준화 분리 (D7) ==')
_T16=_p('table16_mc_separation.csv')
if _T16:
    _r16={r['통계량']: r for r in _csv.DictReader(open(_T16))}
    chk('D7 중앙값 T_obs', float(_r16['중앙값']['T_obs']), 0.78279, 1e-4)
    chk('D7 중앙값 귀무평균', float(_r16['중앙값']['귀무평균']), 0.35426, 1e-4)
    chk('D7 중앙값 표준화분리', float(_r16['중앙값']['표준화분리']), 34.447, 5e-2)
    n=n+1; good=int(_r16['중앙값']['넘지못한복제'])==0; ok&=good
    print(f'{"OK " if good else "FAIL"}  ★관측 중앙값이 귀무 복제 20회 «전부»보다 크다')
    # ★표 2 의 귀무와 D7 의 귀무가 «같은 귀무»여야 한다 (출처 혼재 방지)
    if _T9:
        n=n+1; good=abs(float(_r16['중앙값']['귀무평균'])-float(_r9['중앙값']['귀무']))<0.02
        ok&=good
        print(f'{"OK " if good else "FAIL"}  ★표 2 의 귀무 중앙값과 D7 의 복제 중앙값 평균이 일치한다 '
              f'({float(_r9["중앙값"]["귀무"]):.3f} 대 {float(_r16["중앙값"]["귀무평균"]):.3f})')
else:
    print('SKIP  table16_mc_separation.csv 가 없다')

print('\n== ★제공자 무관 약물 분할 (D6) ==')
_T15=_p('table15_blind_split.csv')
if _T15:
    _r15={r['분할']: r for r in _csv.DictReader(open(_T15))}
    for _k,_med,_mean in (('P1',1.8686,1.8273),('P2',2.0437,2.0604)):
        chk(f'{_k} 반전폭 배(중앙)', float(_r15[_k]['배_중앙']), _med, 1e-3)
        chk(f'{_k} 반전폭 배(평균)', float(_r15[_k]['배_평균']), _mean, 1e-3)
    n=n+1; good=all(float(_r15[k]['배_중앙'])>1.3 and float(_r15[k]['배_평균'])>1.3
                    for k in ('P1','P2')); ok&=good
    print(f'{"OK " if good else "FAIL"}  ★두 제공자 무관 분할 모두 사전 판정 문턱 1.3 을 넘는다')
else:
    print('SKIP  table15_blind_split.csv 가 없어 D6 검사를 건너뛴다')

print('\n== ★층 1 감사 — 본문에 실렸으나 재계산되지 않던 수 ==')
_FG0=_p('table3_fitgate.json')
if _FG0:
    _fg0=json.load(open(_FG0))
    chk('§4.1 사건 수', _fg0['n_rows'], 4362548, 0.5)
    chk('§4.1 환자 수', _fg0['n_patients'], 98841, 0.5)
_T12b=_p('table12_null_B_diag.csv')
if _T12b:
    chk('§4.1 입원 수', float(list(_csv.DictReader(open(_T12b)))[0]['입원']), 158422, 0.5)
_T8b=_p('table8_null_frequency.csv')
if _T8b:
    _t8=list(_csv.DictReader(open(_T8b)))
    chk('§3.5 실제 팔 평균 사건수', float(_t8[0]['평균']), 4660.84, 5e-2)
    n=n+1; good=len(_t8)==4; ok&=good
    print(f'{"OK " if good else "FAIL"}  빈도표가 실제 + 귀무 3팔 = 4행이다')

_T1=_p('table1_ladders_v9.csv')
if _T1:
    _r1={r['설정']: r for r in _csv.DictReader(open(_T1))}
    _nat=_r1['순환 이중 잠재 (native)']
    chk('§4.5 A (BSS)', float(_nat['A']), 0.13608, 1e-5)
    chk('§4.5 B (BSS)', float(_nat['B']), 0.14431, 1e-5)
    chk('§4.5 C (BSS)', float(_nat['C']), 0.14965, 1e-5)
    chk('§4.5 AUC (C)', float(_nat['auc_C']), 0.79494, 1e-4)
    for _lab,_key,_dt,_keep in (('비제약','native — 병동 통제',0.01286,0.84),
                                ('정합','CDLD-A — 병동 통제',0.01157,0.77),
                                ('ALS','ALS — 병동 통제',0.01293,0.69)):
        _base={'비제약':0.01535,'정합':0.01507,'ALS':0.01875}[_lab]
        chk(f'§4.6 {_lab} 병동 통제 Δ_total', float(_r1[_key]['d_total']), _dt, 1e-5)
        chk(f'§4.6 {_lab} 잔존율', round(float(_r1[_key]['d_total'])/_base, 2), _keep, 5e-3)
else:
    print('SKIP  table1_ladders_v9.csv 가 없다')

_FG=_p('table3_fitgate.json')
if _FG:
    _fg=json.load(open(_FG))
    chk('§4.8 적합 관문 ΔBSS', _fg['dBSS_point'], 0.00058, 1e-5)
    chk('§4.8 관문 CI 하한', _fg['ci_lo'], 0.00018, 1e-5)
    chk('§4.8 관문 CI 상한', _fg['ci_hi'], 0.00100, 1e-5)
else:
    print('SKIP  table3_fitgate.json 이 없다')

_T9b=_p('table9_function_decomp_v6.json')
if _T9b:
    _d9=json.load(open(_T9b))
    chk('§5.3 native Var(c)/Var(a)', _d9['native_inter_over_prov']['mean'], 0.784, 1e-3)
    chk('§5.3 정합 Var(c)/Var(a)', _d9['aligned_inter_over_prov']['mean'], 0.528, 1e-3)
_T5b=_p('table5_D_native_B.json')
if _T5b:
    chk('§5.2 B 단계 sd(D)/sd(M)', json.load(open(_T5b))['supported_N20']['dm'], 0.463, 1e-3)

_C5=_p('out_05_cluster_summary.csv')
if _C5:
    _c5=sorted(_csv.DictReader(open(_C5)), key=lambda r: float(r['누락률']))
    for _i,(_nd,_rt) in enumerate([(63,5.3),(88,7.2),(64,8.6),(57,16.2)]):
        chk(f'§4.4 Cluster {_i+1} 약물 수', float(_c5[_i]['약물수']), _nd, 0.5)
        chk(f'§4.4 Cluster {_i+1} 관측 비시행률(%)', round(float(_c5[_i]['누락률'])*100, 1), _rt, 5e-2)
    n=n+1; _rr=[float(r['누락률']) for r in _c5]; good=_rr==sorted(_rr); ok&=good
    print(f'{"OK " if good else "FAIL"}  ★Cluster 번호가 관측 비시행률 «오름차순»이다')
else:
    print('SKIP  out_05_cluster_summary.csv 가 없다')

_P6=_p('out_06_provider_profiles.csv')
if _P6:
    _p6={r['nurse']: r for r in _csv.DictReader(open(_P6))}
    _ORD=[f'cluster_{i+1}' for i in range(4)]   # 관측률 오름차순 canonical
    for _who,_n,_rate,_rat in (('A',5587,0.10220,[3.38,0.92,1.00,1.09]),
                               ('B',4200,0.09905,[0.30,2.34,1.63,1.52])):
        chk(f'§4.3 {_who} n', float(_p6[_who]['n']), _n, 0.5)
        chk(f'§4.3 {_who} 전체 비시행률', float(_p6[_who]['overall']), _rate, 1e-5)
        for _i,_v in enumerate(_rat):
            chk(f'§4.3 {_who} Cluster {_i+1} 배수', float(_p6[_who][_ORD[_i]]), _v, 5e-3)
else:
    print('SKIP  out_06_provider_profiles.csv 가 없다')

_T13b=_p('table13_concentration.csv')
if _T13b:
    _t13=list(_csv.DictReader(open(_T13b)))
    _rate=[r for r in _t13 if 'observed' in r['지표']][0]
    chk('§4.5 상위 10% 관측률 A', float(_rate['A (제공자 없음)']), 363.5, 5e-2)
    chk('§4.5 상위 10% 관측률 C', float(_rate['C (제공자 잠재 포함)']), 372.4, 5e-2)
    chk('§4.5 관측률 원 증분', float(_rate['real incremental gain (C−A)']), 8.8, 5e-2)
    chk('§4.5 관측률 귀무 보정', float(_rate['provider-shuffle-adjusted incremental gain']), 10.7, 5e-2)

print('\n== ★층 B — 해석용 장기 적합의 재현 (D8) ==')
_T19=_p('table19_repro_check.csv')
if _T19:
    _r19=list(_csv.DictReader(open(_T19)))[0]
    chk('D8 가중 멤버십 일치도', float(_r19['가중일치도']), 0.296, 5e-3)
    chk('D8 최대/최소 비 재실행', float(_r19['최대최소비_재실행']), 2.60, 5e-2)
    chk('D8 반전폭 배 재실행', float(_r19['반전폭배_재실행']), 2.36, 5e-2)
    n=n+1; good=float(_r19['가중일치도'])<0.70; ok&=good
    print(f'{"OK " if good else "FAIL"}  ★멤버십 일치도가 사전 문턱 0.70 «미달»이다 — 본문이 그 사실을 적어야 한다')
    n=n+1; good=(float(_r19['최대최소비_재실행'])>=2.0 and float(_r19['반전폭배_재실행'])>=1.3)
    ok&=good
    print(f'{"OK " if good else "FAIL"}  ★관측률 격차와 반전폭 분리는 재적합에서도 문턱을 넘는다')
else:
    print('SKIP  table19_repro_check.csv 가 없다')

print('\n== ★원고 문안 점검 (원고가 함께 있을 때만) ==')
MSS=_p('투고본_한글_v9.md','투고본_한글_v8.md'); SUP=_p('보충자료_한글_v9.md')
if MSS:
    _raw=open(MSS).read()
    _STAMP=re.compile(r'^<!-- ▼.*?<!-- ▲ 작업 원고 표지 끝 ▲ -->\n\n---\n\n', re.S)
    n=n+1; good=bool(_STAMP.match(_raw)); ok&=good
    print(f'{"OK " if good else "FAIL"}  작업 원고 표지가 «맨 앞»에 있고 삭제 범위가 표시돼 있다')
    t=_STAMP.sub('', _raw)                       # 표지는 원고가 아니므로 떼어낸다
    _sup=_STAMP.sub('', open(SUP).read()) if SUP else ''
    ts=t + ('\n'+_sup if SUP else '')
    norm=lambda x: re.sub(r'\s+',' ',re.sub(r'\n>\s*','\n',x))
    tn=norm(t)          # 본문만
    tb=norm(ts)         # 본문 + 보충자료
    if not SUP: print('      (보충자료가 함께 있지 않아 «본문 또는 보충» 항목은 본문만 본다)')
    금칙=['오류','간호사 능력','pre-registered','동시출현 빈도','읽기층 모수화의 산물',
        'well defined as a decomposition of the fitted function',
        'non-additive provider-dependent predictive contribution',
        '만들어낼 수는 없다','거의 면역','크기를 잘못 재는',
        '적합된 함수 위에서 정의하라','추정기 계열의 성질이 아니었다',
        '결론이 귀무 설계에 좌우되지 않는다',
        '가능하면 예측 기술이','답은 아니었다',
        'clinical utility','clinical benefit','prevented omission','excess omission',
        'prespecified matching','strong heterogeneity',
        '원 프로토콜은 최초의 두 추정기',
        '★','작업본','이전 판','앞선 초안','직전 판',
        'v4 까지','v5 는','v6 은','v6 는','v7 은','v7 까지','v8 초록','v8 에서','v9 에서',
        # ── 연구 «과정»의 서술 — 논문에 기여하지 않으면 원고에 없어야 한다
        # 1인칭 자체는 금칙이 아니다 — 「We defined/evaluated/hypothesized」는 정상 문장이다.
        # 막는 것은 구현·디버깅 일지형 표현뿐이다.
        '정정 기록','초기 구현','최초 비교','이전에 보고','정정 후','재실행했다',
        '숨기지 않는다','변호하지 않','미루지 않는다','작업 기록',
        # 분할 민감도의 과대해석 — 분할과 무관하지도, 「더 커진」 것도 아니다
        'partition-independent','partition independent','became stronger','오히려 커진다',
        # ★고치기 «전» 귀무에서 나온 값 — 되살아나면 안 된다
        '0.482','0.537','314,493','1.6~2.0배',
        '주사 급성기','경구 만성유지','예방·PRN 주사','경구·국소 재량',
        # 200→99 의 과대해석 — 포함 규칙은 실제 자료의 선정 규칙이지 귀무의 요구가 아니다
        '원 코호트라면 선정되지 않았을','포함 문턱(제공자당 ≥ 200 사건) 아래']
    for wd in 금칙:
        c=tb.count(re.sub(r'\s+',' ',wd)); ok&= (c==0); n+=1
        print(f'{"OK " if c==0 else "FAIL"}  금칙 「{wd[:34]}」 {c}건')
    필수_본문=[('확증 실행 전에 고정했다',1),
        ('사전 고정한 추정기는 저랭크 이중선형 하나',1),
        ('64차원 학습 잠재 표',1),
        ('같은 데이터베이스에 대한 탐색 분석',1),
        ('프로토콜 이탈은 전부 보충자료에 기록한다',1),
        ('matched on overall non-administration rate',1),
        ('among all eligible pairs',1),
        ('not necessarily the individuals who performed',1),
        ('Most of the concentration was already achieved',1),
        ('prospective triage',1),
        ('estimator family alone',1),('Estimator family alone',1),
        ('should not be treated as interchangeable',1),
        ('remains a methodological limitation',1),
        ('사전 프로토콜에도 두 동결 계획에도',1),
        ('frequency-weighted sample without replacement from the global provider pool',1),
        ('deviation from the original null specification',1),
        ('definition of the null-adjusted contrast',1),
        ('provider- and outcome-blind alternative partition',1),
        ('not specific to the model-derived medication partition',1),
        ('Monte Carlo standardized separation',1),
        ('no tail probability was inferred from it',1),
        ('different constructs and different eligible provider sets',1),
        ('represent this medication-specific reversal',1),
        ('Conditional on this exploratory model-derived partition',1),
        ('retain the same scientific or operational meaning',1),
        ('was not reproducible across refits',1),
        ('exploratory device for displaying',1),
        ('귀무 3회 기준',1),
        ('RQ1',1),('RQ2',1),('RQ4',1)]
    필수_전체=[('product of provider and drug',1),('weighted Pearson r',1),

        ('이 시정은 동결 계획에 근거가 없다',1),
        ('귀무 2회 기준',2),
        ('교환 상대가 없',1),('실제로» 동일한 입원',1),
        ('진단용 복제 계열',1)]
    for lab,lst,src in (('필수·본문',필수_본문,tn),('필수·본문|보충',필수_전체,tb)):
        for wd,least in lst:
            c=src.count(re.sub(r'\s+',' ',wd)); good=c>=least; ok&=good; n+=1
            print(f'{"OK " if good else "FAIL"}  {lab} 「{wd[:32]}」 {c}건 (>= {least})')
    # ★제목 검사 — 절 제목은 «무엇을 보고하는가»를 말한다.
    #   자기 부인("…이며 증거가 아니다")도, 작업 메모("실행 전에 정의함")도,
    #   앞 절에 매달리는 접속사도 제목에 오지 않는다.
    _heads=re.findall(r'^#{1,4}\s+(.+)$',t,re.M)
    _bad_in=['증거가 아니다','저자가 번호를','실행 전에 정의함','쓰지 않는 방법',
             '전문은 보충자료','작업본','예시이며']
    _bad_pre=('그러나 ','그리고 ','그래서 ','따라서 ','또한 ')
    _hit=[h for h in _heads
          if any(w in h for w in _bad_in) or h.lstrip('0123456789. ').startswith(_bad_pre)]
    n=n+1; ok&= not _hit
    print(f'{"OK " if not _hit else "FAIL"}  절 제목에 자기 부인·작업 메모·접속사 시작이 없다 {_hit}')

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
