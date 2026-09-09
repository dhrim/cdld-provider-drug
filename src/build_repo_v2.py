import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
"""교정 귀무 기준으로 공개 저장소 결과 파일을 만든다."""
import json, glob, os, numpy as np, pandas as pd
CK=os.path.join(paths.WORK, 'cdld/ck/'); R=os.path.join(paths.WORK, 'repo_v2/results/')
os.makedirs(R, exist_ok=True)

# ── 겹 수준 원자료 ────────────────────────────────────────────────
rows=[json.load(open(f)) for f in sorted(glob.glob(CK+'*.json'))]
d=pd.DataFrame(rows); d['r2']=1-d.sse/d.sstot
d=d[['tag','arm','fold','kind','n_te','r2','auc','sse','sstot','secs']]
d.sort_values(['tag','kind','arm','fold']).to_csv(R+'fold_level.csv',index=False)
print('fold_level', len(d))

# ── 표 1 ─────────────────────────────────────────────────────────
pd.DataFrame([
 ('Medication administration events','4,362,548'),
 ('Recording providers','936'),('Distinct medications','522'),
 ('Observed provider x drug cells','106,580 / 488,592 (21.81%)'),
 ('Non-administration rate','9.29%'),
 ('Providers in the emar table','1,621'),
 ('  overlap with prescribing physicians','18.7%'),
 ('  overlap with admitting physicians','2.3%'),
 ('  in neither group','81.3%'),
 ('Events per provider (median)','748'),
 ('Patients per provider (median)','107'),
 ('Distinct drugs per provider (median)','89'),
 ('Night-shift share (median)','0.20'),
 ('Care unit assigned','98.8% (42 units)'),
 ('Non-administration rate by care unit','4.5% - 11.2%'),
 ('Providers effectively in one care unit','45%'),
], columns=['item','value']).to_csv(R+'table1_cohort.csv',index=False)

# ── 귀무 설계 표 ──────────────────────────────────────────────────
pd.DataFrame([
 dict(design='observed data', providers_per_admission_mean=3.02,
      median=2, max=635, provider_events_median=2654, provider_events_max=46637,
      share_unchanged=1.000),
 dict(design='null v1 (broken: fancy-index collapse)', providers_per_admission_mean=1.00,
      median=1, max=1, provider_events_median=2505, provider_events_max=182967,
      share_unchanged=np.nan),
 dict(design='null v2 (corrected, used here)', providers_per_admission_mean=3.02,
      median=2, max=635, provider_events_median=2754, provider_events_max=42465,
      share_unchanged=0.0026),
]).to_csv(R+'table_null_design.csv',index=False)

# ── 표 2 · 2b · 3 ────────────────────────────────────────────────
t=pd.read_csv(os.path.join(paths.WORK, 'cdld/out_v2_전체집계.csv'))
ren={'설정':'setting','겹':'folds','귀무':'nulls','주효과':'main_effect','주효과sd':'main_sd',
     '잠재':'latent','잠재sd':'latent_sd','총':'total','총sd':'total_sd','비':'ratio',
     'S':'shuffled_latent','AUC_C':'auc_C'}
t=t.rename(columns=ren)
name={'CDLD 주판':'Cyclic dual latent — main',
      'CDLD + 병동·병동×약물':'Cyclic dual latent — ward & ward x drug',
      'CDLD 시간 분할':'Cyclic dual latent — temporal split',
      'CDLD 잠재 64→16':'Cyclic dual latent — latent 64 to 16',
      'CDLD 약물 상위 100종':'Cyclic dual latent — top-100 drugs',
      'ALS 주판':'Low-rank bilinear — main',
      'ALS 병동 통제':'Low-rank bilinear — ward & ward x drug'}
t['setting']=t.setting.map(lambda s: name.get(s,s))
t.to_csv(R+'table3_settings.csv',index=False,float_format='%.5f')
two=t[t.setting.str.endswith('main')][['setting','A','B','C','auc_C','main_effect','latent','total','ratio','shuffled_latent']]
two.to_csv(R+'table2_two_estimators.csv',index=False,float_format='%.4f')
m=t[t.setting=='Cyclic dual latent — main'].iloc[0]
a=t[t.setting=='Low-rank bilinear — main'].iloc[0]
pd.DataFrame([
 dict(criterion='S',  definition='shuffled-arm latent net gain <= 0', threshold='<= 0',
      cyclic_dual_latent=round(m.shuffled_latent,4), verdict_CDLD='met',
      low_rank=round(a.shuffled_latent,4), verdict_ALS='met'),
 dict(criterion='U0', definition='severity proxies in the baseline', threshold='included',
      cyclic_dual_latent='included', verdict_CDLD='met', low_rank='included', verdict_ALS='met'),
 dict(criterion='R',  definition='entity net value (C-A)', threshold='>= +0.010',
      cyclic_dual_latent=round(m.total,4), verdict_CDLD='met',
      low_rank=round(a.total,4), verdict_ALS='met'),
 dict(criterion='Q2', definition='latent net gain (C-B)', threshold='>= +0.005',
      cyclic_dual_latent=round(m.latent,4), verdict_CDLD='met',
      low_rank=round(a.latent,4), verdict_ALS='met'),
 dict(criterion='Q2b',definition='latent / main-effect net gain', threshold='>= 1.0',
      cyclic_dual_latent=round(m.ratio,3), verdict_CDLD='NOT MET',
      low_rank=round(a.ratio,3), verdict_ALS='met'),
]).to_csv(R+'table2b_adjudication.csv',index=False)
print(two.to_string(index=False))
