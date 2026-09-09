import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
"""emar 사건에 careunit 을 붙인다.

transfers 의 [intime, outtime) 구간에 emar.charttime 이 들어가는 것을 찾는다.
discharge 행(careunit=UNKNOWN, outtime 없음)은 길이 0 표지이므로 버린다.
"""
import pandas as pd, numpy as np

U = paths.DATA + os.sep
OUT = paths.DATA + os.sep

for _f in ('transfers.csv.gz', 'emar_slim.csv.gz'):
    if not os.path.exists(U + _f):
        paths.need_data(f'{_f} under {U}')
tr = pd.read_csv(U+'transfers.csv.gz',
                 usecols=['subject_id','hadm_id','eventtype','careunit','intime','outtime'])
print('transfers', len(tr))
tr = tr[tr.eventtype != 'discharge']
tr = tr[tr.hadm_id.notna() & tr.careunit.notna() & tr.outtime.notna()]
tr['hadm_id'] = tr.hadm_id.astype('int64')
tr['intime']  = pd.to_datetime(tr.intime)
tr['outtime'] = pd.to_datetime(tr.outtime)
print('사용 구간', len(tr), '| 입원', tr.hadm_id.nunique(), '| careunit', tr.careunit.nunique())

em = pd.read_csv(U+'emar_slim.csv.gz', usecols=['hadm_id','emar_id','charttime'])
print('emar(제공자 채움)', len(em))
em = em[em.hadm_id.notna() & em.charttime.notna()].copy()
em['hadm_id'] = em.hadm_id.astype('int64')
em['charttime'] = pd.to_datetime(em.charttime)
print('시간·입원 있는 행', len(em))

tr = tr.sort_values(['intime']).reset_index(drop=True)
em = em.sort_values(['charttime']).reset_index(drop=True)

m = pd.merge_asof(em, tr[['hadm_id','careunit','intime','outtime']],
                  left_on='charttime', right_on='intime',
                  by='hadm_id', direction='backward')
hit = m.outtime.notna() & (m.charttime < m.outtime)
m.loc[~hit, 'careunit'] = np.nan

print('\n=== 결과 ===')
print('배정됨 %d / %d = %.1f%%' % (hit.sum(), len(m), 100*hit.mean()))
out = m.loc[hit, ['emar_id','careunit']]
out.to_csv(OUT+'emar_careunit.csv.gz', index=False, compression='gzip')
print('\n상위 careunit:')
print(out.careunit.value_counts().head(20).to_string())
print('\ncareunit 수:', out.careunit.nunique())
