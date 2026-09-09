"""MIMIC-IV `emar.csv.gz` 에서 이 연구가 쓰는 열만 남긴 슬림 파일을 만든다.

    출력 열 : subject_id · hadm_id · emar_id · poe_id · pharmacy_id ·
              enter_provider_id · charttime · medication · event_txt

MIMIC-IV 는 PhysioNet 자격 인증이 필요하며, 원자료와 파생 파일은 재배포하지 않는다.

    python src/build_emar.py  <MIMIC-IV hosp/emar.csv.gz>  [출력경로]
"""
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

COLS = ['subject_id', 'hadm_id', 'emar_id', 'poe_id', 'pharmacy_id',
        'enter_provider_id', 'charttime', 'medication', 'event_txt']


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else paths.EMAR
    if not os.path.exists(src):
        paths.need_data(f'input file {src}')
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    df = pd.read_csv(src, usecols=COLS)
    df.to_csv(dst, index=False, compression='gzip')
    print(f'{len(df):,} rows -> {dst}')


if __name__ == '__main__':
    main()
