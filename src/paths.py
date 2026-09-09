"""저장소 자립 경로 해석.

하드코딩된 절대경로를 대체한다. 우선순위는 환경변수 -> 저장소 상대 기본값.
저장소 어디에도 개발 기계의 절대경로가 남아 있어서는 안 된다.

    CDLD_ROOT   저장소 루트            기본: 이 파일의 상위 디렉터리
    CDLD_WORK   작업/체크포인트 출력    기본: <ROOT>/work
    CDLD_DATA   MIMIC 파생 캐시 위치   기본: <WORK>/cache
    EMAR_SLIM   슬림 eMAR csv.gz       기본: <DATA>/emar_slim.csv.gz
    MIMIC_HOSP  MIMIC-IV hosp 모듈     기본: <DATA>/hosp

원자료가 없어도 «결과 검증»(verify.py · verify_v5/v6/v8.py)과 «그림 생성»(figs_v6/v8.py)은
전부 돈다. 그것들은 committed 결과 파일만 읽는다.
"""
import os

ROOT = os.environ.get('CDLD_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get('CDLD_WORK') or os.path.join(ROOT, 'work')
DATA = os.environ.get('CDLD_DATA') or os.path.join(WORK, 'cache')
EMAR = os.environ.get('EMAR_SLIM') or os.path.join(DATA, 'emar_slim.csv.gz')
HOSP = os.environ.get('MIMIC_HOSP') or os.path.join(DATA, 'hosp')
RESULTS = os.path.join(ROOT, 'results')


def work(*parts):
    p = os.path.join(WORK, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def need_data(what='MIMIC-derived cache'):
    """원자료가 필요한 지점에서 명확히 안내하고 멈춘다."""
    raise SystemExit(
        f'[cdld] {what} not found.\n'
        f'  Set CDLD_DATA (currently: {DATA}) to a directory containing the\n'
        f'  MIMIC-IV derived inputs, or EMAR_SLIM / MIMIC_HOSP individually.\n'
        f'  MIMIC-IV requires PhysioNet credentialing; raw and derived data are\n'
        f'  not redistributed with this repository.\n'
        f'  Result verification does NOT need this: python src/verify_v8.py')
