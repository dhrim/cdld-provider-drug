"""ICD 코드 → 진단 챕터. `interaction_measure.py` 에서 필요한 부분만 옮겼다.
이 폴더를 독립 실행하기 위한 것이다.
"""
import numpy as np

ICD10_CHAPTER_LETTERS = {
    'A': '감염', 'B': '감염', 'C': '신생물', 'E': '내분비', 'F': '정신', 'G': '신경감각',
    'I': '순환기', 'J': '호흡기', 'K': '소화기', 'L': '피부', 'M': '근골격',
    'N': '비뇨생식', 'O': '임신출산', 'P': '주산기', 'Q': '선천기형', 'R': '증상',
    'S': '손상중독', 'T': '손상중독', 'Z': '보건서비스'}


# ------------------------------------------------------------------ 코호트

ICD9_CHAPTER_RANGES = [
    (1, 139, '감염'), (140, 239, '신생물'), (240, 279, '내분비'), (280, 289, '혈액'),
    (290, 319, '정신'), (320, 389, '신경감각'), (390, 459, '순환기'), (460, 519, '호흡기'),
    (520, 579, '소화기'), (580, 629, '비뇨생식'), (630, 679, '임신출산'), (680, 709, '피부'),
    (710, 739, '근골격'), (740, 759, '선천기형'), (760, 779, '주산기'), (780, 799, '증상'),
    (800, 999, '손상중독')]

AGE_BINS = [-np.inf, 45, 65, 75, np.inf]

AGE_LABELS = ['<45', '45-64', '65-74', '>=75']

URGENT_ADMISSION_TYPES = {'EW EMER.', 'URGENT', 'DIRECT EMER.'}


def icd_chapter(code, version):
    """ICD-9 / ICD-10 진단코드를 공통 장(chapter)으로 매핑한다."""
    if not isinstance(code, str) or not code:
        return '기타'
    code = code.strip().upper()
    if version == 9:
        if code.startswith('V'):
            return '보건서비스'
        if code.startswith('E'):
            return '손상중독'
        try:
            head = int(code[:3])
        except ValueError:
            return '기타'
        for low, high, name in ICD9_CHAPTER_RANGES:
            if low <= head <= high:
                return name
        return '기타'
    letter, rest = code[0], code[1:3]
    if letter == 'D':                      # D00-D48 신생물, D50-D89 혈액
        try:
            return '신생물' if int(rest) <= 48 else '혈액'
        except ValueError:
            return '기타'
    if letter == 'H':                      # 눈·귀를 신경감각으로 묶는다
        return '신경감각'
    if letter in 'VWXY':
        return '손상중독'
    return ICD10_CHAPTER_LETTERS.get(letter, '기타')
