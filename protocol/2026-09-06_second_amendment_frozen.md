# 보완 실험 계획 — 실행 전 사전 고정 (2차)

> **1차 때의 실수를 되풀이하지 않기 위해 쓴다.** 1차 동결 계획에는 「fit gate 실패 시
> 무엇을 조정할 수 있는가」가 한 줄도 없었고, 그럼에도 원고에는 「실행 전에 기록해 둔
> 시정 방침에 따라」라고 썼다. 이번에는 **조정 가능한 것과 판정 규칙을 먼저 적는다.**

성격 문안 —

> Second post-hoc diagnostic experiment, prompted by external review of v5.
> Not part of the original prespecified analysis, nor of the first (2026-09-06 13:01)
> amendment. The plan for this experiment was fixed before executing it.

---

# 1. 무엇을 묻는가

v5 의 중심 주장은 「뒤집힘은 읽기층 제약 때문」이다. 그런데 CDLD-A2 의 성공은
**1차 실행 실패를 본 뒤 잠재표를 감쇠에서 제외하고** 얻었다. 따라서 현재 자료로는

    (가) 명시적 분해 구조 때문인가
    (나) 상호작용 잠재에 훨씬 유리해진 정규화 때문인가

가 분리되지 않는다. **읽기층과 감쇠를 교차시켜 분리한다.**

---

# 2. ★2×2 — 빈 칸 하나를 채운다

|  | 잠재표에 L2 = 1e-4 (기존) | 잠재표 L2 = 0 |
|---|---|---|
| **비제약 readout (native)** | 비 **0.595** (기존) | **★실험 A — 미실행** |
| **정합 readout (CDLD-A)** | 비 **0.221** (1차, 적합 실패) | 비 **2.343** (2차) |

**실험 A** — native C 를, 구조·일정·시드·자료·분할을 그대로 두고
`u_tab`·`v_tab` 만 감쇠에서 제외해 다시 적합한다.
(native 의 잠재표 이름은 `u_tab`, `v_tab` 이다. CDLD-A 의 `u_tab`, `v2_tab` 에 대응)

**실험 B (3점화)** — CDLD-A C 를 잠재표 L2 = **1e-5** 로 적합한다.
「fit gate 를 통과하는 범위에서 방향이 유지되는가」에 답한다.

## 2.1 실행 «전에» 고정하는 판정 규칙

`r_nat(L2=0)` 를 실험 A 의 비, `r_A(1e-5)` 를 실험 B 의 비라 한다.

| 관측 | 실행 전에 정한 결론 | 원고에 쓸 문장 |
|---|---|---|
| **R1** `r_nat(L2=0)` 가 **0.8 미만**이고 `r_A(1e-5) ≥ 1` | 움직인 것은 **읽기층**이다 | The shift was associated with the decomposition-aligned parameterization, and was not reproduced by removing weight decay under the unconstrained readout. |
| **R2** `r_nat(L2=0)` 가 **1.0 이상** | 움직인 것은 **감쇠**다. **중심 주장을 바꾼다** | The apparent effect of readout alignment was reproduced by changing the effective regularisation alone; the decomposition estimate depends on effective regularisation as much as on readout structure. |
| **R3** `r_nat(L2=0)` 가 **0.8 ~ 1.0** | **두 요인이 함께** 작용한다 | Both readout structure and effective regularisation contributed; neither alone accounts for the shift. |
| **R4** 실험 A 가 **fit gate 실패** (C 의 BSS 가 native 기존판보다 δ=0.0025 이상 열등) | 그 칸은 **비교 불가**로 보고. 2×2 를 미완으로 남기고 R1 을 주장하지 않는다 | The corresponding cell could not be evaluated at matched predictive capacity. |

**문턱 0.8 / 1.0 의 근거** — 1.0 은 Q2b 의 사전 기준 그대로다.
0.8 은 native 기존판(0.595)과 1.0 사이의 중간보다 보수적인 지점으로,
「거의 안 움직였다」와 「눈에 띄게 움직였다」를 가르기 위해 실행 전에 정한다.
**결과를 본 뒤 조정하지 않는다.**

## 2.2 이번에는 «조정 가능한 것»을 미리 적는다

실험 A·B 에서 결과를 본 뒤 바꿀 수 있는 것은 **없다.**
구조·에폭·학습률·배치·시드·자료·분할·귀무 팔 전부 고정이며,
감쇠 값만이 이 실험의 «처치» 다.
적합 관문에 걸리면 **그 칸을 비교 불가로 보고**하고(R4) 다른 조정을 하지 않는다.

---

# 3. 함께 닫는 것

## 3.1 실험 C — CDLD-A 3번째 귀무

모든 headline 수치를 **귀무 3회 기준으로 통일**한다.
현재 원고는 초록(3-null) · §4.2·§4.4(2-null) · 민감도 표(3-null)가 섞여 있다.
`shuffle2` 의 B·C 10단위를 추가하면 「귀무2 정합비교」라는 내부 사정이 본문에서 사라진다.

## 3.2 실험 D — 시드 반복

native B 의 본 실행은 모형 «생성 전»에 torch 시드를 고정하지 않았다
(`fit()` 이 `manual_seed` 를 생성 뒤에 건다). aligned B 는 생성 전에 고정했다.
따라서 γ\* 에 초기화 차이가 섞일 여지가 있다.

**3겹(0·1·2) × 추가 2시드 × 두 모형 = 12단위.**
보고: 시드별 γ\* 와 그 범위. **판정 기준을 두지 않는다 — 기술 통계로만 보고한다.**

    ★이미 손에 있는 더 강한 증거를 함께 싣는다
      B 격차가 실팔 5/5 전부 양수(평균 +0.00402) · 귀무팔 10/10 전부 음수(평균 -0.00109)
      재학습 잡음 sd 0.00074
      초기화 잡음은 «자기가 어느 팔에 있는지 모른다»

## 3.3 함수 분해 보강 (재학습 없음)

- **사전 고정 지표를 primary 로 복구** — `frac_inter = Var(c)/[Va+Vb+Vc]`, `sd(D)/sd(M)`
  (실측: native 0.0422 · CDLD-A 0.0232 — **사전 지표로도 순서와 역전이 같다**)
- `Var(c)/Var(a)` 는 **사후 보조**로 강등하고, 분모의 `Var(b)` 가 두 모형에서
  1.087 대 1.341 로 23% 달라 관심 밖 양에 좌우된다는 사유를 명시
- **x 20개의 분포를 저장·보고** — median · IQR · min/max (현재는 평균만 저장)
- **★`corr(c_native, c_aligned)`** — 상호작용 성분 자체의 상관.
  raw η 격자의 상관은 약물 주효과가 분산의 90% 이상이라 거의 1 이 나오므로 쓰지 않는다.
  판정 기준을 두지 않고, 높으면 「같은 상호작용을 찾았고 귀속만 다르다」,
  낮으면 「크기만 같고 서로 다른 상호작용」으로 **양쪽 다 보고**한다.

## 3.4 귀무 기술 정정 (계산 없음, 표만)

| | 실제 | 귀무 0 | 귀무 1 | 귀무 2 |
|---|---|---|---|---|
| 평균 사건수 | 4,661 | 4,661 | 4,661 | 4,661 |
| 중앙값 | 2,654 | 2,754 | 2,724 | 2,691 |
| 최대 | 46,637 | 42,465 | 42,833 | 41,481 |
| **최소** | **200** | **99** | **94** | **108** |
| 순위 상관 | — | 0.991 | 0.990 | 0.991 |

「보존한다」 → **`approximately preserves the marginal provider-frequency distribution`**.
그리고 **최소가 코호트 포함 문턱(제공자당 ≥200 사건) 아래로 내려간다**는 사실과
그 제공자들의 사건 비중을 각주로 보고한다.
대안 귀무(블록 교환)의 **성능 결과도 한 판** 낸다.

---

# 4. 실행 순서와 규모

    실험 A  native C, 잠재표 감쇠 제외        real + 귀무2 × 5겹 = 15단위   약 1.6h
    실험 B  CDLD-A C, 잠재표 L2 1e-5          real + 귀무2 × 5겹 = 15단위   약 1.3h
    실험 C  CDLD-A 3번째 귀무                 B·C × 5겹        = 10단위   약 0.9h
    실험 D  시드 반복                          3겹 × 2시드 × 2모형 = 12단위 약 1.1h
                                                        합계  52단위   약 5시간

    실험 A 를 «먼저» 돌린다. 그 결과 전에는 원고의 중심 문장을 확정하지 않는다.

---

# 5. 동결 기록

    작성 완료 시각 : 2026-09-06T23:29:22+00:00
    이 시점까지 실험 A·B·C·D 관련 계산은 «한 건도» 수행되지 않았다.
    본문 확정분 sha256 (이 블록 제외) :
    a12f1ef3e6b1222272dec691c272e71e277e386a04d9c64c7628eabc038d548b
