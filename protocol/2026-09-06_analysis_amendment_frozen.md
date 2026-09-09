# 분석계획 수정안 — 실행 전 사전 고정

**작성 시각은 문서 말미 §12 에 기계 기록으로 남긴다. 이 시점 이후 어떤 계산도
수행되지 않은 상태에서 아래 11개 항목이 확정되었다.**

> **성격 (반드시 이 문구로 보고한다)**
> Post hoc diagnostic experiment motivated by a structural mismatch identified during
> internal methodological review. **This is not part of the original prespecified
> analysis.** The analysis plan for this diagnostic was fixed before executing it.

---

# 1. 가설

> **개체를 모형에 «포함하는가»로 정의되는 대비는 추정기 계열을 건너 일관되게
> 이식되지만, 모형 «내부의 분해»에 의존하는 대비는 그 분해가 구조로 명시되지
> 않는 한 이식되지 않는다.**

> We hypothesized that contrasts defined by model inclusion would transfer more
> consistently across estimator classes than contrasts that depend on an internal
> decomposition of the model.

이 가설은 서술이 아니라 **예측**이다. 본 진단 실험이 그 첫 번째 시험이다.

---

# 2. estimand 이식성 계층 — 결과표의 배열 규칙

| 층 | 양 | 이식성 |
|---|---|---|
| **1. 포함 대비** | **R = Δ_total = (C−A) 순증** | 모형군이 달라도 **같은 operational contrast**. 값은 달라도 된다. |
| **2. 분해 의존 대비** | **Δ_main, Δ_latent, Q2, Q2b** | A/B/C 가 같은 의미로 정의될 때만 교차 비교 가능. **Q2b 는 분모까지 구조적 의미가 필요해 이 층에서도 더 취약하다.** |
| **3. 구조 고유량** | **S (셔플 잠재 ≤ 0)** | 특정 저랭크 구조가 존재할 때만 정의된다. |

**Q2 에 대한 구분 (반드시 병기)** — Q2 의 숫자는 native CDLD «내부»에서는
「B→C 로 표현 용량을 늘렸을 때 예측 기술이 0.005 이상 늘었는가」라는
operational criterion 으로 여전히 유효하다. 다만 ALS 의 C−B 와 같은
「interaction component」를 재는 양으로 «교차 비교» 할 수는 없다.

**결과표 배열** — 설정 순이 아니라 **위 계층 순**으로 배열한다.
`R 7/7 · Q2 5/7 · Q2b 2/7` 은 **표에는 그대로 남기되 본문 서사에서 계단으로 쓰지 않는다.**
본문 문장은 다음으로 고정한다.

> The inclusion contrast retained the same operational meaning across all settings,
> whereas criteria based on internal model decomposition did not.

**Q2b 각주 (표에 그대로 남기고 각주를 단다)**

> Numerically computable, but not interpretable as a main/interaction decomposition
> ratio under the unconstrained readout.

---

# 3. R 의 문안

**`same operational contrast` 를 쓴다. `same estimand` 를 쓰지 않는다.**

> R preserves the same operational contrast across model families: how much
> out-of-sample predictive skill a model gains when provider information is admitted.
> The numerical value may depend on the estimator; the question does not.

---

# 4. 세 모형과 두 대비가 isolate 하는 것

| 모형 | baseline | readout |
|---|---|---|
| **ALS** (저랭크 이중선형) | 선형 (가법 demeaning) | 가법 + 이중선형 (명시적) |
| **CDLD-A** (신규) | 심층 f(x) | 가법 + 이중선형 (명시적) |
| **native CDLD** | 심층 f(x) | 비제약 비선형 공유 head |

- **native ↔ CDLD-A** — deep baseline · cyclic dual-latent 학습 · 자료 · 분할 ·
  최적화 프레임워크를 «모두 유지»하고 **readout 제약만** 바꾼다. **깨끗한 대비다.**
- **CDLD-A ↔ ALS** — baseline function class, 최적화, 정규화, 학습 동역학, 초기화가
  **동시에** 바뀐다. 여기서 차이가 나도 **원인을 baseline 유연성으로 지목하지 않는다.**
  보고 문안: **all model-family differences other than the explicit decomposition
  structure.** 4번째 모형(선형 baseline + 순환 학습)은 **만들지 않는다.**
  **한계(limitation)로 기술하고 멈춘다.**

**명명** — CDLD-A 는 CDLD 계열 «안»에 둔다. CDLD 의 정의적 기제는
cyclic dual latent(두 잠재표 · 교대 ULD/ILD 턴 · 턴마다 상대표 동기화)이며,
비선형 공유 head 는 정의가 아니라 readout 선택이다. CDLD-A 는 순환·이중잠재를
그대로 유지하고 readout 만 제약한다.

---

# 5. ★A 단계 공유

    A_native = A_CDLD-A            ← 설계로 «동일하게 만든다». 통계적 동등성 검정을 하지 않는다.

- 두 계열의 A 는 동일 구조다: `f(x)` = 근무조·약물·진단챕터 가법코드 + 공변량 스트림
  + **약물 잠재 64** (제공자 항 없음). CDLD-A 의 `η_C = f(x) + α_i + u_i'v_j` 에서
  `v_j` 는 **상호작용 전용의 별도 약물 임베딩**이고 **약물 주효과는 f(x) 안에 남는다**
  (ALS 도 동일 구조: 가법 기저에 약물 주효과, 저랭크 항에 별도 v).
- **기존 관행의 확장임을 명시한다** — 현행 코드는 이미 A 를 팔(real/shuffle) 사이에
  공유하고 있다(A 에 제공자 항이 없어 섞기에 불변). 그 원칙을 모형군 사이로 넓힌 것이다.
- 보고 문안:

  > The A-stage model was shared between native CDLD and CDLD-A so that any difference
  > in the subsequent ladder could not arise from stochastic variation in the common baseline.

- **단위시험 U1** — CDLD-A 의 f(x) 부분 아키텍처가 native A 와 파라미터 형상까지
  일치함을 확인한다.

**대수적 귀결** — A 가 공유되면

    Δ_main,nat − Δ_main,A = (B_nat − A) − (B_A − A) = B_nat − B_A = γ*

즉 **γ\* 가 두 주효과 증분의 차이와 정확히 같아진다. 잔차항이 없다.**

---

# 6. native B 함수 진단 (재학습 필요 — 가중치가 저장돼 있지 않다)

## 6.1 비가법성 대비 D

고정된 event context `x` 에서 제공자·약물만 counterfactual 하게 교체한다.

    D(i, i', j, j' ; x) = [η_B(i,j,x) − η_B(i',j,x)] − [η_B(i,j',x) − η_B(i',j',x)]

**x 를 고정한다.** 실제 관측 사건 둘을 비교하면 약물이 바뀌며 맥락도 바뀌어
D≠0 을 provider×drug 비가법성이라 말할 수 없다.

## 6.2 두 벌 — support 제한

- **D_supported** — 네 조합 `(i,j), (i',j), (i,j'), (i',j')` 가 **모두** 학습셋에서
  **≥ N 사건**으로 관측된 경우만. x 는 그 넷 중 실제 event context 하나로 고정.
  **주 분석 N = 20**, 민감도 **N ∈ {5, 50}**.
- **D_all** — 제한 없이 무작위 조합.

읽기: `D_supported` 큼 → 학습 support 안에서도 B 가 비가법적.
`D_all` 만 큼 → off-support 외삽 가능성. 이 경우 **D_supported 만 보고**한다.

## 6.3 선택 편향 통제

support 문턱은 **사건 수가 많은 제공자를 고른다** — 곧 잠재가 가장 잘 추정된 제공자다.
따라서 **제공자 사건수 삼분위로 층화 보고**하고, 선택된 집합과 전체의 제공자
사건수 분포를 함께 싣는다.

## 6.4 눈금 — 모형 «안»에서 찾는다

γ 축(BSS)으로 환산하지 **않는다** (outcome 분포·기준 예측이 끼어들어 단순 변환이 없다).
대신 같은 함수·같은 척도·같은 x 에서:

    주대비  M(i,i';x) = E_j [ η_B(i,j,x) − η_B(i',j,x) ]
    무차원 눈금 :  sd(D) / sd(M)

「제공자 효과가 약물에 따라, 제공자 효과 자체의 평균 크기 대비 몇 % 변하는가」.

## 6.5 ★격자 이원 분산분해 — native CDLD 에 «정당한» 분해를 하나 돌려준다

고정 x 에서 η_B 를 (제공자 × 약물) 격자 위 함수로 보면 이원 분해가 잘 정의된다.

    η_B(i,j;x) = μ + a_i + b_j + c_ij
    상호작용 분율 = Var(c) / [Var(a) + Var(b) + Var(c)]

**예측 기술의 분해와 달리 «함수» 의 분해는 정의된다.** 논문의 주장은
「예측 기술 증분을 주효과/상호작용으로 쪼갤 구조가 없다」이지 「함수를 쪼갤 수 없다」가
아니다. 보고 문안:

> Q2b is not defined as a decomposition of predictive skill under the unconstrained
> readout. It is, however, well defined as a decomposition of the fitted function on
> the provider × drug grid, and that value is reported here.

**단서 3개를 본문에 명시한다** — (a) logit 척도의 «적합된 함수» 속성이며 모집단
estimand 가 아니다, (b) 격자 위 가중(동시출현 빈도)에 의존하므로 가중을 명시한다,
(c) x 에 의존하므로 x 를 여러 개 뽑아 분포로 보고한다.

## 6.6 D 와 γ\* 의 역할 분리 — 크기 일치를 «기대하지 않는다»

native B 의 head 는 provider×drug 만이 아니라 provider×age, provider×ICU,
provider×(임의 공변량), 고차 조합까지 만든다. **γ\* 는 그 전부를 담고 D 는 한 축만
겨냥한다. 두 양은 같지 않다.**

> `D_supported` 는 native B 의 provider–drug 비가법성이 **실제 존재함**을 보여주고,
> γ\* 는 additive readout 을 강제했을 때 **사라지는 predictive skill** 을 보여준다.
> 둘의 **동시 관찰**이 unconstrained B-stage readout 이라는 기전을 지지한다.

γ 의 이름: **non-additive provider-dependent predictive contribution**.
「흡수한 provider×drug interaction」이라고 부르지 않는다.

**★사전에 정한 읽기** — **D 가 작은데 γ\* 가 크면**, native B 가 흡수한 것은
provider×약물이 아니라 **provider×공변량**이다. 그 경우 「잠재가 재는 것이 약물
특이성이다」라는 실질 주장 자체가 흔들리므로, **그 사실을 그대로 보고한다.**

## 6.7 단위시험

- **U2** — CDLD-A 의 B 는 `|D| < 1e-5` 를 **반드시 통과**해야 한다 (구현 정확성).
- **U3** — CDLD-A 의 C 에서 `D` 가 `u_i'v_j` 형태의 이중선형과 일치해야 한다.
- native 의 B 는 **pass/fail 로 두지 않고 같은 통계량을 «측정»만** 한다
  (우연히 거의 가법일 수 있다).

---

# 7. Fit gate — 모형이 «망가졌는지» 만 본다

    관문 : CDLD-A 의 C 가 native 의 C 보다 δ 이상 열등하지 않을 것 (단측)
    δ    = 0.0025

**δ 의 정박점** — 주 설정 4개(CDLD main 0.00573 · CDLD ward 0.00504 ·
ALS main 0.01134 · ALS ward 0.00770)의 잠재 증분 중 **최소값 0.00504 의 절반**.
특정 모형의 특정 해석에 기대지 않는다.

**비순환 명시 문장 (본문에 넣는다)**

> The margin is anchored to a predictive-skill increment, which is well defined in
> every model considered; the paper's claim of non-portability concerns the
> *decomposition interpretation* of that increment, not its numerical magnitude.

**성격 문안** — `prespecified non-inferiority margin` 이라고 쓰지 않는다.
**`a diagnostic margin fixed before executing the post-hoc experiment`** 로 쓴다.

**검정 — 환자 수준 짝지은 부트스트랩**

1. 재표본 단위 = **subject_id**. 폴드 분할도 이미 subject_id 기준이며, 5폴드 OOF 를
   이어 붙이면 각 환자가 정확히 한 번 나온다.
2. **두 모형에 같은 재표본**을 쓴다. 재표본 b 마다
   `ΔBSS_b = [SSE_native(b) − SSE_A(b)] / SStot(b)` — 분모가 공통이라 상쇄된다.
3. 기준 예측기는 **행별 폴드의 학습 유병률**로 고정.
   `SStot(b) = Σ_b (y − ȳ_train(fold(행)))²`. 재표본에서 재추정하지 않는다.
4. **2,000 재표본.** 관문: `ΔBSS` 의 95% CI **하한 > −δ**.

**해석의 한계 — 반드시 문안에 넣는다**

> conditional on the fitted models

이 CI 는 training procedure 의 불확실성이 아니라 evaluation population sampling
uncertainty 다. 「CDLD-A 라는 모형군이 native 에 비열등하다」고 말하지 않는다.
native C 의 폴드 간 SD **0.0041** 을 병기하되, **추론 구간이 아니라**
`variability across independently fitted fold models` 로만 기술한다.

**B 에는 관문을 걸지 않는다.** CDLD-A 의 B 가 native B 보다 나쁜 것이 제약의
정의이자 가설이다. 관문에 넣으면 가설을 실패 조건으로 만드는 셈이다.
**A 에도 관문을 걸지 않는다 — §5 로 공유하여 문제를 없앴다.**

---

# 8. Signal preservation diagnostic

    κ = Δ_total,CDLD-A − Δ_total,native            (둘 다 null 보정)

**Fit gate 와 다른 질문이다.** Fit gate 는 「모형이 단순히 망가졌나」,
κ 는 「provider inclusion signal 자체가 얼마나 이동했나」를 본다. **별도로 보고한다.**

    Δ_total,native = 0.01535 ,  R 기준 = +0.010
    → κ < −0.00535 이면 CDLD-A 에서 «R 이 실패»한다

---

# 9. ★귀속 분해 규칙 (점 예측이 아니다)

## 9.1 항등식

    γ* = (B_nat − B_A) − (B_nat,null − B_A,null)        null 보정 B 단계 격차
    κ  = Δ_total,A − Δ_total,nat

    A 공유 전제에서
      Δ_main,A   = m − γ*
      Δ_latent,A = l + γ* + κ
      r_A        = (l + γ* + κ) / (m − γ*)

      m = Δ_main,nat = 0.00963 ,  l = Δ_latent,nat = 0.00573 ,  r_nat = 0.5947

**이것은 가설이 아니라 정의에서 따라 나오는 항등식이다.** 그러므로 예측이 아니라
**귀속**에 쓴다.

## 9.2 사전 고정하는 것 — 예측값이 아니라 «분해 규칙»

    관측 이동  Δr = r_A − r_nat

    γ*-만의 반사실 :  r(γ*, 0) = (l + γ*) / (m − γ*)
    κ-만의 반사실  :  r(0, κ)  = (l + κ) / m

    readout 제약이 설명하는 몫 = r(γ*,0) − r_nat
    총신호 이동이 설명하는 몫  = r(0,κ)  − r_nat
    잔차(교호)                = Δr − (위 둘의 합)

| 관측 | 사전 고정된 읽기 |
|---|---|
| γ\* 몫이 Δr 의 대부분 | **Branch B** — unconstrained readout 이 뒤집힘의 주원인 |
| κ 몫이 대부분 | 뒤집힘이 아니라 **총 제공자 신호 자체가 이동** — §10 Branch D 축으로 |
| 둘 다 작고 Δr 도 작다 | **Branch A** — readout 을 맞춰도 갈린다 |
| 잔차가 지배 | 두 항의 비선형 교호 — 일반식으로만 기술하고 귀속하지 않는다 |

## 9.3 조건부 참조값 (표에 남기되 «조건부» 로 명시)

κ = 0 을 가정할 때:

| 목표 | 필요한 γ\* |
|---|---|
| r_A = 1.00 (Q2b 판정 뒤집힘) | **γ\* = 0.00195** |
| r_A = 1.475 (ALS ward) | γ\* = 0.00342 |
| r_A = 1.509 (ALS main) | **γ\* = 0.00351** |

> Conditional on preservation of the total null-adjusted provider inclusion effect
> (κ ≈ 0), these γ\* values would produce the corresponding ratios.

---

# 10. 해석 분기 — 2×2

|  | **R 유지** (κ > −0.00535) | **R 실패** (κ < −0.00535) |
|---|---|---|
| **비가 ALS 로 이동** | **Branch B** | **Branch D** |
| **비가 여전히 갈림** | **Branch A** | **Branch D** |
| **fit gate 실패** | **Branch C** | **Branch C** |

**Branch A** — CDLD-A 가 fit gate 를 통과하고 κ≈0 이면서도 ALS 와 다른 분해를 보인다.
→ readout alignment 만으로 차이가 설명되지 않는다. **decomposition 의
estimator-class sensitivity 가 남는다.**

**Branch B** — 다음 넷이 «동시에» 성립할 때에 한해 선언한다.
(1) A 공유, (2) C 가 fit gate 통과, (3) κ ≈ 0, (4) 관측 γ\* 를 넣은 §9.1 식이
실제 비 이동을 설명. → *The decomposition flip was largely attributable to the
unconstrained B-stage readout.* `D_supported` 가 native B 의 비가법성을 별도로
보여주면 기전 증거가 하나 더 붙는다.

**Branch C** — fit gate 실패. **「논문 없음」이 아니라 가장 약한 버전의 같은 논문.**
착지점: (1) §2 의 수정된 계층, (2) native B 의 비가법성 «측정치»와 §6.5 의 함수 분해,
(3) R 의 7/7 강건성, (4) 왜 정합 비교가 이 코호트에서 실행 불가능했는지의 기록.
결론 문장: *the comparison cannot be adjudicated at fixed predictive capacity in
this cohort.* **미해결을 미해결로 보고한다.**

**Branch D** — 분해를 구조로 강제하자 **총 포함 효과 자체가 무너졌다** (κ < −0.00535).
읽기: *inclusion contrast 의 이식성조차 readout 자유도에 의존한다.*
착지점: 계층 1층이 약해지므로 **portability 를 주장하지 않고
「어떤 대비도 무비판적으로 모형군을 건너지 않는다」는 더 보수적인 결론**으로 간다.
투고처를 임상정보학에서 **방법론 쪽으로 옮긴다.**

**Branch D 를 지금 적어두는 이유** — 그것이 나왔을 때 **보고할 것인지를 결과를
보기 전에 정해두기 위해서다.** 이 논문의 유일한 진짜 자산이 「실패한 사전 기준도
그대로 보고했다」이므로, 여기서 빈칸을 남기면 자산을 잃는다.

---

# 11. 실행 범위와 계산 계획

    주 실행 : base 설정, real + null ×2, 환자 단위 5겹
              A 는 공유(계산 0) · B, C 만 학습
    실측 기반 추정 : real B+C 0.95h + null 팔당 0.84h × 2 = 약 2.6h

- **ward 는 base 결과를 본 뒤 결정**한다. 뒤집힘의 원판이 base 이다.
- **null 2회로 한다.** 기존 실험에서 null 반복 간 일치가 **0.00004** 였으므로
  3회째는 안정성에 거의 기여하지 않는다. 프로토콜 동수를 맞추려면 base 결과 후 추가.
- **native B 재학습이 선행된다** — 가중치(state_dict)가 저장돼 있지 않아 §6 진단이
  불가능하다. base real 5폴드 A·B 를 재학습하며 이번에는 체크포인트를 저장한다.
  **재학습된 B 의 지표가 기존 기록과 일치하는지 먼저 확인**하고(동일 seed·동일 일정),
  일치하지 않으면 진단을 진행하지 않고 원인을 먼저 규명한다.

## 프로토콜 이탈 기록 (원판 대비)

| 항목 | 원 프로토콜 | 실제 | 사유 |
|---|---|---|---|
| null 반복 | ×2 | ×3 (CDLD 본실험) | Monte Carlo 안정성 |
| 잠재 차원 16 | 없음 | 실행함 | 추가 exploratory sensitivity |
| provider ≥500 | 계획됨 | **미실행** | — |
| Sliding Scale 민감도 | 계획됨 | **미실행 → 이번에 완료 예정** | — |
| Hold Dose 결과 민감도 | 없음 | 이번에 추가 | 검토 지적 |
| **CDLD-A 진단 실험** | 없음 | 이번에 추가 | **post hoc diagnostic** |

**「프로토콜 이탈은 없었다」는 문장을 원고에서 삭제하고 위 표로 대체한다.**

## 지표 명칭

`R²` 를 **`Brier skill score (BSS; reference = training-fold prevalence)`** 로 개명한다.
근거: 코드의 `sstot = Σ(y_te − ȳ_tr)²` 는 학습폴드 유병률 상수 예측기를 표본외에서
평가한 것이므로 `1 − BS_model/BS_climatology` 와 정확히 일치한다.

---

# 12. 동결 기록

    작성 완료 시각 : 2026-09-06T13:01:24+00:00
    v4 동결 스냅숏 : out/20260906_v4_snapshot/  (파일 1046개, MANIFEST.md 에 sha256)
    이 시점까지 CDLD-A 관련 계산은 «한 건도» 수행되지 않았다.
    native B 재학습, D 진단, CDLD-A 구현·실행은 «모두 이 문서 이후»에 시작된다.

본 문서의 sha256 (본문 확정분, 이 블록 제외) :
    d6f930125165527f4967871646bf5348026228ed23800b1cf20e5acc1419e839
