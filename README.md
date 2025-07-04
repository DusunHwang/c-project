# 희소 데이터에서의 GPR 모델 비교 연구 -1

300차원, 3000개 샘플의 희소 데이터에서 다양한 Gaussian Process Regression (GPR) 모델들의 성능을 비교하고, MSE와 모델의 불확실성(σ) 값 간 상관관계를 분석하는 연구입니다.

## 🎯 연구 목표

- 희소 데이터 환경에서 다양한 GPR 모델 성능 비교
- MSE-σ 상관관계 분석을 통한 불확실성 정량화 품질 평가  
- 최신 GPR 기법들의 희소 데이터 처리 성능 비교
- 실용적인 모델 선택 가이드라인 제공

## 📊 데이터 특성

- **차원**: 300차원
- **샘플 수**: 3000개
- **희소성**: 행당 평균 10개 non-zero 특성 (밀도 ~3.3%)
- **노이즈 레벨**: 0.1 (조정 가능)

## 🤖 구현된 GPR 모델들

### 기본 모델들
- **Standard GPR**: RBF, Matérn 커널 사용
- **Sparse Variational GP (SVGP)**: 유도점 기반 변분 추론

### 고급 모델들  
- **Actually Sparse VGP**: 진정한 희소성 달성하는 변분 GP
- **Sparse Orthogonal VI**: 직교 분해 기반 희소 변분 추론
- **Robust Sparse GPR**: 이상치에 강건한 희소 GP
- **Deep Sparse GP**: 다층 GP with 희소 유도점

## 🚀 사용법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 개별 모델 실험

```python
# 기본 GPR 모델들만 실험
python gpr_sparse_comparison.py
```

### 3. 종합 실험 실행

```python
# 모든 모델들을 포함한 종합 비교 실험
python comprehensive_gpr_experiment.py
```

### 4. 고급 모델들만 테스트

```python
# 고급 GPR 모델들 개별 테스트
python advanced_gpr_models.py
```

## 📈 평가 지표

### 예측 정확도
- **MSE (Mean Squared Error)**: 예측 오차
- **MAE (Mean Absolute Error)**: 절대 예측 오차  
- **R² Score**: 결정계수

### 불확실성 정량화
- **Mean Uncertainty**: 평균 예측 불확실성
- **MSE-σ Correlation**: 예측 오차와 불확실성 간 상관관계
- **Calibration Quality**: 불확실성 보정 품질

### 계산 효율성
- **Training Time**: 모델 훈련 시간
- **Prediction Time**: 예측 시간
- **Memory Usage**: 메모리 사용량

## 📋 주요 파일 구조

```
c-project/
├── gpr_sparse_comparison.py      # 기본 GPR 모델들 + 실험 프레임워크
├── advanced_gpr_models.py        # 고급 GPR 모델들 구현
├── comprehensive_gpr_experiment.py # 종합 실험 스크립트
├── requirements.txt              # 필요한 패키지 목록
└── README.md                     # 이 파일
```

## 🔬 실험 결과

실험을 실행하면 다음과 같은 결과를 얻을 수 있습니다:

1. **성능 비교 차트**: 모든 모델의 MSE, R², 불확실성 비교
2. **MSE-σ 상관관계 분석**: 각 모델의 불확실성 정량화 품질
3. **계산 효율성 분석**: 정확도 대비 계산 비용 비교
4. **종합 보고서**: `gpr_comparison_report.md` 파일로 저장

## 🎨 시각화

- **성능 비교 대시보드**: 9개 차트로 구성된 종합 성능 비교
- **MSE-σ 상관관계 플롯**: 각 모델별 오차-불확실성 관계 시각화
- **효율성 분석**: 정확도 vs 계산시간 트레이드오프 분석

## 🔧 커스터마이징

### 데이터 설정 변경

```python
experiment = ComprehensiveGPRExperiment(
    n_samples=5000,      # 샘플 수 증가
    n_features=500,      # 차원 증가  
    density=0.05,        # 희소성 조정
    noise_level=0.05,    # 노이즈 레벨 조정
)
```

### 모델 하이퍼파라미터 조정

```python
# SVGP 설정 예시
svgp = SVGPWrapper(
    n_inducing=200,      # 유도점 수 증가
    kernel_type='matern', # 커널 변경
    epochs=100,          # 훈련 에포크 증가
    lr=0.001            # 학습률 조정
)
```

## 📚 주요 개념

### Sparse Gaussian Process
- 전체 데이터 대신 유도점(Inducing Points) 사용으로 계산 복잡도 감소
- O(N³)에서 O(M²N)으로 복잡도 개선 (M << N)

### MSE-σ 상관관계
- 예측 오차와 모델의 불확실성 추정 간 상관관계
- 높은 상관관계 = 모델이 자신의 예측 불확실성을 정확히 인지
- 신뢰할 수 있는 불확실성 정량화의 핵심 지표

### 희소성 활용
- 실제 데이터의 희소 구조를 활용한 효율적 GP 구현
- 불필요한 특성 자동 제거를 통한 진정한 희소성 달성

## 🤝 기여하기

이 연구에 기여하고 싶으시다면:

1. 새로운 GPR 모델 구현
2. 평가 지표 추가
3. 시각화 개선
4. 성능 최적화

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 자유롭게 사용할 수 있습니다.

## 📞 문의

구현 관련 질문이나 개선 제안이 있으시면 이슈를 생성해 주세요.

---

**희소 데이터에서의 GPR 모델 비교 연구** - 실용적이고 신뢰할 수 있는 불확실성 정량화를 위한 모델 선택 가이드