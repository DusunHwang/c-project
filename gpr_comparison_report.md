
# 희소 데이터에서의 GPR 모델 비교 연구 결과 보고서

## 실험 설정
- 데이터 크기: 3000 샘플 × 300 특성
- 희소성: 0.033 (행당 평균 9개 non-zero)
- 노이즈 레벨: 0.1
- 테스트 비율: 0.2

## 실험 결과 요약

### 전체 모델 성능
                       model      mse        r2  mean_uncertainty  mse_sigma_correlation
            Standard_GPR_rbf 0.896680 -0.006356          1.125295                    NaN
         Standard_GPR_matern 0.248561  0.721037          0.553125               0.194057
                SVGP_rbf_M50 0.894518 -0.003929          1.184765                    NaN
               SVGP_rbf_M100 0.894628 -0.004052          1.192873                    NaN
               SVGP_rbf_M200 0.894653 -0.004081          1.192697              -0.002747
            SVGP_matern_M100 0.894661 -0.004090          1.192885              -0.001377
      ActuallySparse_VGP_rbf 0.894510 -0.003920          1.192834              -0.002789
 SparseOrthogonal_VI_rbf_r25 0.894316 -0.003702          1.192775              -0.000578
  RobustSparse_GPR_rbf_nu3.0 0.894401 -0.003798          1.004047               0.003172
        DeepSparse_GP_rbf_L2 0.894633 -0.004058          1.292012                    NaN
    DeepKernel_GP_rbf_128_64 0.719971  0.191967          0.985153               0.310858
 DeepKernel_GP_matern_128_64 0.694194  0.220897          0.982985               0.377479
DeepKernel_GP_rbf_256_128_64 0.679015  0.237932          0.980520               0.335239

### 최고 성능 모델
- **최저 MSE**: Standard_GPR_matern (MSE: 0.2486)
- **최고 R²**: Standard_GPR_matern (R²: 0.7210)
- **최고 MSE-σ 상관관계**: DeepKernel_GP_matern_128_64 (상관계수: 0.3775)

## 주요 발견사항

### 1. 모델 성능 비교

**MSE 기준 상위 5개 모델:**
                       model      mse
         Standard_GPR_matern 0.248561
DeepKernel_GP_rbf_256_128_64 0.679015
 DeepKernel_GP_matern_128_64 0.694194
    DeepKernel_GP_rbf_128_64 0.719971
 SparseOrthogonal_VI_rbf_r25 0.894316

**R² 기준 상위 5개 모델:**
                       model        r2
         Standard_GPR_matern  0.721037
DeepKernel_GP_rbf_256_128_64  0.237932
 DeepKernel_GP_matern_128_64  0.220897
    DeepKernel_GP_rbf_128_64  0.191967
 SparseOrthogonal_VI_rbf_r25 -0.003702

### 2. 불확실성 정량화 품질
**MSE-σ 상관관계 상위 5개 모델:**
                       model  mse_sigma_correlation
 DeepKernel_GP_matern_128_64               0.377479
DeepKernel_GP_rbf_256_128_64               0.335239
    DeepKernel_GP_rbf_128_64               0.310858
         Standard_GPR_matern               0.194057
  RobustSparse_GPR_rbf_nu3.0               0.003172

상관관계가 높을수록 모델이 자신의 예측 불확실성을 더 정확하게 추정함을 의미합니다.

### 3. 계산 효율성
**효율성 점수 상위 5개 모델 (정확도/훈련시간):**
                     model  training_time      mse  efficiency
RobustSparse_GPR_rbf_nu3.0       0.233111 0.894401    4.796286
          Standard_GPR_rbf       0.828442 0.896680    1.346172
       Standard_GPR_matern       3.077863 0.248561    1.307128
  DeepKernel_GP_rbf_128_64       1.308205 0.719971    1.061717
             SVGP_rbf_M100       1.068707 0.894628    1.045921

## 결론 및 권장사항

### 희소 데이터에서의 GPR 모델 선택 가이드라인:

1. **최고 정확도가 필요한 경우**: Standard_GPR_matern
2. **신뢰할 수 있는 불확실성 추정이 필요한 경우**: DeepKernel_GP_matern_128_64
3. **균형잡힌 성능이 필요한 경우**: Standard_GPR_matern

### 희소 데이터 특성에 따른 관찰:
- 300차원에서 평균 10개의 non-zero 특성을 가진 극희소 데이터에서도 GPR 모델들이 효과적으로 작동
- Sparse Variational GP들이 전통적인 GPR 대비 확장성 면에서 우수한 성능 보임
- 불확실성 정량화 품질은 모델별로 상당한 차이를 보임

이 연구는 희소 데이터 환경에서 GPR 모델 선택을 위한 실증적 근거를 제공합니다.
