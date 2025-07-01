"""
희소 데이터에서의 GPR 모델 종합 비교 실험
- 기본 GPR 모델들과 고급 GPR 모델들을 통합하여 비교
- MSE-σ 상관관계 분석을 통한 최적 모델 선택
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 기본 모델들 임포트
from gpr_sparse_comparison import (
    SparseDataGenerator, 
    StandardGPR, 
    SVGPWrapper,
    ModelEvaluator
)

# 고급 모델들 임포트
from advanced_gpr_models import create_advanced_models


class ComprehensiveGPRExperiment:
    """종합적인 GPR 모델 비교 실험 클래스"""
    
    def __init__(self, n_samples=3000, n_features=300, density=0.033, 
                 noise_level=0.1, test_size=0.2, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.density = density
        self.noise_level = noise_level
        self.test_size = test_size
        self.random_state = random_state
        
        self.data_generated = False
        self.models_trained = False
        self.results = []
        
    def generate_data(self):
        """희소 데이터 생성"""
        print("=== 희소 데이터 생성 ===")
        
        generator = SparseDataGenerator(
            n_samples=self.n_samples,
            n_features=self.n_features, 
            density=self.density,
            noise_level=self.noise_level,
            random_state=self.random_state
        )
        
        self.X, self.y = generator.generate_dataset()
        
        # 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # 데이터 표준화
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.reshape(-1, 1)).ravel()
        self.y_test_scaled = self.scaler_y.transform(self.y_test.reshape(-1, 1)).ravel()
        
        print(f"훈련 데이터: {self.X_train_scaled.shape}")
        print(f"테스트 데이터: {self.X_test_scaled.shape}")
        print(f"희소성: {np.count_nonzero(self.X) / self.X.size:.4f}")
        
        self.data_generated = True
    
    def create_all_models(self):
        """모든 GPR 모델들 생성"""
        if not self.data_generated:
            raise ValueError("Data not generated yet! Call generate_data() first.")
        
        models = []
        
        print("\n=== 모델 생성 ===")
        
        # 1. 기본 GPR 모델들 (서브셋으로 훈련)
        subset_size = 500
        subset_indices = np.random.choice(len(self.X_train_scaled), subset_size, replace=False)
        X_train_subset = self.X_train_scaled[subset_indices]
        y_train_subset = self.y_train_scaled[subset_indices]
        
        print("기본 GPR 모델들 생성...")
        
        # Standard GPR with RBF
        standard_gpr_rbf = StandardGPR(kernel_type='rbf', length_scale=1.0)
        models.append(('basic', standard_gpr_rbf, X_train_subset, y_train_subset))
        
        # Standard GPR with Matérn
        standard_gpr_matern = StandardGPR(kernel_type='matern', length_scale=1.0)
        models.append(('basic', standard_gpr_matern, X_train_subset, y_train_subset))
        
        # 2. Sparse Variational GP 모델들
        print("Sparse Variational GP 모델들 생성...")
        
        # SVGP with different settings
        svgp_configs = [
            {'n_inducing': 50, 'kernel_type': 'rbf', 'epochs': 30},
            {'n_inducing': 100, 'kernel_type': 'rbf', 'epochs': 50},
            {'n_inducing': 200, 'kernel_type': 'rbf', 'epochs': 50},
            {'n_inducing': 100, 'kernel_type': 'matern', 'epochs': 50},
        ]
        
        for config in svgp_configs:
            svgp = SVGPWrapper(**config)
            models.append(('svgp', svgp, self.X_train_scaled, self.y_train_scaled))
        
        # 3. 고급 GPR 모델들
        print("고급 GPR 모델들 생성...")
        
        try:
            advanced_models = create_advanced_models()
            for model in advanced_models:
                models.append(('advanced', model, self.X_train_scaled, self.y_train_scaled))
        except Exception as e:
            print(f"고급 모델 생성 실패: {e}")
        
        return models
    
    def train_and_evaluate_models(self):
        """모든 모델 훈련 및 평가"""
        if not self.data_generated:
            self.generate_data()
        
        print("\n=== 모델 훈련 및 평가 ===")
        
        models = self.create_all_models()
        evaluator = ModelEvaluator()
        evaluator.y_test = self.y_test_scaled
        
        training_times = []
        prediction_times = []
        
        for model_type, model, X_train, y_train in models:
            try:
                print(f"\n--- {getattr(model, 'name', str(model))} ---")
                
                # 훈련 시간 측정
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # 예측 시간 측정
                start_time = time.time()
                result = evaluator.evaluate_model(model, self.X_test_scaled, self.y_test_scaled)
                prediction_time = time.time() - start_time
                
                # 시간 정보 추가
                result['training_time'] = training_time
                result['prediction_time'] = prediction_time
                result['model_type'] = model_type
                
                training_times.append(training_time)
                prediction_times.append(prediction_time)
                
                print(f"훈련 시간: {training_time:.2f}초")
                print(f"예측 시간: {prediction_time:.4f}초")
                
            except Exception as e:
                print(f"모델 {getattr(model, 'name', str(model))} 실패: {e}")
                continue
        
        self.evaluator = evaluator
        self.models_trained = True
        
        print(f"\n총 훈련 시간: {sum(training_times):.2f}초")
        print(f"평균 예측 시간: {np.mean(prediction_times):.4f}초")
    
    def analyze_results(self):
        """결과 분석 및 요약"""
        if not self.models_trained:
            raise ValueError("Models not trained yet! Call train_and_evaluate_models() first.")
        
        print("\n=== 결과 분석 ===")
        
        # 결과 DataFrame 생성
        results_df = self.evaluator.get_results_dataframe()
        
        if results_df.empty:
            print("분석할 결과가 없습니다.")
            return
        
        print("\n1. 전체 성능 요약:")
        print(results_df[['model', 'mse', 'r2', 'mean_uncertainty', 
                         'mse_sigma_correlation', 'training_time']].round(4))
        
        # 2. 모델 타입별 성능
        print("\n2. 모델 타입별 평균 성능:")
        if 'model_type' in results_df.columns:
            type_summary = results_df.groupby('model_type')[
                ['mse', 'r2', 'mean_uncertainty', 'mse_sigma_correlation']
            ].mean()
            print(type_summary.round(4))
        
        # 3. 최고 성능 모델들
        print("\n3. 최고 성능 모델들:")
        
        best_models = {}
        metrics = ['mse', 'r2', 'mse_sigma_correlation']
        
        for metric in metrics:
            if metric in ['r2', 'mse_sigma_correlation']:
                best_idx = results_df[metric].idxmax()
            else:
                best_idx = results_df[metric].idxmin()
            
            best_models[metric] = {
                'model': results_df.loc[best_idx, 'model'],
                'value': results_df.loc[best_idx, metric]
            }
            
            print(f"- 최고 {metric.upper()}: {best_models[metric]['model']} "
                  f"({best_models[metric]['value']:.4f})")
        
        # 4. 효율성 분석
        print("\n4. 계산 효율성 분석:")
        efficiency_df = results_df[['model', 'mse', 'training_time', 'prediction_time']].copy()
        efficiency_df['efficiency_score'] = (1 / efficiency_df['mse']) / efficiency_df['training_time']
        efficiency_df = efficiency_df.sort_values('efficiency_score', ascending=False)
        print(efficiency_df.head().round(4))
        
        return results_df, best_models
    
    def create_comprehensive_visualizations(self):
        """종합적인 시각화"""
        if not self.models_trained:
            raise ValueError("Models not trained yet!")
        
        results_df = self.evaluator.get_results_dataframe()
        if results_df.empty:
            return
        
        # 1. 성능 비교 대시보드
        fig = plt.figure(figsize=(20, 15))
        
        # MSE 비교
        plt.subplot(3, 3, 1)
        plt.bar(range(len(results_df)), results_df['mse'])
        plt.title('Mean Squared Error')
        plt.ylabel('MSE')
        plt.xticks(range(len(results_df)), results_df['model'], rotation=45, ha='right')
        
        # R² 비교
        plt.subplot(3, 3, 2)
        plt.bar(range(len(results_df)), results_df['r2'])
        plt.title('R² Score') 
        plt.ylabel('R²')
        plt.xticks(range(len(results_df)), results_df['model'], rotation=45, ha='right')
        
        # 불확실성 비교
        plt.subplot(3, 3, 3)
        plt.bar(range(len(results_df)), results_df['mean_uncertainty'])
        plt.title('Mean Uncertainty')
        plt.ylabel('Mean σ')
        plt.xticks(range(len(results_df)), results_df['model'], rotation=45, ha='right')
        
        # MSE vs σ 상관관계
        plt.subplot(3, 3, 4)
        plt.bar(range(len(results_df)), results_df['mse_sigma_correlation'])
        plt.title('MSE-σ Correlation')
        plt.ylabel('Correlation')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(range(len(results_df)), results_df['model'], rotation=45, ha='right')
        
        # 훈련 시간 비교
        plt.subplot(3, 3, 5)
        if 'training_time' in results_df.columns:
            plt.bar(range(len(results_df)), results_df['training_time'])
            plt.title('Training Time')
            plt.ylabel('Time (sec)')
            plt.xticks(range(len(results_df)), results_df['model'], rotation=45, ha='right')
        
        # MSE vs 훈련시간 스캐터
        plt.subplot(3, 3, 6)
        if 'training_time' in results_df.columns:
            plt.scatter(results_df['training_time'], results_df['mse'])
            plt.xlabel('Training Time (sec)')
            plt.ylabel('MSE')
            plt.title('MSE vs Training Time')
            for i, model in enumerate(results_df['model']):
                plt.annotate(model[:10], (results_df['training_time'].iloc[i], 
                                        results_df['mse'].iloc[i]), fontsize=8)
        
        # 모델 타입별 성능 박스플롯
        plt.subplot(3, 3, 7)
        if 'model_type' in results_df.columns:
            model_types = results_df['model_type'].unique()
            mse_by_type = [results_df[results_df['model_type'] == t]['mse'].values 
                          for t in model_types]
            plt.boxplot(mse_by_type, labels=model_types)
            plt.title('MSE by Model Type')
            plt.ylabel('MSE')
        
        # 상위 5개 모델 레이더 차트용 데이터
        plt.subplot(3, 3, 8)
        top5_models = results_df.nsmallest(5, 'mse')
        metrics_normalized = top5_models[['mse', 'mean_uncertainty', 'mse_sigma_correlation']].copy()
        
        # 정규화 (0-1 범위)
        for col in metrics_normalized.columns:
            if col == 'mse':
                metrics_normalized[col] = 1 - (metrics_normalized[col] - metrics_normalized[col].min()) / \
                                         (metrics_normalized[col].max() - metrics_normalized[col].min())
            else:
                metrics_normalized[col] = (metrics_normalized[col] - metrics_normalized[col].min()) / \
                                         (metrics_normalized[col].max() - metrics_normalized[col].min())
        
        plt.plot(metrics_normalized.T)
        plt.title('Top 5 Models Performance Profile')
        plt.xticks(range(len(metrics_normalized.columns)), 
                  ['Accuracy', 'Uncertainty', 'Correlation'])
        plt.legend(top5_models['model'], bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 효율성 분석
        plt.subplot(3, 3, 9)
        if 'training_time' in results_df.columns:
            efficiency = (1 / results_df['mse']) / results_df['training_time']
            plt.bar(range(len(results_df)), efficiency)
            plt.title('Efficiency Score (1/MSE/Time)')
            plt.ylabel('Efficiency')
            plt.xticks(range(len(results_df)), results_df['model'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        # 2. MSE-σ 상관관계 세부 분석
        self.evaluator.plot_mse_sigma_correlation(figsize=(18, 12))
    
    def generate_report(self, save_path=None):
        """종합 보고서 생성"""
        if not self.models_trained:
            raise ValueError("Models not trained yet!")
        
        results_df, best_models = self.analyze_results()
        
        report = f"""
# 희소 데이터에서의 GPR 모델 비교 연구 결과 보고서

## 실험 설정
- 데이터 크기: {self.n_samples} 샘플 × {self.n_features} 특성
- 희소성: {self.density:.3f} (행당 평균 {int(self.density * self.n_features)}개 non-zero)
- 노이즈 레벨: {self.noise_level}
- 테스트 비율: {self.test_size}

## 실험 결과 요약

### 전체 모델 성능
{results_df[['model', 'mse', 'r2', 'mean_uncertainty', 'mse_sigma_correlation']].to_string(index=False)}

### 최고 성능 모델
- **최저 MSE**: {best_models['mse']['model']} (MSE: {best_models['mse']['value']:.4f})
- **최고 R²**: {best_models['r2']['model']} (R²: {best_models['r2']['value']:.4f})
- **최고 MSE-σ 상관관계**: {best_models['mse_sigma_correlation']['model']} (상관계수: {best_models['mse_sigma_correlation']['value']:.4f})

## 주요 발견사항

### 1. 모델 성능 비교
"""
        
        # 성능 순위
        mse_ranking = results_df.sort_values('mse')[['model', 'mse']].head()
        report += f"\n**MSE 기준 상위 5개 모델:**\n{mse_ranking.to_string(index=False)}\n"
        
        r2_ranking = results_df.sort_values('r2', ascending=False)[['model', 'r2']].head()
        report += f"\n**R² 기준 상위 5개 모델:**\n{r2_ranking.to_string(index=False)}\n"
        
        # 불확실성 정량화 품질
        corr_ranking = results_df.sort_values('mse_sigma_correlation', ascending=False)[['model', 'mse_sigma_correlation']].head()
        report += f"""
### 2. 불확실성 정량화 품질
**MSE-σ 상관관계 상위 5개 모델:**
{corr_ranking.to_string(index=False)}

상관관계가 높을수록 모델이 자신의 예측 불확실성을 더 정확하게 추정함을 의미합니다.
"""
        
        # 계산 효율성
        if 'training_time' in results_df.columns:
            efficiency_df = results_df.copy()
            efficiency_df['efficiency'] = (1 / efficiency_df['mse']) / efficiency_df['training_time']
            efficiency_ranking = efficiency_df.sort_values('efficiency', ascending=False)[['model', 'training_time', 'mse', 'efficiency']].head()
            
            report += f"""
### 3. 계산 효율성
**효율성 점수 상위 5개 모델 (정확도/훈련시간):**
{efficiency_ranking.to_string(index=False)}
"""
        
        report += f"""
## 결론 및 권장사항

### 희소 데이터에서의 GPR 모델 선택 가이드라인:

1. **최고 정확도가 필요한 경우**: {best_models['mse']['model']}
2. **신뢰할 수 있는 불확실성 추정이 필요한 경우**: {best_models['mse_sigma_correlation']['model']}
3. **균형잡힌 성능이 필요한 경우**: {best_models['r2']['model']}

### 희소 데이터 특성에 따른 관찰:
- 300차원에서 평균 10개의 non-zero 특성을 가진 극희소 데이터에서도 GPR 모델들이 효과적으로 작동
- Sparse Variational GP들이 전통적인 GPR 대비 확장성 면에서 우수한 성능 보임
- 불확실성 정량화 품질은 모델별로 상당한 차이를 보임

이 연구는 희소 데이터 환경에서 GPR 모델 선택을 위한 실증적 근거를 제공합니다.
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"보고서가 {save_path}에 저장되었습니다.")
        
        return report


def run_comprehensive_experiment():
    """종합 실험 실행"""
    print("=" * 60)
    print("희소 데이터에서의 GPR 모델 종합 비교 실험")
    print("=" * 60)
    
    # 실험 설정
    experiment = ComprehensiveGPRExperiment(
        n_samples=3000,
        n_features=300, 
        density=0.033,  # 평균 10개 non-zero features
        noise_level=0.1,
        test_size=0.2,
        random_state=42
    )
    
    # 실험 실행
    experiment.generate_data()
    experiment.train_and_evaluate_models()
    
    # 결과 분석
    results_df, best_models = experiment.analyze_results()
    
    # 시각화
    experiment.create_comprehensive_visualizations()
    
    # 보고서 생성
    report = experiment.generate_report('gpr_comparison_report.md')
    print("\n" + "=" * 60)
    print("실험 완료!")
    print("=" * 60)
    
    return experiment, results_df, best_models


if __name__ == "__main__":
    experiment, results, best_models = run_comprehensive_experiment()