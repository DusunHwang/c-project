"""
희소 데이터에서의 GPR 모델 비교 연구
- 300차원, 3000개 샘플의 희소 데이터에서 다양한 GPR 모델 성능 비교
- MSE와 모델의 sigma(불확실성) 값 간 상관관계 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import gpytorch
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
import warnings
warnings.filterwarnings('ignore')

class SparseDataGenerator:
    """희소 데이터 생성 클래스"""
    
    def __init__(self, n_samples=3000, n_features=300, density=0.033, noise_level=0.1, random_state=42):
        """
        Args:
            n_samples: 샘플 수 (기본: 3000)
            n_features: 특성 수 (기본: 300) 
            density: 희소성 밀도 - 평균 10/300 = 0.033 (기본: 0.033)
            noise_level: 노이즈 레벨 (기본: 0.1)
            random_state: 랜덤 시드 (기본: 42)
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.density = density
        self.noise_level = noise_level
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_sparse_features(self):
        """희소 특성 데이터 생성"""
        # 희소 행렬 생성 - 각 행당 평균 10개의 non-zero 값
        X_sparse = sparse_random(
            self.n_samples, 
            self.n_features, 
            density=self.density,
            format='csr',
            random_state=self.random_state
        )
        
        # 값들을 정규분포에서 샘플링하여 대체
        X_sparse.data = np.random.normal(0, 1, len(X_sparse.data))
        
        # dense 형태로 변환
        X = X_sparse.toarray()
        
        return X
    
    def generate_target_function(self, X):
        """복잡한 타겟 함수 생성"""
        # 몇 개의 중요한 특성만 선택
        important_features = [10, 50, 100, 150, 200, 250]
        
        # 선형 조합
        y = np.zeros(X.shape[0])
        for i, feat_idx in enumerate(important_features):
            weight = 2.0 / (i + 1)  # importance weights
            y += weight * X[:, feat_idx]
        
        # 비선형 상호작용 추가
        y += 0.5 * X[:, 10] * X[:, 50]
        y += 0.3 * np.sin(X[:, 100] * 2)
        y += 0.2 * np.exp(-X[:, 150]**2)
        
        # 노이즈 추가
        noise = np.random.normal(0, self.noise_level, len(y))
        y += noise
        
        return y
    
    def generate_dataset(self):
        """완전한 데이터셋 생성"""
        print(f"희소 데이터 생성: {self.n_samples}x{self.n_features}, density={self.density:.3f}")
        
        X = self.generate_sparse_features()
        y = self.generate_target_function(X)
        
        # 통계 정보 출력
        non_zero_per_row = np.mean(np.count_nonzero(X, axis=1))
        print(f"행당 평균 non-zero 특성 수: {non_zero_per_row:.1f}")
        print(f"전체 희소성: {np.count_nonzero(X) / X.size:.4f}")
        print(f"타겟 변수 범위: [{y.min():.3f}, {y.max():.3f}]")
        
        return X, y


class StandardGPR:
    """표준 Gaussian Process Regression"""
    
    def __init__(self, kernel_type='rbf', length_scale=1.0, noise_level=0.1):
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.model = None
        self.name = f"Standard_GPR_{kernel_type}"
    
    def create_kernel(self):
        """커널 생성"""
        if self.kernel_type == 'rbf':
            kernel = RBF(length_scale=self.length_scale)
        elif self.kernel_type == 'matern':
            kernel = Matern(length_scale=self.length_scale, nu=1.5)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # 노이즈 커널 추가
        kernel += WhiteKernel(noise_level=self.noise_level)
        return kernel
    
    def fit(self, X_train, y_train):
        """모델 훈련"""
        kernel = self.create_kernel()
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=3
        )
        
        print(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test, return_std=True):
        """예측 수행"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        y_pred, y_std = self.model.predict(X_test, return_std=True)
        
        if return_std:
            return y_pred, y_std
        return y_pred


class SparseVariationalGP(ApproximateGP):
    """Sparse Variational Gaussian Process (GPyTorch 기반)"""
    
    def __init__(self, inducing_points, kernel_type='rbf'):
        # Variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseVariationalGP, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        if kernel_type == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_type == 'matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        
        self.name = f"SVGP_{kernel_type}"
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPWrapper:
    """SVGP 모델 래퍼 클래스"""
    
    def __init__(self, n_inducing=100, kernel_type='rbf', lr=0.01, epochs=100):
        self.n_inducing = n_inducing
        self.kernel_type = kernel_type
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.likelihood = None
        self.name = f"SVGP_{kernel_type}_M{n_inducing}"
    
    def fit(self, X_train, y_train):
        """모델 훈련"""
        print(f"Training {self.name}...")
        
        # 텐서로 변환
        train_x = torch.FloatTensor(X_train)
        train_y = torch.FloatTensor(y_train)
        
        # Inducing points 초기화 (랜덤 선택)
        inducing_indices = torch.randperm(len(train_x))[:self.n_inducing]
        inducing_points = train_x[inducing_indices].clone()
        
        # 모델 및 likelihood 생성
        self.model = SparseVariationalGP(inducing_points, self.kernel_type)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # 훈련 모드
        self.model.train()
        self.likelihood.train()
        
        # 옵티마이저 설정
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.lr)
        
        # 손실 함수
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))
        
        # 훈련 루프
        losses = []
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        return self
    
    def predict(self, X_test, return_std=True):
        """예측 수행"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # 예측 모드
        self.model.eval()
        self.likelihood.eval()
        
        test_x = torch.FloatTensor(X_test)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            y_pred = observed_pred.mean.numpy()
            
            if return_std:
                y_std = observed_pred.stddev.numpy()
                return y_pred, y_std
            return y_pred


class ModelEvaluator:
    """모델 성능 평가 클래스"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_model(self, model, X_test, y_test, model_name=None):
        """단일 모델 평가"""
        if model_name is None:
            model_name = getattr(model, 'name', 'Unknown')
        
        # 예측 수행
        y_pred, y_std = model.predict(X_test, return_std=True)
        
        # 메트릭 계산
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 불확실성 메트릭
        mean_std = np.mean(y_std)
        std_std = np.std(y_std)
        
        # MSE-σ 상관관계
        mse_sigma_corr = np.corrcoef(np.abs(y_test - y_pred), y_std)[0, 1]
        
        result = {
            'model': model_name,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mean_uncertainty': mean_std,
            'std_uncertainty': std_std,
            'mse_sigma_correlation': mse_sigma_corr,
            'predictions': y_pred,
            'uncertainties': y_std
        }
        
        self.results.append(result)
        
        print(f"\n=== {model_name} 성능 ===")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"평균 불확실성: {mean_std:.4f}")
        print(f"MSE-σ 상관계수: {mse_sigma_corr:.4f}")
        
        return result
    
    def get_results_dataframe(self):
        """결과를 DataFrame으로 반환"""
        if not self.results:
            return pd.DataFrame()
        
        df_data = []
        for result in self.results:
            row = {k: v for k, v in result.items() 
                   if k not in ['predictions', 'uncertainties']}
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def plot_mse_sigma_correlation(self, figsize=(15, 10)):
        """MSE-σ 상관관계 시각화"""
        if not self.results:
            print("No results to plot!")
            return
        
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if n_models == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, result in enumerate(self.results):
            ax = axes[i]
            
            y_pred = result['predictions']
            y_std = result['uncertainties']
            
            # 예측 오차와 불확실성 산점도
            if hasattr(self, 'y_test'):
                errors = np.abs(self.y_test - y_pred)
                ax.scatter(y_std, errors, alpha=0.6, s=20)
                ax.set_xlabel('Predicted Uncertainty (σ)')
                ax.set_ylabel('Absolute Error')
                
                # 상관계수 표시
                corr = result['mse_sigma_correlation']
                ax.set_title(f"{result['model']}\nCorr: {corr:.3f}")
                
                # 추세선
                z = np.polyfit(y_std, errors, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(y_std.min(), y_std.max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)
            else:
                ax.text(0.5, 0.5, 'No test data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(result['model'])
        
        # 빈 subplot 숨기기
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_comparison(self, figsize=(12, 8)):
        """모델 성능 비교 시각화"""
        df = self.get_results_dataframe()
        if df.empty:
            print("No results to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # MSE 비교
        axes[0,0].bar(df['model'], df['mse'])
        axes[0,0].set_title('Mean Squared Error')
        axes[0,0].set_ylabel('MSE')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # R² 비교
        axes[0,1].bar(df['model'], df['r2'])
        axes[0,1].set_title('R² Score')
        axes[0,1].set_ylabel('R²')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 평균 불확실성 비교
        axes[1,0].bar(df['model'], df['mean_uncertainty'])
        axes[1,0].set_title('Mean Uncertainty')
        axes[1,0].set_ylabel('Mean σ')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # MSE-σ 상관계수 비교
        axes[1,1].bar(df['model'], df['mse_sigma_correlation'])
        axes[1,1].set_title('MSE-σ Correlation')
        axes[1,1].set_ylabel('Correlation')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()


def main():
    """메인 실험 실행"""
    print("=== 희소 데이터에서의 GPR 모델 비교 연구 ===\n")
    
    # 1. 데이터 생성
    generator = SparseDataGenerator(
        n_samples=3000, 
        n_features=300, 
        density=0.033,  # 평균 10개 non-zero features
        noise_level=0.1
    )
    
    X, y = generator.generate_dataset()
    
    # 2. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. 데이터 표준화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    print(f"\n훈련 데이터: {X_train_scaled.shape}")
    print(f"테스트 데이터: {X_test_scaled.shape}")
    
    # 4. 모델 정의 및 훈련
    models = []
    
    # 4.1 표준 GPR (소규모 서브셋으로)
    subset_size = 500  # 계산 효율성을 위해 서브셋 사용
    subset_indices = np.random.choice(len(X_train_scaled), subset_size, replace=False)
    X_train_subset = X_train_scaled[subset_indices]
    y_train_subset = y_train_scaled[subset_indices]
    
    standard_gpr_rbf = StandardGPR(kernel_type='rbf', length_scale=1.0)
    standard_gpr_rbf.fit(X_train_subset, y_train_subset)
    models.append(standard_gpr_rbf)
    
    standard_gpr_matern = StandardGPR(kernel_type='matern', length_scale=1.0)
    standard_gpr_matern.fit(X_train_subset, y_train_subset)
    models.append(standard_gpr_matern)
    
    # 4.2 Sparse Variational GP
    svgp_rbf = SVGPWrapper(n_inducing=100, kernel_type='rbf', epochs=50)
    svgp_rbf.fit(X_train_scaled, y_train_scaled)
    models.append(svgp_rbf)
    
    svgp_matern = SVGPWrapper(n_inducing=100, kernel_type='matern', epochs=50)
    svgp_matern.fit(X_train_scaled, y_train_scaled)
    models.append(svgp_matern)
    
    # 다양한 inducing points 수로 실험
    svgp_rbf_200 = SVGPWrapper(n_inducing=200, kernel_type='rbf', epochs=50)
    svgp_rbf_200.fit(X_train_scaled, y_train_scaled)
    models.append(svgp_rbf_200)
    
    # 5. 모델 평가
    evaluator = ModelEvaluator()
    evaluator.y_test = y_test_scaled  # 시각화를 위해 저장
    
    for model in models:
        evaluator.evaluate_model(model, X_test_scaled, y_test_scaled)
    
    # 6. 결과 분석 및 시각화
    print("\n=== 전체 결과 요약 ===")
    results_df = evaluator.get_results_dataframe()
    print(results_df.round(4))
    
    # 시각화
    evaluator.plot_performance_comparison()
    evaluator.plot_mse_sigma_correlation()
    
    # 7. 최고 성능 모델 찾기
    best_mse_model = results_df.loc[results_df['mse'].idxmin()]
    best_r2_model = results_df.loc[results_df['r2'].idxmax()]
    best_corr_model = results_df.loc[results_df['mse_sigma_correlation'].idxmax()]
    
    print(f"\n=== 최고 성능 모델 ===")
    print(f"최저 MSE: {best_mse_model['model']} (MSE: {best_mse_model['mse']:.4f})")
    print(f"최고 R²: {best_r2_model['model']} (R²: {best_r2_model['r2']:.4f})")
    print(f"최고 MSE-σ 상관관계: {best_corr_model['model']} (상관계수: {best_corr_model['mse_sigma_correlation']:.4f})")
    
    return evaluator, results_df

if __name__ == "__main__":
    evaluator, results = main()