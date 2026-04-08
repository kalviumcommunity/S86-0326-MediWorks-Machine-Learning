"""
linear_regression_best_practices.py
-----------------------------------
Best practices guide for Linear Regression: handling real-world challenges like
multicollinearity, outliers, feature scaling, and model selection.

Topics Covered
--------------
1. Detecting and handling multicollinearity
2. Outlier detection and robust regression
3. Feature scaling strategies
4. Model selection (Linear vs Ridge vs Lasso)
5. Cross-validation strategies
6. Assumption checking
7. When to use regularization

Run this file to understand practical considerations when deploying
Linear Regression models to production.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.robust import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# 1. MULTICOLLINEARITY DETECTION
# ============================================================================

def detect_multicollinearity(X: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Detect multicollinearity using correlation analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with column names.
    threshold : float
        Correlation threshold (default: 0.8).

    Returns
    -------
    pd.DataFrame
        DataFrame of correlated feature pairs.
    """
    corr_matrix = X.corr().abs()

    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find correlations above threshold
    correlated_features = [
        (column, row, upper.loc[row, column])
        for column in upper.columns
        for row in upper.index
        if upper.loc[row, column] > threshold
    ]

    if correlated_features:
        result_df = pd.DataFrame(
            correlated_features,
            columns=["Feature1", "Feature2", "Correlation"]
        ).sort_values("Correlation", ascending=False)
        return result_df
    else:
        return pd.DataFrame(columns=["Feature1", "Feature2", "Correlation"])


def compute_variance_inflation_factor(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each feature.

    VIF > 5 suggests multicollinearity.
    VIF > 10 indicates problematic multicollinearity.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.DataFrame
        VIF scores for each feature.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]

    return vif_data.sort_values("VIF", ascending=False)


# ============================================================================
# 2. OUTLIER DETECTION AND ROBUST REGRESSION
# ============================================================================

def detect_outliers_zscore(y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using z-score method.

    Points with |z-score| > threshold are flagged as outliers.

    Parameters
    ----------
    y : np.ndarray
        Target values.
    threshold : float
        Z-score threshold (default: 3.0).

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates outlier.
    """
    z_scores = np.abs((y - np.mean(y)) / np.std(y))
    return z_scores > threshold


def detect_outliers_iqr(y: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Outliers are defined as points outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].

    Parameters
    ----------
    y : np.ndarray
        Target values.
    multiplier : float
        IQR multiplier (default 1.5 is standard Tukey fences).

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates outlier.
    """
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return (y < lower_bound) | (y > upper_bound)


def train_robust_regression(X_train, y_train, epsilon=1.35):
    """
    Train Huber Regression: less sensitive to outliers than Linear Regression.

    Combines MSE loss on small residuals with MAE loss on large residuals.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    epsilon : float
        Threshold for switching between MSE and MAE (default: 1.35).

    Returns
    -------
    HuberRegressor
        Fitted model.
    """
    model = HuberRegressor(epsilon=epsilon, max_iter=1000)
    model.fit(X_train, y_train)
    return model


# ============================================================================
# 3. FEATURE SCALING AND STANDARDIZATION
# ============================================================================

def compare_scaling_strategies(X, y):
    """
    Compare different scaling strategies and their effects on model performance.

    Strategies: No scaling, StandardScaler, MinMaxScaler, RobustScaler.
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    from sklearn.model_selection import cross_val_score

    scalers = {
        "No Scaling": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }

    print("\nEffect of Different Scaling Strategies:")
    print("=" * 60)

    results = {}
    for name, scaler in scalers.items():
        if scaler is None:
            model = LinearRegression()
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        else:
            pipeline = Pipeline(
                [("scaler", scaler), ("model", LinearRegression())])
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

        results[name] = cv_scores.mean()
        print(
            f"{name:20s} | Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("=" * 60)
    best_strategy = max(results, key=results.get)
    print(f"Best strategy: {best_strategy}\n")

    return results


# ============================================================================
# 4. MODEL SELECTION: LINEAR VS RIDGE VS LASSO
# ============================================================================

def compare_regularization_models(X_train, X_test, y_train, y_test):
    """
    Compare Linear Regression with Ridge and Lasso on the same data.

    Helps decide which regularization (if any) is beneficial.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use cross-validation to select best alpha for Ridge and Lasso
    ridge_cv = RidgeCV(alphas=np.logspace(-2, 3, 100), cv=5)
    ridge_cv.fit(X_train_scaled, y_train)

    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, max_iter=5000)
    lasso_cv.fit(X_train_scaled, y_train)

    # Linear Regression (no regularization)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Evaluate all three
    models = {
        "Linear Regression": lr,
        f"Ridge (α={ridge_cv.alpha_:.4f})": ridge_cv,
        f"Lasso (α={lasso_cv.alpha_:.4f})": lasso_cv,
    }

    print("\nModel Comparison (with Optimal Regularization):")
    print("=" * 70)
    print(f"{'Model':<30} | {'Train R²':>10} | {'Test R²':>10} | {'Test RMSE':>10}")
    print("-" * 70)

    results = {}
    for name, model in models.items():
        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)
        test_rmse = np.sqrt(mean_squared_error(
            y_test, model.predict(X_test_scaled)))

        results[name] = {"train_r2": train_r2,
                         "test_r2": test_r2, "rmse": test_rmse}
        print(f"{name:<30} | {train_r2:>10.4f} | {test_r2:>10.4f} | {test_rmse:>10.4f}")

    print("=" * 70)
    print("\nInterpretation:")
    print("  - Train R² much higher than Test R²: Overfitting (use regularization)")
    print("  - Ridge vs Lasso: Lasso provides feature selection (sparse coefficients)")
    print()

    return ridge_cv, lasso_cv, lr


# ============================================================================
# 5. LEARNING CURVES: BIAS-VARIANCE TRADEOFF
# ============================================================================

def plot_learning_curve_analysis(X, y, model_name="Linear Regression"):
    """
    Analyze learning curves to diagnose high bias vs high variance.

    High bias (underfitting): train and test curves are both low and flat
    High variance (overfitting): large gap between train and test curves
    """
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt

    model = LinearRegression()

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="r2",
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    print(f"\nLearning Curve Analysis for {model_name}:")
    print("=" * 60)
    print(f"{'Train Size':<15} | {'Train R² (std)':>20} | {'Test R² (std)':>20}")
    print("-" * 60)

    # Every other point for brevity
    for i, size in enumerate(train_sizes[::2]):
        print(f"{int(size):<15} | {train_mean[i*2]:.4f} (±{train_std[i*2]:.4f}) | "
              f"{test_mean[i*2]:.4f} (±{test_std[i*2]:.4f})")

    print("=" * 60)

    gap = train_mean[-1] - test_mean[-1]
    if gap > 0.2:
        diagnosis = "HIGH VARIANCE - Model is overfitting. Use regularization."
    elif train_mean[-1] < 0.3:
        diagnosis = "HIGH BIAS - Model is underfitting. Use more features or complex model."
    else:
        diagnosis = "BALANCED - Model generalization is reasonable."

    print(f"\nFinal train-test gap: {gap:.4f}")
    print(f"Diagnosis: {diagnosis}\n")


# ============================================================================
# 6. RESIDUAL DIAGNOSTICS
# ============================================================================

def check_residual_assumptions(y_test, y_pred):
    """
    Check if residuals satisfy Linear Regression assumptions.
    """
    residuals = y_test - y_pred

    print("\nResidual Diagnostics:")
    print("=" * 60)

    # 1. Mean close to zero
    residual_mean = np.mean(residuals)
    print(f"1. Mean of residuals: {residual_mean:.6f}")
    print(f"   ✓ OK (should be ≈ 0)" if abs(
        residual_mean) < 1e-10 else "   ✗ WARNING")

    # 2. Homoscedasticity (constant variance)
    lower_half = residuals[y_pred < np.median(y_pred)]
    upper_half = residuals[y_pred >= np.median(y_pred)]
    lower_std = np.std(lower_half)
    upper_std = np.std(upper_half)
    var_ratio = max(lower_std, upper_std) / min(lower_std, upper_std)

    print(f"\n2. Homoscedasticity check (variance ratio): {var_ratio:.4f}")
    print(f"   ✓ OK (ratio should be close to 1)" if var_ratio <
          2 else "   ✗ WARNING - heteroscedasticity")

    # 3. Normality (Shapiro-Wilk test on sample)
    from scipy import stats
    sample_residuals = residuals[::max(
        1, len(residuals)//500)]  # Sample if too many
    stat, p_value = stats.shapiro(sample_residuals)

    print(f"\n3. Normality (Shapiro-Wilk p-value): {p_value:.4f}")
    print(f"   ✓ OK (p > 0.05)" if p_value >
          0.05 else f"   ✗ WARNING - residuals not normal")

    # 4. Outliers
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
    outlier_count = np.sum(z_scores > 3)
    outlier_pct = 100 * outlier_count / len(residuals)

    print(
        f"\n4. Outliers (|z-score| > 3): {outlier_count} ({outlier_pct:.2f}%)")
    print(f"   ✓ OK (<5% typical)" if outlier_pct <
          5 else "   ✗ WARNING - many outliers")

    print("=" * 60 + "\n")


# ============================================================================
# 7. PRACTICAL WORKFLOW
# ============================================================================

def comprehensive_workflow():
    """
    Demonstrate complete workflow with best practices.
    """
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION: COMPREHENSIVE BEST PRACTICES WORKFLOW")
    print("=" * 70)

    # Generate synthetic data with some challenges
    print("\n[1] Generating synthetic regression data with challenges...")
    X, y = make_regression(
        n_samples=300,
        n_features=20,
        n_informative=10,
        noise=20,
        random_state=42
    )

    # Add some correlation (multicollinearity)
    X_df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    X_df["Feature_20"] = X_df["Feature_0"] + np.random.randn(len(X)) * 2

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    # 1. Check multicollinearity
    print("\n[2] Multicollinearity Check:")
    print("-" * 70)
    multicollinear_pairs = detect_multicollinearity(X_train, threshold=0.7)
    if len(multicollinear_pairs) > 0:
        print(
            f"Found {len(multicollinear_pairs)} highly correlated feature pairs:")
        print(multicollinear_pairs.head().to_string(index=False))
    else:
        print("No high multicollinearity detected.")

    # 2. Outlier detection
    print("\n[3] Outlier Detection:")
    print("-" * 70)
    outliers_zscore = detect_outliers_zscore(y_train, threshold=2.5)
    outliers_iqr = detect_outliers_iqr(y_train)
    print(f"Z-score method: {np.sum(outliers_zscore)} outliers detected")
    print(f"IQR method: {np.sum(outliers_iqr)} outliers detected")

    # 3. Compare scaling strategies
    print("\n[4] Feature Scaling Strategy:")
    print("-" * 70)
    compare_scaling_strategies(X_train, y_train)

    # 4. Model selection
    print("[5] Model Selection (Linear vs Ridge vs Lasso):")
    print("-" * 70)
    ridge, lasso, lr = compare_regularization_models(
        X_train, X_test, y_train, y_test)

    # 5. Residual diagnostics
    print("[6] Residual Analysis:")
    print("-" * 70)
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_train)
    X_test_pred = scaler.transform(X_test)
    ridge.fit(X_test_scaled, y_train)
    y_pred = ridge.predict(X_test_pred)
    check_residual_assumptions(y_test, y_pred)

    print("=" * 70)
    print("WORKFLOW COMPLETE\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION: BEST PRACTICES & REAL-WORLD CONSIDERATIONS")
    print("=" * 70)

    # Run comprehensive workflow
    comprehensive_workflow()

    # Summary
    print("\nKEY TAKEAWAYS:")
    print("-" * 70)
    print("1. Always check for multicollinearity (VIF or correlation matrix)")
    print("2. Detect and handle outliers appropriately")
    print("3. Compare scaling strategies for your data")
    print("4. Use cross-validation to select regularization (Ridge/Lasso)")
    print("5. Analyze learning curves to diagnose bias vs variance")
    print("6. Always validate residual assumptions")
    print("7. Robust regression is useful when outliers are present")
    print("8. Feature scaling is critical for interpretability and regularization")
    print("-" * 70 + "\n")
