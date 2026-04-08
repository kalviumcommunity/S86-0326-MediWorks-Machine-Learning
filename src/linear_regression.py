"""
linear_regression.py
--------------------
Linear Regression model training and utilities for regression tasks.

Purpose
-------
Implement Linear Regression following best practices from the lesson:
- Train models on preprocessed data
- Support feature scaling via pipelines
- Compare against baseline (mean predictor)
- Provide coefficient interpretation
- Use proper train/test split protocol

Design Notes
------------
- LinearRegression itself doesn't require scaling for predictions, but scaling
  is essential for:
  1. Gradient descent optimization (if used)
  2. Coefficient magnitude comparison
  3. Regularized variants (Ridge, Lasso)
  4. Numerical stability
  5. Preventing subtle data leakage
  
- All training functions follow the same pattern: accept preprocessed data,
  optional hyperparameters, return fitted model. No printing inside functions.
  
- Pipelines ensure the scaler is fit ONLY on training data and applied
  consistently to test data, preventing data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

from src.config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Linear Regression Training
# ---------------------------------------------------------------------------

def train_linear_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> LinearRegression:
    """
    Fit a LinearRegression model on training data.

    Implements the standard approach: fit model using closed-form solution
    (Normal Equation via efficient LAPACK solver). No hyperparameters to tune;
    the solver automatically computes optimal coefficients and intercept.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed feature matrix of shape (n_samples, n_features).
    y_train : pd.Series
        Continuous target variable of shape (n_samples,).
    random_state : int
        Random seed (included for API consistency; LinearRegression is deterministic).

    Returns
    -------
    LinearRegression
        Fitted model ready for prediction or evaluation.

    Raises
    ------
    ValueError
        If X_train and y_train have different numbers of rows.

    Examples
    --------
    >>> model = train_linear_regression(X_train, y_train)
    >>> y_pred = model.predict(X_test)

    Notes
    -----
    LinearRegression's closed-form solution minimizes Mean Squared Error (MSE):
        MSE = (1/n) * sum((y_pred - y_true)^2)

    The solution is scale-invariant for predictions but not for coefficient
    interpretation. Use a scaling pipeline for coefficient comparison.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} rows but "
            f"y_train has {len(y_train)} rows."
        )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def train_linear_regression_with_scaling(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Train Linear Regression with feature scaling in a Pipeline.

    Ensures scaler is fit ONLY on training data and same transformation
    is applied to test data. This prevents subtle but common data leakage.

    Use this when you need to compare coefficient magnitudes across features
    with different scales (e.g., age vs. income).

    Parameters
    ----------
    X_train : np.ndarray
        Raw or preprocessed feature matrix.
    y_train : pd.Series
        Continuous target variable.
    random_state : int
        Random seed (for API consistency).

    Returns
    -------
    Pipeline
        Fitted pipeline with StandardScaler and LinearRegression.
        Call pipeline.predict() exactly like a regular model.

    Examples
    --------
    >>> pipeline = train_linear_regression_with_scaling(X_train, y_train)
    >>> y_pred = pipeline.predict(X_test)  # Scaler is applied automatically
    >>> model = pipeline.named_steps["model"]  # Access the LR model for coefficients
    >>> print(model.coef_)
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} rows but "
            f"y_train has {len(y_train)} rows."
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_ridge_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    alpha: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Ridge:
    """
    Train Ridge Regression: Linear Regression with L2 regularization.

    Ridge adds a penalty term to the loss function that shrinks large
    coefficients. Useful when:
    - Multicollinearity is present (highly correlated features)
    - Model is overfitting
    - You want to stabilize coefficient estimates

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed feature matrix.
    y_train : pd.Series
        Continuous target variable.
    alpha : float
        Regularization strength (default: 1.0).
        - alpha=0: behaves like Linear Regression
        - Large alpha: coefficients shrink towards zero
        Tune via cross-validation.
    random_state : int
        Random seed (for API consistency).

    Returns
    -------
    Ridge
        Fitted Ridge regression model.

    Examples
    --------
    >>> model = train_ridge_regression(X_train, y_train, alpha=0.1)
    >>> y_pred = model.predict(X_test)
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} rows but "
            f"y_train has {len(y_train)} rows."
        )

    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)

    return model


def train_lasso_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    alpha: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Lasso:
    """
    Train Lasso Regression: Linear Regression with L1 regularization.

    Lasso (Least Absolute Shrinkage and Selection Operator) penalizes
    the absolute value of coefficients, often driving some to exactly zero.
    Useful for:
    - Automatic feature selection
    - Interpretability (sparse models)
    - High-dimensional datasets

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed feature matrix.
    y_train : pd.Series
        Continuous target variable.
    alpha : float
        Regularization strength (default: 1.0).
        - alpha=0: behaves like Linear Regression
        - Large alpha: more coefficients become zero
        Tune via cross-validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Lasso
        Fitted Lasso regression model.

    Examples
    --------
    >>> model = train_lasso_regression(X_train, y_train, alpha=0.01)
    >>> y_pred = model.predict(X_test)
    >>> # Count non-zero coefficients (feature selection)
    >>> selected_features = np.sum(model.coef_ != 0)
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} rows but "
            f"y_train has {len(y_train)} rows."
        )

    model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
    model.fit(X_train, y_train)

    return model


# ---------------------------------------------------------------------------
# Baseline Model (Mean Predictor)
# ---------------------------------------------------------------------------

def train_baseline_regressor(
    X_train: np.ndarray,
    y_train: pd.Series,
    strategy: str = "mean",
    random_state: int = RANDOM_STATE,
) -> DummyRegressor:
    """
    Fit a baseline DummyRegressor for regression tasks.

    The DummyRegressor provides a simple baseline using trivial strategies:
    - 'mean': Always predict the mean of training targets
    - 'median': Always predict the median
    - 'quantile': Predict a specific quantile
    - 'constant': Always predict a specified constant

    This establishes minimum performance that any regression model must exceed.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix (ignored by DummyRegressor).
    y_train : pd.Series
        Continuous target values.
    strategy : str
        Baseline strategy (default: 'mean').
    random_state : int
        Random seed (used by 'quantile' strategy).

    Returns
    -------
    DummyRegressor
        Fitted baseline model.

    Examples
    --------
    >>> baseline = train_baseline_regressor(X_train, y_train)
    >>> baseline_pred = baseline.predict(X_test)
    >>> baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} rows but "
            f"y_train has {len(y_train)} rows."
        )

    baseline = DummyRegressor(strategy=strategy)
    baseline.fit(X_train, y_train)

    return baseline


# ---------------------------------------------------------------------------
# Coefficient Interpretation
# ---------------------------------------------------------------------------

def get_coefficients_dataframe(
    model,
    feature_names: list,
) -> pd.DataFrame:
    """
    Extract and display model coefficients in an interpretable DataFrame.

    Parameters
    ----------
    model : LinearRegression, Ridge, or Lasso
        Fitted regression model.
    feature_names : list
        Names of features corresponding to coefficient positions.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['Feature', 'Coefficient'].
        Sorted by absolute coefficient value (most influential first).

    Examples
    --------
    >>> coef_df = get_coefficients_dataframe(model, X_train.columns.tolist())
    >>> print(coef_df)
           Feature  Coefficient
        0  Location        5.200
        1  Bedrooms        2.100
        2  Size            0.040
        3  Age          -0.800

    Interpretation Notes
    --------------------
    - Coefficient sign: positive = feature increases prediction, negative = decreases
    - Magnitude indicates strength of relationship
    - Only comparable across features if they are standardized first
    - Large multicollinearity can destabilize coefficients
    """
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)

    return coef_df


# ---------------------------------------------------------------------------
# Residual Analysis
# ---------------------------------------------------------------------------

def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute residuals (prediction errors).

    Residuals = y_true - y_pred

    Use residual plots to check assumptions:
    - vs. fitted values: check linearity and homoscedasticity
    - Q-Q plot: check normality
    - Scale-location plot: check homoscedasticity
    - Residuals vs. order: check independence

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    np.ndarray
        Residuals (errors).
    """
    return y_true - y_pred
