"""
test_feature_types.py
---------------------
Validation script to test feature type definitions and validation functions.
"""

import sys
import pandas as pd
import numpy as np

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    ALL_FEATURES,
    TARGET_COLUMN,
)
from src.data_preprocessing import (
    validate_feature_separation,
    print_feature_summary,
)


def test_config_validation():
    """Test that config assertions pass."""
    print("Testing config validation...")
    
    # Test 1: Target not in features
    assert TARGET_COLUMN not in ALL_FEATURES, "FAIL: Target in features"
    print("✓ Target column not in features")
    
    # Test 2: No overlap between numerical and categorical
    overlap = set(NUMERICAL_FEATURES) & set(CATEGORICAL_FEATURES)
    assert len(overlap) == 0, f"FAIL: Overlap found: {overlap}"
    print("✓ No overlap between numerical and categorical")
    
    # Test 3: Excluded columns not in features
    for col in EXCLUDED_COLUMNS:
        assert col not in ALL_FEATURES, f"FAIL: {col} in features"
    print("✓ Excluded columns not in features")
    
    # Test 4: Feature counts
    print(f"\nFeature counts:")
    print(f"  Numerical: {len(NUMERICAL_FEATURES)}")
    print(f"  Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"  Total: {len(ALL_FEATURES)}")
    print(f"  Excluded: {len(EXCLUDED_COLUMNS)}")
    
    print("\n✓ All config validations passed!\n")


def test_feature_validation():
    """Test feature validation functions with mock data."""
    print("Testing feature validation functions...")
    
    # Create mock data
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'patient_id': [f'P{i:05d}' for i in range(n_rows)],
        'age': np.random.uniform(20, 80, n_rows),
        'length_of_stay': np.random.uniform(1, 15, n_rows),
        'num_procedures': np.random.randint(0, 10, n_rows),
        'num_medications': np.random.randint(0, 20, n_rows),
        'num_diagnoses': np.random.randint(0, 10, n_rows),
        'department': np.random.choice(['Emergency', 'ICU', 'General Medicine'], n_rows),
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'admission_type': np.random.choice(['Emergency', 'Elective'], n_rows),
        'bed_type': np.random.choice(['General', 'ICU'], n_rows),
        'readmitted': np.random.randint(0, 2, n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Separate X and y
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Test validation (should pass)
    try:
        validate_feature_separation(X, y)
        print("✓ Feature separation validation passed")
    except ValueError as e:
        print(f"✗ Feature separation validation failed: {e}")
        return False
    
    # Drop ID columns
    X_clean = X.drop(columns=['patient_id'])
    
    # Print summary
    print("\nFeature summary:")
    print_feature_summary(X_clean, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    
    # Test with target in features (should fail)
    print("Testing error detection...")
    X_bad = df.copy()  # includes target
    try:
        validate_feature_separation(X_bad, y)
        print("✗ Should have detected target in features")
        return False
    except ValueError as e:
        print(f"✓ Correctly detected error: {str(e)[:60]}...")
    
    print("\n✓ All feature validation tests passed!\n")
    return True


def main():
    print("="*60)
    print("FEATURE TYPE DEFINITION VALIDATION")
    print("="*60 + "\n")
    
    test_config_validation()
    test_feature_validation()
    
    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    main()
