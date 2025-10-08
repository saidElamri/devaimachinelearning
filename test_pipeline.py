# test_pipeline.py
import pandas as pd
from pipeline import preprocess_and_split
import numpy as np

def test_pipeline():
    # Small fake dataset
    # NEW (big enough for stratify)
    data = pd.DataFrame({
    'age': [25, 30, 35, np.nan],
    'income': [50000, 60000, 70000, 80000],
    'gender': ['male', 'female', 'male', None],
    'plan_type': ['basic', 'premium', 'basic', 'premium'],
    'Churn': [0, 1, 0, 1]
})

    # Run preprocessing and split
    out = preprocess_and_split(data, target_col='Churn', test_size=0.5, random_state=42)

    # Extract processed data and pipeline
    X_train_proc = out['X_train_proc']
    X_test_proc = out['X_test_proc']
    y_train = out['y_train']
    y_test = out['y_test']
    pipeline = out['pipeline']

    # Basic checks
    assert not X_train_proc.isna().any().any(), "X_train still has NaNs"
    assert not X_test_proc.isna().any().any(), "X_test still has NaNs"
    assert X_train_proc.shape[0] == y_train.shape[0], "X_train and y_train shape mismatch"
    assert X_test_proc.shape[0] == y_test.shape[0], "X_test and y_test shape mismatch"

    print(" Pipeline test passed!")
    print("X_train processed shape:", X_train_proc.shape)
    print("X_test processed shape:", X_test_proc.shape)
    print("y_train shape:", y_train.shape)

if __name__ == "__main__":
    test_pipeline()
