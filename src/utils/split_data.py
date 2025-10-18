import numpy as np
import pandas as pd

def split_train_val_test(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    
    train_end = int(m * train_size)
    val_end = train_end + int(m * val_size)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Use .iloc to keep pandas DataFrame/Series
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
