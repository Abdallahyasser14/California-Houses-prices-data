import numpy as np

def mean_absolute_error_2(y_true, y_pred):   
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error_2(y_true , y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    return np.mean((y_true - y_pred)**2)
