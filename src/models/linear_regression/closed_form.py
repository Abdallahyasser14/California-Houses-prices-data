import numpy as np
class linear_regression_closed_form:
    def __init__(self,lambda_param=0.0):
        self.lambda_param = lambda_param
    
    def fit(self,X,y):
        X=X.copy()
        X["bias"] = 1
        x_matrix=X.to_numpy()
        y_matrix=y.to_numpy().reshape(-1,1)
        x_matrix_transpose = x_matrix.T

        self.parameters= np.linalg.inv(x_matrix_transpose @ x_matrix + self.lambda_param * np.identity(x_matrix.shape[1])) @ (x_matrix_transpose @ y_matrix )
        # [b1,w1,w2,...,wn]^T

    def predict(self,X):
        X=X.copy()
        X["bias"] = 1
        return X.to_numpy() @ self.parameters


