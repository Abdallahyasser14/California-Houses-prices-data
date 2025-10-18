import numpy as np

class linear_regression_gradient_descent:
    def __init__(self,iterations=20,learning_rate=0.1) :
   
        self.iterations=iterations
        self.parameters=0
        self.learning_rate=learning_rate
    
    def fit(self,X,y):
        X=X.copy()
        X["bias"] = 1
        self.parameters=np.zeros((X.shape[1],1))
        x_matrix=X.to_numpy()
        y_matrix=y.to_numpy().reshape(-1,1)

        # parameters => (d+1,1)
        # x=> (n,d+1)
        # y_pred =>(n,1)  =x*parameters

        for iteration in range(self.iterations):
            y_pred=x_matrix @  self.parameters
            error = y_pred - y_matrix  # (n,1)
            # x (n,d+1) (n,1)   => (d+1,1)
            loss = (1 / (2 * X.shape[0])) * np.sum(error**2)
            self.parameters-= self.learning_rate* (1 / X.shape[0]) * (x_matrix.T @ error)

            print(f"Epoch {iteration+1}: Loss = {loss}")

    def predict(self,X) :
         X=X.copy()
         X["bias"] = 1
         x_matrix=X.to_numpy()
         return x_matrix @  self.parameters



         # => [1,2,3,4] n (n,1)


            
            # [b1,w1,w2,...,wn]^T
