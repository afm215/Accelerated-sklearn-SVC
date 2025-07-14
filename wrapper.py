from sklearn.svm import SVC
from kernels.utils import compute_kernel_matrix, compute_kernel_matrix_cross

class SVCWrapper:

    def __init__(self, kernel="linear", random_state=42, class_weight = None):
        self.model = SVC(kernel="precomputed", random_state=random_state, max_iter=10000, class_weight = class_weight)
        self.kernel = kernel
        self.gamma = None
        self.covariance_matrix_inv = None

    

    def predict(self, X_infer):
        if not hasattr(self, 'X_train'):
            raise ValueError("Model has not been fitted yet. Call 'fit' with training data first.")

        kernel_matrix = compute_kernel_matrix_cross(X_infer, self.X_train, gamma=self.gamma, kernel=self.kernel, covariance_inv=self.covariance_matrix_inv)
        return self.model.predict(kernel_matrix)
        

    def fit(self, X, y, log = False):
        self.X_train = X
        if log:
            print(f"Computing SVC kernel matrix {self.kernel} kernel...")
        if self.kernel == "mahalanobis":
            kernel_matrix, gamma, covariance_matrix_inv = compute_kernel_matrix(X, gamma="scale", kernel=self.kernel)
            if log:
                print(f"Kernel matrix computed with shape {kernel_matrix.shape} and gamma={gamma}.")
            self.covariance_matrix_inv = covariance_matrix_inv
        else:
            kernel_matrix, gamma = compute_kernel_matrix(X, kernel=self.kernel)
            if log:
                print(f"Kernel matrix computed with shape {kernel_matrix.shape} and gamma={gamma}.")
        
        self.gamma = gamma
        self.model = self.model.fit(kernel_matrix, y)
        if log:
            print("SVC model fitted successfully.")

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)