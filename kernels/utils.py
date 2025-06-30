import torch
from mahalanobis import compute_self_mahalanobis_distances_batched, compute_cross_mahalanobis_distances_batched

def compute_self_mahalanobis_distances(X):
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            ### convert numpy array to torch tensor
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to("cuda" if torch.cuda.is_available() else "cpu")
        deltas = X.unsqueeze(1) - X.unsqueeze(0)
        covariance_matrix = torch.cov(X.T)
        covariance_matrix_inv = torch.linalg.inv(covariance_matrix)
        cross_distances = torch.matmul(deltas, torch.matmul(covariance_matrix_inv, deltas.transpose(1, 2))).cpu()
    return cross_distances, covariance_matrix_inv

def compute_cross_mahalanobis_distances(X1, X2, cov_inv):
    with torch.no_grad():
        if not isinstance(X1, torch.Tensor):
            ### convert numpy array to torch tensor
            X1 = torch.tensor(X1, dtype=torch.float32)
        if not isinstance(X2, torch.Tensor):
            ### convert numpy array to torch tensor
            X2 = torch.tensor(X2, dtype=torch.float32)
        X1 = X1.to("cuda" if torch.cuda.is_available() else "cpu")
        X2 = X2.to("cuda" if torch.cuda.is_available() else "cpu")
        deltas = X1.unsqueeze(1) - X2.unsqueeze(0)
        print("delta shape is ", deltas.shape)
        cross_distances = torch.matmul(deltas, torch.matmul(cov_inv, deltas.transpose(1, 2))).cpu()
    return cross_distances

def compute_kernel_matrix(X, gamma="scale", kernel="rbf"):
    """
    Args:
        X (torch.Tensor): A 2D tensor of shape (n_samples, n_features).
        gamma (float): The gamma parameter for the RBF kernel.
        kernel (str): The type of kernel to compute. Currently only "rbf" and "linear" is supported.
    Returns:
        torch.Tensor: A 2D tensor of shape (n_samples, n_samples)
                      containing the RBF kernel values.
    """
    import gc
    ## cleaning cuda memory
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            ### convert numpy array to torch tensor
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to("cuda" if torch.cuda.is_available() else "cpu")
        if gamma == "scale":
            X_var = torch.var(X).item()
            gamma = 1.0 / (X.shape[1] * X_var)
        elif gamma == "auto":
            gamma = 1.0 / X.shape[1]
        else:
            assert isinstance(gamma,  float), "Gamma must be a number or 'scale'/'auto'."
        
        print(f"\nStarting RBF kernel matrix computation on {X.device} with gamma={gamma}...")
        assert kernel in ["rbf", "linear", "mahalanobis"], "Kernel must be either 'rbf' or 'linear'."
        
        sq_norms = torch.sum(X**2, dim=1, keepdim=True).cpu()

        # assert torch.all(torch.abs(sq_norms - 1) < 1e-6), "Input features must be L2 normalized."
        if kernel == "mahalanobis":
            print("Using Mahalanobis kernel, computing self distances...")
            ### correct gamma for mahalanobis kernel
            gamma =  1 # gamma * X_var
            sq_distances, covariance_matrix_inv = compute_self_mahalanobis_distances_batched(X) #compute_self_mahalanobis_distances(X)
            kernel_matrix = torch.exp(-gamma * sq_distances)
            print(f"Kernel matrix computation finished. Shape: {kernel_matrix.shape}")
            return kernel_matrix.numpy(), gamma, covariance_matrix_inv
        
        
        dot_products = torch.matmul(X, X.T).cpu()
        if kernel == "linear":
            print("Using linear kernel, returning dot product matrix.")
            return dot_products, None
        
        
        sq_distances = sq_norms + sq_norms.T - 2* dot_products
        # sq_distances_np = np.clip(sq_distances, a_min=0, a_max=None)
        sq_distances_np = torch.clamp(sq_distances, min=0)

        kernel_matrix = torch.exp(-gamma * sq_distances_np)

        print(f"Kernel matrix computation finished. Shape: {kernel_matrix.shape}")
        return kernel_matrix.numpy(), gamma
def compute_kernel_matrix_cross(X1, X2, gamma=None,covariance_inv = None,  kernel="linear"):
    """
    Computes the RBF kernel matrix between rows of two tensors X1 and X2 on GPU.
    K(x1_i, x2_j) = exp(-gamma * ||x1_i - x2_j||^2)

    Args:
        X1 (torch.Tensor): A 2D tensor of shape (n_samples1, n_features).
        X2 (torch.Tensor): A 2D tensor of shape (n_samples2, n_features).
        gamma (float): The gamma parameter for the RBF kernel.
    Returns:
        torch.Tensor: A 2D tensor of shape (n_samples1, n_samples2)
                      containing the RBF kernel values.
    """

    import gc
    ## cleaning cuda memory
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        if not isinstance(X1, torch.Tensor):
            ### convert numpy array to torch tensor
            X1 = torch.tensor(X1, dtype=torch.float32)
        
        if not isinstance(X2, torch.Tensor):
            ### convert numpy array to torch tensor
            X2 = torch.tensor(X2, dtype=torch.float32)

        X1 = X1.to("cuda" if torch.cuda.is_available() else "cpu")
        X2 = X2.to("cuda" if torch.cuda.is_available() else "cpu")

        sq_norms1 = torch.sum(X1**2, dim=1, keepdim=True).cpu() # (n_samples1, 1)
        sq_norms2 = torch.sum(X2**2, dim=1, keepdim=False).cpu() # (n_samples2,) -> (1, n_samples2) after broadcasting for addition

        dot_products = torch.matmul(X1, X2.T).cpu() # (n_samples1, n_samples2)
        if kernel == "mahalanobis":
            print("Using Mahalanobis kernel, computing cross distances...")
            if covariance_inv is None:
                raise ValueError("Covariance matrix must be provided for Mahalanobis kernel.")
            if gamma is None:
                raise ValueError("Gamma must be provided for Mahalanobis kernel.")
            sq_distances = compute_cross_mahalanobis_distances_batched(X1, X2, covariance_inv) #compute_cross_mahalanobis_distances(X1, X2, covariance_inv)
        if kernel == "linear":
            print("Using linear kernel, returning dot product matrix.")
            return dot_products.numpy()
        assert gamma is not None, "Gamma must be provided for RBF kernel."

        # squared Euclidean distances: ||x1_i - x2_j||^2 = ||x1_i||^2 + ||x2_j||^2 - 2 * x1_i.T * x2_j
        sq_distances = (sq_norms1 + sq_norms2) - 2 * dot_products
        sq_distances = torch.clamp(sq_distances, min=0)

        kernel_matrix = torch.exp(-gamma * sq_distances)
        print(f"Cross-kernel matrix computation finished. Shape: {kernel_matrix.shape}")
        return kernel_matrix.numpy()