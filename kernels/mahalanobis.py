import torch
from tqdm import tqdm
def compute_self_mahalanobis_distances_batched(X, batch_size=32):

    with torch.no_grad():
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = X.to(device)
        N, D = X.shape

        cov = torch.cov(X.T) * 512
        eps = 1e-6
        cov += eps * torch.eye(D, device=device)
        cov_inv = torch.linalg.inv(cov)

        dist_matrix = torch.empty((N, N), dtype=torch.float32, device=device)

        for i in tqdm(range(0, N, batch_size),desc="Computing self-distances"):
            end_i = min(i + batch_size, N)
            X_i = X[i:end_i]  # shape: [B1, D]
            delta_i = X_i.unsqueeze(1) - X.unsqueeze(0)  # shape: [B1, N, D]
            mahal_sq = torch.einsum("bid,dd,bid->bi", delta_i, cov_inv, delta_i)  # [B1, N]
            dist_matrix[i:end_i] = mahal_sq

        dist_matrix = dist_matrix.cpu()
        print(torch.min(dist_matrix), torch.max(dist_matrix))
        dist_matrix = dist_matrix.clamp(min=0)

    return dist_matrix, cov_inv.cpu()

def compute_cross_mahalanobis_distances_batched(X1, X2, cov_inv, batch_size=32):
    import torch

    with torch.no_grad():
        # Convert to tensors if needed
        if not isinstance(X1, torch.Tensor):
            X1 = torch.tensor(X1, dtype=torch.float32)
        if not isinstance(X2, torch.Tensor):
            X2 = torch.tensor(X2, dtype=torch.float32)
        if not isinstance(cov_inv, torch.Tensor):
            cov_inv = torch.tensor(cov_inv, dtype=torch.float32)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        X1 = X1.to(device)
        X2 = X2.to(device)
        cov_inv = cov_inv.to(device)

        N1, D = X1.shape
        N2 = X2.shape[0]

        # Preallocate distance matrix
        dist_matrix = torch.empty((N1, N2), dtype=torch.float32, device=device)

        # Process X1 in batches
        for i in tqdm(range(0, N1, batch_size), desc="Computing distances"):
            end_i = min(i + batch_size, N1)
            X1_batch = X1[i:end_i]  # shape: [B1, D]
            deltas = X1_batch.unsqueeze(1) - X2.unsqueeze(0)  # [B1, N2, D]
            dists = torch.einsum("bid,dd,bid->bi", deltas, cov_inv, deltas)  # [B1, N2]
            dist_matrix[i:end_i] = dists

        dist_matrix = dist_matrix.cpu()
        print(torch.min(dist_matrix), torch.max(dist_matrix))
        dist_matrix = torch.clamp(dist_matrix, min=0.0)

    return dist_matrix