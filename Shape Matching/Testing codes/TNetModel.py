import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. T-Net Architecture (PyTorch Module) ---
class TNet(nn.Module):
    """
    PointNet's T-Net (Transformation Network) for learning 3x3 rotation matrices.
    It takes a point cloud as input and outputs 6 numbers that are then
    orthogonalized to form an SO(3) rotation matrix.
    """
    def __init__(self, k=3):
        """
        Initializes the T-Net.
        Args:
            k (int): The dimension of the transformation matrix (e.g., 3 for 3x3 spatial transform).
                     The output will be k*k, but we predict 2 vectors of size k.
        """
        super(TNet, self).__init__()
        self.k = k

        # Shared MLPs (Multi-Layer Perceptrons)
        # These layers process each point independently.
        self.mlp1 = nn.Sequential(
            nn.Conv1d(k, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

        # Fully Connected Layers after Max Pooling
        # These layers process the global feature vector to predict the transformation.
        # Output is 2*k (6 for 3x3 matrix, representing two 3D vectors)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2 * k), # Output 6 numbers for 3x3 matrix (two 3D vectors)
        )

    def forward(self, x):
        """
        Forward pass of the T-Net.
        Args:
            x (torch.Tensor): Input point cloud tensor of shape (B, N, k),
                              where B is batch size, N is number of points, k is dimension (e.g., 3 for XYZ).
        Returns:
            torch.Tensor: A tensor of shape (B, 2*k) representing the raw output
                          before orthogonalization.
        """
        # Permute input to (B, k, N) for Conv1d
        x = x.permute(0, 2, 1)

        # Apply shared MLPs
        x = self.mlp1(x) # Output shape: (B, 1024, N)

        # Max pooling across the N points to get a global feature vector
        x = torch.max(x, 2, keepdim=True)[0] # Output shape: (B, 1024, 1)
        x = x.view(-1, 2048) # Reshape to (B, 1024)

        # Apply fully connected layers to predict the 6 numbers
        x = self.fc(x) # Output shape: (B, 2*k)

        return x

# --- 2. Differentiable Gram-Schmidt Orthogonalization ---
def gram_schmidt_orthogonalization(vectors_6d):
    """
    Performs differentiable Gram-Schmidt orthogonalization on a batch of 6D vectors
    to produce a batch of 3x3 SO(3) rotation matrices.

    Args:
        vectors_6d (torch.Tensor): A tensor of shape (B, 6), where B is batch size.
                                   Each 6D vector represents two 3D vectors [v1_x, v1_y, v1_z, v2_x, v2_y, v2_z].

    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) representing the batch of
                      SO(3) rotation matrices.
    """
    batch_size = vectors_6d.shape[0]

    # Reshape the 6D output into two 3D vectors for each item in the batch
    v1 = vectors_6d[:, :3] # (B, 3)
    v2 = vectors_6d[:, 3:] # (B, 3)

    # Step 1: Normalize v1 to get u1
    # Add a small epsilon for numerical stability to avoid division by zero if magnitude is 0
    u1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)

    # Step 2: Project v2 onto u1, then subtract from v2 to get u2_prime
    # dot product: (v2 * u1).sum(dim=1, keepdim=True) -> (B, 1)
    proj_u1_v2 = (u1 * v2).sum(dim=1, keepdim=True) * u1 # (B, 3)
    u2_prime = v2 - proj_u1_v2 # (B, 3)

    # Step 3: Normalize u2_prime to get u2
    # Handle cases where u2_prime might be very small or zero (e.g., v1 and v2 are collinear)
    # In such cases, we need to find an orthogonal vector.
    # A robust way is to use cross product with a default vector, then re-orthogonalize.
    # For training, the network should learn to avoid degenerate cases.
    u2 = u2_prime / (torch.norm(u2_prime, dim=1, keepdim=True) + 1e-8)

    # Step 4: Calculate u3 as the cross product of u1 and u2
    u3 = torch.cross(u1, u2, dim=1) # (B, 3)
    u3 = u3 / (torch.norm(u3, dim=1, keepdim=True) + 1e-8) # Re-normalize for numerical stability

    # Construct the rotation matrix by stacking u1, u2, u3 as columns
    # R = [u1.T, u2.T, u3.T]
    # u1, u2, u3 are (B, 3). Stack them along a new dimension, then permute.
    rotation_matrix = torch.stack([u1, u2, u3], dim=2) # (B, 3, 3)

    return rotation_matrix

# --- 3. Geodesic Angular Distance Loss ---
def geodesic_loss(R_pred, R_true, L2=True, reduction='mean'):
    """
    Calculates the geodesic angular distance between two batches of rotation matrices.
    The formula is: theta = arccos((trace(R_pred^T * R_true) - 1) / 2)
    Args:
        R_pred (torch.Tensor): Predicted rotation matrices (B, 3, 3).
        R_true (torch.Tensor): Ground truth rotation matrices (B, 3, 3).
    Returns:
        torch.Tensor: Mean geodesic angular distance in radians.
    """
    # R_pred_T * R_true
    R_diff = torch.matmul(R_pred.transpose(1, 2), R_true) # (B, 3, 3)

    # Trace of R_diff
    # Sum diagonal elements: R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2] # (B,)

    # Clamp the argument to arccos to avoid NaN due to floating point errors
    # The argument should be between -1 and 3 for rotation matrices, but numerical
    # inaccuracies can push it slightly outside [-1, 3].
    # For arccos((trace - 1)/2), the argument should be in [-1, 1].
    # trace is in [-1, 3] for valid rotation matrices.
    # (trace - 1) / 2 is in [-1, 1].
    arg = torch.clamp((trace - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7) # Add epsilon for robustness

    # Calculate angle in radians
    angle = torch.acos(arg) # (B,)
    if(L2):
        angle = angle*angle
    if(reduction != 'mean'):
        return angle
    # Return mean angle across the batch
    return torch.mean(angle)

def trace_loss(R_pred, R_true):
    R_diff = torch.matmul(R_pred.transpose(1, 2), R_true)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    return -torch.mean(trace)
def point_alignment_loss(P_aligned, P_canonical):
    # P_aligned, P_canonical: (B, N, 3)
    return torch.mean(torch.norm(P_aligned - P_canonical, dim=2))  # scalar
