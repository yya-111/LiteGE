import torch
from pytorch3d.ops import knn_points


import torch


def addnoise_pointcloud(sampled_pts: torch.Tensor, noise_std = 0.01):
    #noise_std = 0.01  # Adjust as needed
    sampled_pts = sampled_pts + torch.randn_like(sampled_pts) * noise_std
    return sampled_pts

def slice_point_cloud_middle_tensor(pc: torch.Tensor, keep_ratio: float = 0.85) -> torch.Tensor:
    """
    Slices a point cloud tensor along a random axis to keep the middle region
    containing `keep_ratio` of points.

    Args:
        pc (torch.Tensor): Input point cloud of shape (N, 3)
        keep_ratio (float): Fraction of points to retain (e.g., 0.80 to 0.85)

    Returns:
        torch.Tensor: Sliced point cloud tensor of shape (~N * keep_ratio, 3)
    """
    assert pc.ndim == 2 and pc.shape[1] == 3, "Input tensor must be of shape (N, 3)"
    axis = torch.randint(0, 3, (1,)).item()

    axis_coords = pc[:, axis]

    lower_p = (1 - keep_ratio) / 2
    upper_p = (1 + keep_ratio) / 2

    lower_thresh = torch.quantile(axis_coords, lower_p)
    upper_thresh = torch.quantile(axis_coords, upper_p)

    mask = (axis_coords >= lower_thresh) & (axis_coords <= upper_thresh)
    return pc[mask]


def sample_mesh(verts, faces, sample_count:int=8000):
    with torch.no_grad():
        # --- 1. Area-weighted surface sampling
        v0, v1, v2 = verts[faces].unbind(1)  # (F, 3) each
        face_areas = 0.5 * (v1 - v0).cross(v2 - v0, dim=1).norm(dim=1)  # (F,)
        sum_face_areas = face_areas.sum()
        probs = face_areas / sum_face_areas
        face_idx = torch.multinomial(probs, sample_count, replacement=True)

        device = verts.device
        dtype = verts.dtype
        u = torch.rand(sample_count, 1, device=device, dtype=dtype)
        v = torch.rand(sample_count, 1, device=device, dtype=dtype)

        eps = 1e-6
        mask = (u + v) > (1 - eps)
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - (u + v)

        sampled_pts = (
            w * v0[face_idx] +
            u * v1[face_idx] +
            v * v2[face_idx]
        )  # (N, 3)
        return sampled_pts.float()



def preprocess_pointcloud(
    verts_ori:torch.Tensor,
    sampled_pts: torch.Tensor, 
    prints=False
    #faces: torch.Tensor, 
    #sample_count: int = 8000
) -> torch.Tensor:
    """
    Preprocess a mesh by sampling surface points and applying PCA-based orientation,
    unit-area scaling, and mesh-bounding-box centering.

    Steps:
    
    2. Compute PCA on sampled points in PyTorch to find principal axes.
    3. Enforce consistent axis signs so anatomical extremes lie in + direction.
    4. Align both mesh and point cloud to principal axes.
    5. Scale both to 1.7 surface area bounding box.

    Returns:
        torch.Tensor: Sampled point cloud of shape (sample_count, 3),
                      rotated, scaled, and centered.
    """
    with torch.no_grad():
        
        sample_count = sampled_pts.shape[0]
        # --- 2. PCA orientation (based on sampled points)
        mean_pt = sampled_pts.mean(dim=0)
        centered = sampled_pts - mean_pt
        #if(prints):
        #    print(centered[:10])
        cov = centered.T @ centered / (sample_count - 1)
        #if(prints):
        #    print(cov)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        for i in range(3):  # for each eigenvector
            if eigvecs[0, i] < 0:
                eigvecs[:, i] *= -1

        #if(prints):
        #    print(eigvals, eigvecs)
        order = torch.argsort(eigvals, descending=True)
        axes = eigvecs[:, order]  # (3, 3)
        #if(prints):
        #    print(axes)
        # --- 3. Enforce consistent axis signs
        proj = centered @ axes
        #if(prints):
        #    print(proj[:10])
        for i in range(2):
            if torch.median(proj[:, i]) < 0:
                axes[:, i] *= -1
                #print(-1)
                #print(torch.median(proj[:, i]))
                #if i < 2:
                #    axes[:, 2] = torch.cross(axes[:, 0], axes[:, 1], dim=0)
            #else:
                #print(1)
                #print(torch.median(proj[:, i]))
        axes[:, 2] = torch.cross(axes[:, 0], axes[:, 1], dim=0)
        R = axes.T  # Rotation matrix (3,3)

        # --- 4. Rotate, center and scale mesh and point cloud
        #verts_aligned = (R @ (verts - mean_pt).T).T
        verts_ori = (R @ (verts_ori - mean_pt).T).T
        sampled_pts = (R @ (sampled_pts - mean_pt).T).T
        #print(sampled_pts[:10])

        #scale_init = 1.0 / torch.sqrt(sum_face_areas)

        min_pt, _ = sampled_pts.min(dim=0)
        max_pt, _ = sampled_pts.max(dim=0)
        dims = max_pt - min_pt
        area = 2 * (dims[0]*dims[1] + dims[1]*dims[2] + dims[2]*dims[0])
        scale = 1.7 / torch.sqrt(area)
        

        #verts_aligned *= scale
        sampled_pts *= scale
        verts_ori *= scale
        #print(sampled_pts[:10])

        

        
    return sampled_pts, verts_ori, scale.item()#, verts_aligned, verts_ori, scale/scale_init

def toprint(toprint, text):
    if(toprint):
        print(text)
def preprocess_pointcloud_batch(sampled_pts: torch.Tensor) -> torch.Tensor:
    """
    Preprocess a batch of point clouds (B, N, 3) using PCA orientation,
    consistent axis flipping, and scaling to fit 1.7 surface area bounding box.

    Args:
        sampled_pts (torch.Tensor): (B, N, 3) point clouds

    Returns:
        torch.Tensor: Processed point clouds (B, N, 3)
    """
    with torch.no_grad():
        B, N, _ = sampled_pts.shape

        # --- 1. Centering
        mean = sampled_pts.mean(dim=1, keepdim=True)           # (B, 1, 3)
        centered = sampled_pts - mean                          # (B, N, 3)
        
        #if(B > 1):
        #    toprint(B > 1, centered[2,:10])
        # --- 2. Covariance and PCA
        cov = torch.einsum('bij,bik->bjk', centered, centered) / (N - 1)  # (B, 3, 3)
        #if(B>1):
        #    print(cov[2])
        eigvals, eigvecs = torch.linalg.eigh(cov)# eigvecs: (B, 3, 3)
        # eigvecs: (B, 3, 3), where eigvecs[b, :, i] is the i-th eigenvector for batch b
        sign = torch.where(eigvecs[:, 0, :] < 0, -1.0, 1.0)  # (B, 3)
        eigvecs = eigvecs * sign.unsqueeze(1)               # (B, 3, 3)

        #if(B > 1):
        #    toprint(B > 1,eigvals[2]) 
        #    toprint(B > 1,eigvecs[2])
        idx = torch.argsort(eigvals, dim=1, descending=True)             # (B, 3)

        # sort eigenvectors
        #batch_indices = torch.arange(B, device=sampled_pts.device).unsqueeze(-1)
        axes = torch.gather(eigvecs, 2, idx.unsqueeze(1).expand(-1, 3, -1))  # (B, 3, 3)
        #if(B > 1):
        #    toprint(B > 1,axes[2])
        # --- 3. Flip signs based on median projection
        #proj = torch.einsum('bij,bkj->bki', axes, centered)   # (B, N, 3)
        proj = torch.einsum('bni,bij->bnj', centered, axes)  # (B, N, 3)
        #if(B > 1):
        #    toprint(B > 1,proj[2,:10])
        for i in range(2):  # Loop over all 3 axes
            median = torch.median(proj[:, :, i], dim=1).values  # (B,)
            flip_mask = median < 0                              # (B,)
            sign = torch.where(flip_mask, -1.0, 1.0)            # (B,)
            #print(sign)
            #print(median)
            axes[:, :, i] *= sign.unsqueeze(1)

            # Only recompute axis 2 if axis 0 or 1 was flipped
            """if i < 2:
                flipped_batches = flip_mask.nonzero(as_tuple=True)[0]
                if len(flipped_batches) > 0:
                    axes[flipped_batches, :, 2] = torch.cross(
                        axes[flipped_batches, :, 0],
                        axes[flipped_batches, :, 1],
                        dim=1
                    )"""
        
        axes[:, :, 2] = torch.cross(
            axes[:, :, 0],
            axes[:, :, 1],
            dim=1
        )
        


        # --- 4. Apply rotation
        sampled_pts_aligned = torch.einsum('bij,bnj->bni', axes.transpose(1, 2), centered)  # (B, N, 3)
        #print(sampled_pts_aligned[:10])
        # --- 5. Compute scaling to bounding box area
        min_pt = sampled_pts_aligned.min(dim=1).values  # (B, 3)
        max_pt = sampled_pts_aligned.max(dim=1).values  # (B, 3)
        dims = max_pt - min_pt                          # (B, 3)
        area = 2 * (dims[:, 0]*dims[:, 1] + dims[:, 1]*dims[:, 2] + dims[:, 2]*dims[:, 0])  # (B,)
        scale = (1.7 / torch.sqrt(area)).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        # Apply scaling
        sampled_pts_scaled = sampled_pts_aligned * scale  # (B, N, 3)
        #print(sampled_pts_scaled[:10])
        return sampled_pts_scaled, scale[:,0,0]


def findUDF_from_PointCloud(sampled_pts_batch: torch.Tensor, important_points_batch: torch.Tensor) -> torch.Tensor:
    """
    Computes the unsigned distance function (UDF) from each important point to its
    nearest neighbor in the sampled point cloud.

    Args:
        sampled_pts_batch (B, N, 3): Batch of point clouds.
        important_points_batch (B, M, 3): Batch of important points describing the shapes in the dataset.

    Returns:
        Tensor of shape (B, M): Euclidean distances to nearest neighbor of the important points (UDF).
    """
    knn_output = knn_points(important_points_batch, sampled_pts_batch, K=1, return_sorted=False)
    return torch.sqrt(knn_output.dists.squeeze(2))    # (B, M)



def pca_transform_torch(x, pca_mean, pca_components):
    # x: shape (B, D)
    with torch.no_grad():
        x_centered = x - pca_mean
        return torch.matmul(x_centered, pca_components.T)  # result: (B, K)




