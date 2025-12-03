import torch
from pytorch3d.ops import knn_points

import lzma
import torch



def decompress_lzma_to_numpy(input_lzma_path):
    """
    Decompresses an LZMA file and converts the data back to a NumPy array.
    Note: This requires knowing the original array's dtype and shape.
    A more robust solution would store this metadata alongside the compressed data.

    Args:
        input_lzma_path (str): Path to the input .lzma file.

    Returns:
        numpy.ndarray: The decompressed NumPy array, or None if an error occurs.
    """
    try:
        with open(input_lzma_path, 'rb') as f:
            compressed_bytes = f.read()

        # Decompress the bytes
        decompressed_bytes = lzma.decompress(compressed_bytes)
        print(f"Decompressed {input_lzma_path}")

        # To convert back to a numpy array, you need the original dtype and shape.
        # This example assumes you know it. In a real application, you'd need
        # to store this metadata.
        # Example: assuming the original array was float64 and shape (100, 100)
        # dtype = np.float64
        # shape = (100, 100)
        # array_data = np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
        # return array_data

        print("Decompression successful. Note: Converting back to numpy array requires original dtype and shape.")
        return decompressed_bytes # Return bytes as conversion requires metadata

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_lzma_path}")
        return None
    except lzma.LZMAError as e:
        print(f"LZMA decompression error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during decompression: {e}")
        return None
def step_geodesic_perpendicular_single_normalized(
    model, source_norm, destination_norm, pca, point_cloud, point_normals, mean_coord, std_coord, lr=1e-2
):
    """
    Performs a gradient descent step in normalized space, using surface normals in original space.

    Inputs:
        - source_norm: (3,) source in normalized space
        - destination_norm: (3,) destination in normalized space
        - pca: (F,) PCA feature vector
        - point_cloud: (N, 3) original 3D point cloud
        - point_normals: (N, 3) normals of point cloud
        - mean_coord, std_coord: (3,) used to de-normalize
        - lr: step size
    Output:
        - destination_world: (3,) projected to shape surface (world coords)
    """
    destination_norm = destination_norm.clone().detach().requires_grad_(True)
    source_norm = source_norm.unsqueeze(0)
    pca = pca.unsqueeze(0)

    # Predict and backpropagate
    pred_dist = model(source_norm, destination_norm.unsqueeze(0), pca)
    print(pred_dist)
    pred_dist.backward()

    with torch.no_grad():
        grad = destination_norm.grad  # in normalized space

        # Convert current normalized destination to world coordinates
        dest_world = destination_norm * std_coord + mean_coord

        # Get nearest point in original shape
        dists = torch.norm(point_cloud - dest_world.unsqueeze(0), dim=1)
        nn_idx = dists.argmin()
        normal = point_normals[nn_idx]

        # Convert gradient to world space for proper direction
        grad_world = grad * std_coord  # chain rule: d(world) = d(norm) * std

        # Remove component along normal
        dot = torch.dot(grad_world, normal)
        perp_grad_world = grad_world - dot * normal

        # Take step in world space
        dest_world_new = dest_world - lr * perp_grad_world

        # Project to nearest shape point
        dists = torch.norm(point_cloud - dest_world_new, dim=1)
        nn_idx = dists.argmin()
        projected_world = point_cloud[nn_idx]

        # Return re-normalized for next step if needed
        projected_norm = (projected_world - mean_coord) / std_coord

    return projected_world, projected_norm
from pytorch3d.ops import knn_points

def step_geodesic_single_normalized(
    model, source_norm, destination_norm, pca, point_cloud, mean_coord, std_coord, lr=1e-2
):
    """
    Performs a gradient descent step in normalized space, using surface normals in original space.

    Inputs:
        - source_norm: (3,) source in normalized space
        - destination_norm: (3,) destination in normalized space
        - pca: (1,F,) PCA feature vector
        - point_cloud: (N, 3) original 3D point cloud
        - point_normals: (N, 3) normals of point cloud
        - mean_coord, std_coord: (3,) used to de-normalize
        - lr: step size
    Output:
        - destination_world: (3,) projected to shape surface (world coords)
    """
    
    destination_norm = destination_norm.clone().unsqueeze(0).detach().requires_grad_(True)
    source_norm = source_norm.unsqueeze(0).detach()
    #pca = pca.unsqueeze(0).detach()
    destination = destination_norm*std_coord[None] + mean_coord[None]
    source = source_norm * std_coord[None] + mean_coord[None]
    #print(point_cloud.shape)
    #print(destination.shape, source.shape, std_coord.shape,mean_coord.shape)

    

    # Predict and backpropagate
    pred_dist = model(source_norm, destination_norm, pca, pca) +1.42*(destination - source).norm(dim=1)[:,None]
    #print("Distance:", pred_dist)
    pred_dist.sum().backward()

    with torch.no_grad():
        grad = destination_norm.grad  # in normalized space
        grad /= grad.norm()
        #print(f"Step : grad_norm = {grad.norm().item():.5f}") 
        #print()

        # Convert current normalized destination to world coordinates
        dest_world = destination_norm * std_coord + mean_coord
        #print("Destination:",dest_world)

        # Get nearest point in original shape
        #dists = torch.norm(point_cloud - dest_world.unsqueeze(0), dim=1)
        #nn_idx = dists.argmin()
        #normal = point_normals[nn_idx]

        # Convert gradient to world space for proper direction
        grad_world = grad * std_coord  # chain rule: d(world) = d(norm) * std
        #grad_world /= grad_world.norm()

        # Remove component along normal
        #dot = torch.dot(grad_world, normal)
        #perp_grad_world = grad_world - dot * normal

        # Take step in world space
        dest_world_new = dest_world - lr * grad_world
        #print("Destination new :",dest_world_new, dest_world_new.shape)
        #print("Points shape: ", point_cloud.shape)
        # Project to nearest shape point
        #dists = torch.norm(point_cloud - dest_world_new, dim=1)
        #nn_idx = dists.argmin()
        #print("Distance to point cloud:",dists.min())
        #projected_world = point_cloud[nn_idx]
        # Reshape to match batch dimensions: (1, N, 3) and (1, 1, 3)
        pc = point_cloud.unsqueeze(0)        # shape: (1, N, 3)
        query = dest_world_new.unsqueeze(0)  # shape: (1, 1, 3)

        # Find the nearest neighbor (k=1)
        knn_output = knn_points(query, pc, K=1, return_nn=True)
        projected_world = knn_output.knn[0, 0, 0]  # shape: (3,)
        #print(projected_world.shape)

        # Return re-normalized for next step if needed
        projected_norm = (projected_world - mean_coord) / std_coord

    return projected_world, projected_norm, pred_dist.item()


def step_geodesic_tangent_batched_normalized(
    model, source_norm, destination_norm, dest_normal, pca, point_cloud, point_cloud_normal, mean_coord, std_coord, lr=1e-2
):
    """
    Batched gradient step along geodesic using a model and point cloud projection.

    Inputs:
        - source_norm: (B, 3)
        - destination_norm: (B, 3)
        - dest_normal: (B,3) normal of destination point
        - pca: (B, F)
        - point_cloud: (B, N, 3)
        - mean_coord: (3,)
        - std_coord: (3,)
        - lr: learning rate (scalar)

    Outputs:
        - projected_world: (B, 3) — projected to nearest surface point in world space
        - projected_norm: (B, 3) — same point in normalized space
        - pred_dist: B— predicted distance loss (scalar)
        - projected_normals - B,3 projected point normals
        
    """

    B = source_norm.shape[0]

    # Enable gradient tracking for destination
    destination_norm = destination_norm.clone().detach().requires_grad_(True)
    source_norm = source_norm.detach()
    pca = pca.detach()

    # Broadcast mean and std
    mean_coord = mean_coord[None, :]      # (1, 3)
    std_coord = std_coord[None, :]        # (1, 3)

    # Convert to world space
    destination = destination_norm * std_coord + mean_coord  # (B, 3)
    source = source_norm * std_coord + mean_coord            # (B, 3)

    # Model forward pass
    pred_dist = model(source_norm, destination_norm, pca, pca)  # (B, 1) 
    pred_dist = pred_dist + 1.42 * torch.norm(destination - source, dim=1, keepdim=True)  # (B, 1)
    pred_dist_value = pred_dist.detach().clone()  # Safely clone value
    #print(pred_dist)

    # Backpropagate
    pred_dist.sum().backward()

    with torch.no_grad():
        grad = destination_norm.grad  # (B, 3)
        #print(pred_dist_value)
        #grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-9)
        

        # Step in world space
        dest_world = destination_norm * std_coord + mean_coord  # (B, 3)
        #print(pred_dist_value)
        grad_world = grad * std_coord  # (B, 3)
        #print(pred_dist_value)
        
        # Remove component along normal
        dot = (grad_world * dest_normal).sum(dim=1, keepdim=True)  # (B, 1)
        #print(pred_dist_value)
        perp_grad_world = grad_world - dot * dest_normal  # (B, 3)
        #print(pred_dist_value)
        perp_grad_world = perp_grad_world / (perp_grad_world.norm(dim=1, keepdim=True) + 1e-9)
        #print(pred_dist_value)
        
        dest_world_new = dest_world - lr * perp_grad_world  # (B, 3)
        #print(pred_dist_value)

        # Project to nearest point on shape surface
        # point_cloud: (B, N, 3), dest_world_new: (B, 3)
        query = dest_world_new[:, None, :]  # (B, 1, 3)
        knn_output = knn_points(query, point_cloud, K=1, return_nn=True)
        projected_world = knn_output.knn[:, 0, 0]  # (B, 3)
        #print(pred_dist_value)
        projected_indices = knn_output.idx[:, 0, 0]  # (B,)
        #print(pred_dist_value)
        #print(projected_indices.shape, point_cloud_normal.shape)
        projected_normal = point_cloud_normal[torch.arange(B, device=projected_indices.device), projected_indices]

        # Re-normalize
        projected_norm = (projected_world - mean_coord) / std_coord  # (B, 3)
        
        #print(projected_normal)

    return projected_world, projected_norm, pred_dist_value, projected_normal


def step_geodesic_batched_normalized(
    model, source_norm, destination_norm, pca, point_cloud, mean_coord, std_coord, lr=1e-2
):
    """
    Batched gradient step along geodesic using a model and point cloud projection.

    Inputs:
        - source_norm: (B, 3)
        - destination_norm: (B, 3)
        - pca: (B, F)
        - point_cloud: (B, N, 3)
        - mean_coord: (3,)
        - std_coord: (3,)
        - lr: learning rate (scalar)

    Outputs:
        - projected_world: (B, 3) — projected to nearest surface point in world space
        - projected_norm: (B, 3) — same point in normalized space
        - pred_dist.item(): float — total predicted distance loss (scalar)
    """

    B = source_norm.shape[0]

    # Enable gradient tracking for destination
    destination_norm = destination_norm.clone().detach().requires_grad_(True)
    source_norm = source_norm.detach()
    pca = pca.detach()

    # Broadcast mean and std
    mean_coord = mean_coord[None, :]      # (1, 3)
    std_coord = std_coord[None, :]        # (1, 3)

    # Convert to world space
    destination = destination_norm * std_coord + mean_coord  # (B, 3)
    source = source_norm * std_coord + mean_coord            # (B, 3)

    # Model forward pass
    pred_dist = model(source_norm, destination_norm, pca, pca)  # (B, 1) 
    pred_dist = pred_dist + 1.42 * torch.norm(destination - source, dim=1, keepdim=True)  # (B, 1)

    # Backpropagate
    pred_dist.sum().backward()

    with torch.no_grad():
        grad = destination_norm.grad  # (B, 3)
        #grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-9)

        # Step in world space
        dest_world = destination_norm * std_coord + mean_coord  # (B, 3)
        grad_world = grad * std_coord  # (B, 3)
        #grad_world = grad_world / (grad_world.norm(dim=1, keepdim=True) + 1e-9)
        dest_world_new = dest_world - lr * grad_world  # (B, 3)

        # Project to nearest point on shape surface
        # point_cloud: (B, N, 3), dest_world_new: (B, 3)
        #query = dest_world_new[:, None, :]  # (B, 1, 3)
        #knn_output = knn_points(query, point_cloud, K=1, return_nn=True)
        #projected_world = knn_output.knn[:, 0, 0]  # (B, 3)
        # point_cloud: (B, N, 3), dest_world_new: (B, 3)
        query = dest_world_new[:, None, :]  # (B, 1, 3)

        # Compute squared distances
        dists = torch.sum((point_cloud - query)**2, dim=-1)  # (B, N)

        # Find nearest point indices
        min_indices = dists.argmin(dim=-1)  # (B,)

        # Gather nearest points
        projected_world = point_cloud[torch.arange(point_cloud.shape[0], device=point_cloud.device), min_indices]  # (B, 3)

        # Re-normalize
        projected_norm = (projected_world - mean_coord) / std_coord  # (B, 3)

    return projected_world, projected_norm, pred_dist
    #return dest_world_new, (dest_world_new - mean_coord)/std_coord , pred_dist





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


def preprocess_pointcloud_withnormals(
    verts_ori:torch.Tensor,
    sampled_pts: torch.Tensor, 
    samples_normals:torch.Tensor,
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
        samples_normals = (R @ (samples_normals).T).T
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

        

        
    return sampled_pts, samples_normals, verts_ori, scale.item()

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





def sample_mesh_withnormals(vertices, faces, sample_count=10000):
    """
    Sample points and normals from a mesh using area-weighted sampling.

    Args:
        vertices: (V, 3) torch tensor
        faces: (F, 3) torch tensor (indices into vertices)
        sample_count: number of points to sample

    Returns:
        samples: (sample_count, 3)
        normals: (sample_count, 3)
    """

    # Step 1: get vertices of each face
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Step 2: compute face areas using cross product
    face_normals = torch.cross(v1 - v0, v2 - v0)  # (F, 3)
    face_areas = 0.5 * torch.norm(face_normals, dim=1)  # (F,)

    # Step 3: sample faces proportional to area
    face_probs = face_areas / face_areas.sum()
    sampled_face_indices = torch.multinomial(face_probs, sample_count, replacement=True)  # (sample_count,)

    # Gather face vertices
    v0_sampled = v0[sampled_face_indices]
    v1_sampled = v1[sampled_face_indices]
    v2_sampled = v2[sampled_face_indices]
    normals_sampled = face_normals[sampled_face_indices]
    normals_sampled = torch.nn.functional.normalize(normals_sampled, dim=1)  # (sample_count, 3)

    # Step 4: sample points in the triangle using barycentric coordinates
    u = torch.rand(sample_count, 1, device=vertices.device)
    v = torch.rand(sample_count, 1, device=vertices.device)
    is_outside = u + v > 1
    u[is_outside] = 1 - u[is_outside]
    v[is_outside] = 1 - v[is_outside]
    w = 1 - (u + v)

    # Compute point positions
    samples = v0_sampled * w + v1_sampled * u + v2_sampled * v  # (sample_count, 3)

    return samples, normals_sampled

