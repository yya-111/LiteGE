import torch

import numpy as np
import sklearn
from sklearn.decomposition import  PCA

def compute_mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Compute total surface area of a triangle mesh using vectorized operations.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        faces (np.ndarray): (M, 3) array of triangle indices.

    Returns:
        float: Total surface area.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_prod = np.cross(edge1, edge2)
    area = 0.5 * np.linalg.norm(cross_prod, axis=1)
    return np.sum(area)

def scale_mesh_to_unit_surface_area(vertices: np.ndarray, faces: np.ndarray):
    """
    Scale mesh so that its surface area becomes 1.

    Returns:
        scaled_vertices (np.ndarray), scale_factor (float)
    """
    current_area = compute_mesh_surface_area(vertices, faces)
    if current_area <= 0:
        raise ValueError("Mesh surface area must be positive.")
    
    scale_factor = 1 / np.sqrt(current_area)
    vertices_scaled = vertices * scale_factor
    return vertices_scaled, scale_factor

def processmesh(vertices: np.ndarray, faces: np.ndarray):
    """
    Process a mesh given its vertices and faces.

    Steps:
    - Center mesh at origin
    - Scale to unit surface area
    - Align via PCA

    Args:
        vertices (np.ndarray): (N, 3) vertex positions.
        faces (np.ndarray): (M, 3) triangle indices.

    Returns:
        processed_vertices (np.ndarray): (N, 3) transformed vertex positions.
        scale_factor (float): Scaling factor applied to the mesh.
    """
    # Center mesh
    center = np.mean(vertices, axis=0, keepdims=True)
    vertices_centered = vertices - center

    # Scale to unit area
    vertices_scaled, scale_factor = scale_mesh_to_unit_surface_area(vertices_centered, faces)

    # Align via PCA
    pca = PCA(svd_solver = 'full')
    pca.fit(vertices_scaled)
    inv_rot = pca.components_.T  # Transpose of PCA components
    vertices_aligned = np.matmul(vertices_scaled , inv_rot)

    return vertices_aligned, scale_factor


from scipy.spatial.transform import Rotation as R
import numpy as np
def apply_full_random_rotation(points: np.ndarray) -> np.ndarray:
    """
    Apply a random 3D rotation to a point cloud.

    Args:
        points (np.ndarray): (N, 3) array of 3D points.

    Returns:
        rotated_points (np.ndarray): (N, 3) array of rotated 3D points.
    """
    r = R.random()  # uniform random rotation
    rotated_points = r.apply(points)
    return rotated_points



import numpy as np
from sklearn.decomposition import PCA
import torch

def processmesh_augment_axis(vertices: np.ndarray, faces: np.ndarray):
    """
    Process a mesh given its vertices and faces.

    Steps:
    - Center mesh at origin using sampled points
    - Scale to unit surface area
    - Align via PCA
    - Randomly flip PCA axes for augmentation

    Args:
        vertices (np.ndarray): (N, 3) vertex positions.
        faces (np.ndarray): (M, 3) triangle indices.

    Returns:
        processed_vertices (np.ndarray): (N, 3) transformed vertex positions.
        scale_factor (float): Scaling factor applied to the mesh.
    """
    # Sample points for more robust center and PCA
    sampled_pts = sample_mesh(torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda(), sample_count=8000)
    sampled_pts_np = sampled_pts.float().cpu().numpy()

    # Center mesh
    center = np.mean(sampled_pts_np, axis=0, keepdims=True)
    vertices_centered = vertices - center
    sampled_pts_np -= center

    # Scale to unit area
    vertices_scaled, scale_factor = scale_mesh_to_unit_surface_area(vertices_centered, faces)
    sampled_pts_np = sampled_pts_np * scale_factor

    # PCA for alignment
    pca = PCA(svd_solver='full')
    pca.fit(sampled_pts_np)
    inv_rot = pca.components_.T  # shape (3, 3)

    # Random axis flip for augmentation (each axis has 50% chance to flip)
    flips = np.random.choice([1, -1], size=(3,))
    flip_matrix = np.diag(flips)
    inv_rot_aug = inv_rot @ flip_matrix

    # Apply aligned rotation
    vertices_aligned = np.matmul(vertices_scaled, inv_rot_aug)

    return vertices_aligned, scale_factor





def processmesh_noalign(vertices: np.ndarray, faces: np.ndarray):
    """
    Process a mesh given its vertices and faces.

    Steps:
    - Center mesh at origin
    - Scale to unit surface area
    - Align via PCA

    Args:
        vertices (np.ndarray): (N, 3) vertex positions.
        faces (np.ndarray): (M, 3) triangle indices.

    Returns:
        processed_vertices (np.ndarray): (N, 3) transformed vertex positions.
        scale_factor (float): Scaling factor applied to the mesh.
    """
    # Center mesh
    sampled_pts = sample_mesh(torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda(), sample_count=8000)
    sampled_pts_np = sampled_pts.float().cpu().numpy()

    # Center mesh
    center = np.mean(sampled_pts_np, axis=0, keepdims=True)
    vertices_centered = vertices - center

    # Scale to unit area
    vertices_scaled, scale_factor = scale_mesh_to_unit_surface_area(vertices_centered, faces)

    # Align via PCA
    #pca = PCA(svd_solver = 'full')
    #pca.fit(vertices_scaled)
    ##inv_rot = pca.components_.T  # Transpose of PCA components
    #vertices_aligned = np.matmul(vertices_scaled , inv_rot)

    return vertices_scaled, scale_factor


def processpointcloud(vertices: np.ndarray, verts_ori):
    """
    Process a mesh given its vertices and faces.

    Steps:
    - Center mesh at origin
    - Scale to unit surface area
    - Align via PCA

    Args:
        vertices (np.ndarray): (N, 3) vertex positions.
        faces (np.ndarray): (M, 3) triangle indices.

    Returns:
        processed_vertices (np.ndarray): (N, 3) transformed vertex positions.
        scale_factor (float): Scaling factor applied to the mesh.
    """
    # Center mesh
    center = np.mean(vertices, axis=0, keepdims=True)
    vertices_centered = vertices - center
    verts_ori = verts_ori - center

    
    #vertices_scaled, scale_factor = scale_mesh_to_unit_surface_area(vertices_centered, faces)

    # Align via PCA
    pca = PCA(svd_solver = 'full')
    pca.fit(vertices_centered)
    inv_rot = pca.components_.T  # Transpose of PCA components
    vertices_aligned = np.matmul(vertices_centered , inv_rot)
    verts_ori = np.matmul(verts_ori, inv_rot)
    
    # Scale to 1.7 area bounding box
    min_pt = np.min(vertices_aligned, axis=0)
    max_pt = np.max(vertices_aligned, axis=0)

    dims = max_pt - min_pt
    area = 2 * (dims[0]*dims[1] + dims[1]*dims[2] + dims[2]*dims[0])
    scale = 1.7 / np.sqrt(area)
    
    vertices_aligned *= scale
    verts_ori *= scale


    return vertices_aligned, verts_ori, scale


def getVoxelCenters(resolution = 127, pitch= 0.02):
    device = 'cuda'

    # === Step 1: Generate voxel center points (same for all meshes) === #
    half = resolution // 2
    grid_range = torch.arange(-half, half + 1, device=device) * pitch

    x = grid_range  # torch.arange(-half, half+1)*pitch
    y = grid_range
    z = grid_range

    # This makes axis-0 → x, axis-1 → y, axis-2 → z:
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')  
    voxel_centers = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)  # (N,3)
    voxel_centers_batched = voxel_centers.unsqueeze(0)  # (1, N, 3)
    return voxel_centers, voxel_centers_batched

import kaolin.ops.conversions as conversions
from kaolin.ops.mesh import check_sign
def getVoxelResult(verts, faces, voxel_centers_batched, resolution = 127, pitch=0.02):
    # Move to CUDA
    verts_cuda = torch.from_numpy(verts).float().cuda()#torch.from_numpy(verts.float()).cuda()
    faces_cuda = faces#torch.from_numpy(faces).cuda()
    total_length = resolution * pitch  # 2.54
    half_length = total_length / 2     # 1.27
    
    
    inside = check_sign(verts_cuda[None], faces_cuda.long(), voxel_centers_batched)  # bool
    voxelinside = inside.view(resolution, resolution, resolution)  # reshape to 3D grid
    
    origin = torch.tensor([-half_length, -half_length, -half_length], device=verts_cuda.device)  # (3,)
    scale = torch.tensor([total_length], device=verts_cuda.device)   # (3,)

    # Expand origin and scale to match batch size
    origin = origin.unsqueeze(0)  # (1, 3)
    scale = scale   # (1)

    voxelgrids = conversions.trianglemeshes_to_voxelgrids(
        verts_cuda[None], faces_cuda, resolution,
        origin=origin, scale=scale,
        return_sparse=False
    )
    voxelgrids = voxelgrids.view(resolution, resolution, resolution).bool()
    
    voxres = voxelgrids | voxelinside
    
    return voxres
    
    


import torch
import kaolin.ops.conversions as conversions
from kaolin.ops.mesh import check_sign

def getVoxelResult_batch_fixed(verts_b, faces, voxel_centers_batched, resolution=127, pitch=0.02):
    """
    Batch-version preserving original fixed-grid logic.

    Args:
        verts_b: (B, V, 3) numpy or torch tensor
        faces: (F, 3)
        voxel_centers_batched: (B, R^3, 3) precomputed for check_sign
    Returns:
        inside_grid: (B, R, R, R) bool grid from check_sign
        voxelgrids: (B, R, R, R) bool voxel occupancy
    """
    verts = torch.as_tensor(verts_b, dtype=torch.float32, device='cuda')  # (B, V, 3)
    faces = torch.as_tensor(faces, dtype=torch.long, device='cuda')       # (F, 3)

    B = verts.shape[0]
    total_length = resolution * pitch
    half_length = total_length / 2

    # 1) check_sign
    inside = check_sign(verts, faces, voxel_centers_batched)  # (B, R³)
    inside_grid = inside.view(B, resolution, resolution, resolution)

    # 2) fixed-grid origin & scale replicated across batch
    origin = torch.full((B, 3), -half_length, device=verts.device)
    scale = torch.full((B,), total_length, device=verts.device)

    # 3) trianglemeshes_to_voxelgrids using batch
    voxelgrids = conversions.trianglemeshes_to_voxelgrids(
        verts, faces, resolution,
        origin=origin, scale=scale,
        return_sparse=False
    )  # -> (B, R, R, R)
    voxelgrids = voxelgrids.bool()
    
    final_voxel = voxelgrids | inside_grid
    return final_voxel

    


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

import trimesh
import os
import numpy as np
import sklearn
from sklearn.decomposition import  PCA
def scale_mesh_to_unit_surface_area_tri(mesh):
    current_area = mesh.area
    if current_area <= 0:
        raise ValueError("Mesh surface area must be positive.")
    
    scale_factor = 1 / np.sqrt(current_area)
    mesh.apply_scale(scale_factor)
    
    return mesh, scale_factor
def processmesh(mesh):
    #from tqdm import tqdm
#res_pc = []
#for i in tqdm(range(train.shape[0])):
    center = (np.mean(mesh.vertices,axis = 0))[None,:]
    mesh.vertices = mesh.vertices - center
    #print(center)
    #print(np.mean(train[i], axis= 0))
    #print(train[i].shape)
    #mesh = trimesh.Trimesh(vertices = train[i], faces = template.faces)
    #pc =  sample_surface(mesh, 20000)
    #res_pc.append(pc)
    mesh, sc = scale_mesh_to_unit_surface_area_tri(mesh)
    pca = PCA()
    pc_i = mesh.vertices 
    pca.fit(pc_i)
    pca_comp = pca.components_
    #print(pca_comp.shape)
    inv_rot = np.transpose(pca_comp)
    #ident = np.matmul(pca_comp,inv)
    #print(ident)
    mesh.vertices  = np.matmul(mesh.vertices , inv_rot)
    
    return mesh,sc

import numpy as np

def random_rotation_with_max_angle(max_angle_deg=30.0):
    """Generate a random rotation matrix with geodesic angle ≤ max_angle_deg"""
    max_angle_rad = np.radians(max_angle_deg)

    # Random rotation axis (uniformly sampled on unit sphere)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    # Random rotation angle ∈ [-max_angle, max_angle]
    angle = np.random.uniform(-max_angle_rad, max_angle_rad)

    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R

def apply_random_rotation(points, max_angle_deg=30.0):
    R = random_rotation_with_max_angle(max_angle_deg)
    return points @ R.T

def apply_random_rotation(points, max_angle_deg=30.0):
    R = random_rotation_with_max_angle(max_angle_deg)
    return points @ R.T


def preprocess_mesh(
    verts:torch.Tensor,
    faces:torch.Tensor,
) -> torch.Tensor:
    """
    Preprocess a mesh by applying PCA-based orientation, centering and scaling to unit area

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
        
        sample_count = verts.shape[0]
        # --- 2. PCA orientation (based on sampled points)
        mean_pt = verts.mean(dim=0)
        centered = verts - mean_pt
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
        
        # --- 3. Enforce consistent axis signs
        proj = centered @ axes
        #if(prints):
        #    print(proj[:10])
        for i in range(2):
            if torch.median(proj[:, i]) < 0:
                axes[:, i] *= -1
        axes[:, 2] = torch.cross(axes[:, 0], axes[:, 1], dim=0)
        R = axes.T  # Rotation matrix (3,3)

        # --- 4. Rotate, center and scale mesh
        
        verts = (R @ (verts - mean_pt).T).T
        
        v0, v1, v2 = verts[faces].unbind(1)  # (F, 3) each
        face_areas = 0.5 * (v1 - v0).cross(v2 - v0, dim=1).norm(dim=1)  # (F,)
        sum_face_areas = face_areas.sum()
        scale = 1.0 / torch.sqrt(sum_face_areas)

        
        verts *= scale

        

        
    return verts, scale.item()
    

def preprocess_pointcloud(
    verts_ori:torch.Tensor,
    verts_remesh:torch.Tensor,
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
        verts_remesh = (R @ (verts_remesh - mean_pt).T).T
        sampled_pts = (R @ (sampled_pts - mean_pt).T).T
        #print(sampled_pts[:10])

        #scale_init = 1.0 / torch.sqrt(sum_face_areas)

        min_pt, _ = sampled_pts.min(dim=0)
        max_pt, _ = sampled_pts.max(dim=0)
        dims = max_pt - min_pt
        area = 2 * (dims[0]*dims[1] + dims[1]*dims[2] + dims[2]*dims[0])
        scale = 1.7 / torch.sqrt(area)
        

        verts_remesh *= scale
        
        sampled_pts *= scale
        verts_ori *= scale
        #print(sampled_pts[:10])

        

        
    return sampled_pts, verts_ori, verts_remesh, scale.item()#, verts_aligned, verts_ori, scale/scale_init




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





def pca_transform_torch(x, pca_mean, pca_components):
    # x: shape (B, D)
    with torch.no_grad():
        x_centered = x - pca_mean
        return torch.matmul(x_centered, pca_components.T)  # result: (B, K)




