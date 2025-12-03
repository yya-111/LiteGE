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
    sampled_pts = sample_mesh(torch.from_numpy(vertices), torch.from_numpy(faces), sample_count=8000)
    sampled_pts_np = sampled_pts.float().cpu().numpy()

    # Center mesh
    center = np.mean(sampled_pts_np, axis=0, keepdims=True)
    vertices_centered = vertices - center
    # Scale to unit area
    vertices_scaled, scale_factor = scale_mesh_to_unit_surface_area(vertices_centered, faces)

    # Align via PCA
    #pca = PCA(svd_solver = 'full')
    #pca.fit(vertices_scaled)
    #inv_rot = pca.components_.T  # Transpose of PCA components
    #vertices_aligned = np.matmul(vertices_scaled , inv_rot)

    return vertices_scaled, scale_factor



from scipy.spatial.transform import Rotation as R_scipy

def rotate_pc_random_withgt(canonical_pc):
    R_rand_np = R_scipy.random().as_matrix()         # (3, 3)
    pc_rotated = canonical_pc @ R_rand_np.T
    
    return torch.from_numpy(pc_rotated), torch.from_numpy(R_rand_np)

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


