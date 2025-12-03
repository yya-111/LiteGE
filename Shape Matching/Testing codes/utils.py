import torch

import numpy as np
import sklearn
from sklearn.decomposition import  PCA
import trimesh
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
    
from sklearn.neighbors import NearestNeighbors
def computecosinesim(meshpca, meshpcaori):
    # Normalize the rows
    meshpca_norm = meshpca / np.linalg.norm(meshpca, axis=1, keepdims=True)
    meshpcaori_norm = meshpcaori / np.linalg.norm(meshpcaori, axis=1, keepdims=True)

    # Compute cosine similarity as row-wise dot product
    cos_sim = np.sum(meshpca_norm * meshpcaori_norm, axis=1)
    return cos_sim
def compute_udf(pc: np.ndarray, vox_center_include: np.ndarray) -> np.ndarray:
    """
    Compute UDF (Unsigned Distance Function) from vox_center_include to the point cloud pc.

    Args:
        pc: (N, 3) array of point cloud data
        vox_center_include: (M, 3) array of query points

    Returns:
        udf: (M,) array of unsigned distances to the nearest point in pc
    """
    # Fit nearest neighbors model on point cloud
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(pc)

    # Query distances from each voxel center to the closest point in the point cloud
    distances, _ = nn.kneighbors(vox_center_include)

    return distances[:, 0]

def processmesh(vertices: np.ndarray, faces: np.ndarray):
    """
    Process a mesh given its vertices and faces.

    Steps:
    - Center mesh at origin
    - Scale to unit surface area
    

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



import torch

# Assume this function is already defined elsewhere and works with torch tensors:
# def sample_mesh(vertices: torch.Tensor, faces: torch.Tensor, sample_count: int) -> torch.Tensor:
#     """Samples points uniformly from the surface of a mesh."""
#     pass


def calculate_mesh_surface_area(vertices: torch.Tensor, faces: torch.Tensor):
    """
    Calculates the total surface area of a triangular mesh.

    Args:
        vertices (torch.Tensor): (N, 3) vertex positions.
        faces (torch.Tensor): (M, 3) triangle indices.

    Returns:
        total_area (torch.Tensor): Scalar tensor representing the total surface area.
    """
    
    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute edge vectors for each triangle
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Calculate the cross product of edge vectors
    # Magnitude of cross product is twice the area of the parallelogram formed by the vectors
    # So, 0.5 * norm(cross_product) is the area of the triangle
    cross_product = torch.cross(edge1, edge2, dim=1) # Shape (M, 3)
    triangle_areas = 0.5 * torch.norm(cross_product, dim=1) # Shape (M,)

    # Sum all triangle areas to get the total mesh surface area
    total_area = torch.sum(triangle_areas)
    return total_area


def scale_mesh_to_unit_surface_area_torch(vertices: torch.Tensor, faces: torch.Tensor):
    """
    Scales the mesh such that its total surface area is 1.0.

    Args:
        vertices (torch.Tensor): (N, 3) vertex positions.
        faces (torch.Tensor): (M, 3) triangle indices.

    Returns:
        scaled_vertices (torch.Tensor): (N, 3) scaled vertex positions.
        scale_factor (float): Scaling factor applied to the mesh.
        
    """
    with torch.no_grad():
        current_area = calculate_mesh_surface_area(vertices, faces)



        scale_factor = torch.sqrt(1.0 / current_area)
        scaled_vertices = vertices * scale_factor

    return scaled_vertices, scale_factor.item()


def processmesh_torch(vertices: torch.Tensor, faces: torch.Tensor):
    """
    Process a mesh given its vertices and faces using PyTorch.

    Steps:
    - Center mesh at origin
    - Scale to unit surface area

    Args:
        vertices (torch.Tensor): (N, 3) vertex positions. Expected float type.
        faces (torch.Tensor): (M, 3) triangle indices. Expected long (int64) type.

    Returns:
        processed_vertices (torch.Tensor): (N, 3) transformed vertex positions.
        scale_factor (float): Scaling factor applied to the mesh.
    """
    # Ensure tensors are on the desired device (GPU if available) and correct dtypes
    

    # Center mesh at origin based on sampled points
    # The `sample_mesh` function is assumed to be defined and accept/return torch tensors
    sampled_pts = sample_mesh(vertices, faces, sample_count=8000)
    center = torch.mean(sampled_pts, dim=0, keepdim=True) # Shape (1, 3)
    vertices_centered = vertices - center
    sampled_pts -= center

    # Scale to unit surface area
    vertices_scaled, scale_factor = scale_mesh_to_unit_surface_area_torch(vertices_centered, faces)
    sampled_pts *= scale_factor
    return vertices_scaled, scale_factor, sampled_pts


import pymeshlab
import numpy as np
import torch
import tempfile
import os
from sklearn.neighbors import NearestNeighbors

def poisson_disk_sample(mesh: trimesh.Trimesh, num_samples: int):
    with tempfile.NamedTemporaryFile(suffix=".off", delete=False) as f:
        mesh.export(f.name)
        temp_path = f.name

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(temp_path)
    ms.apply_filter('generate_sampling_poisson_disk', samplenum=num_samples, subsample=True)
    sampled = ms.current_mesh()
    os.remove(temp_path)
    return np.array(sampled.vertex_matrix())

def compute_avg_nearest_neighbor(points):
    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    dists, _ = nbrs.kneighbors(points)
    return dists[:, 1].mean()

def processmeshbroken_torch(vertices: torch.Tensor, faces: torch.Tensor):
    """
    Center mesh using Poisson disk samples and scale so that mean NN distance between 7000 Poisson samples is 0.01181

    Args:
        vertices (torch.Tensor): (N, 3) float32
        faces (torch.Tensor): (M, 3) int64

    Returns:
        vertices_scaled (torch.Tensor): (N, 3) processed vertices
        scale_factor (float): mesh was scaled by this factor
        sampled_pts (torch.Tensor): (7000, 3) poisson samples from processed mesh
    """
    import trimesh

    # Convert torch tensors to numpy
    verts_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    # Create Trimesh object
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

    # Poisson disk sample before scaling
    poisson_samples = poisson_disk_sample(mesh, num_samples=7000)
    nn_mean = compute_avg_nearest_neighbor(poisson_samples)

    # Compute scaling factor to make mean NN distance = 0.01181
    target_dist = 0.01181
    scale_factor = target_dist / nn_mean

    # Center using Poisson sample mean
    center = poisson_samples.mean(axis=0)
    verts_centered = verts_np - center
    poisson_samples -= center

    # Scale both vertices and samples
    verts_scaled = verts_centered * scale_factor
    poisson_samples_scaled = poisson_samples * scale_factor

    # Convert back to torch
    vertices_scaled = torch.from_numpy(verts_scaled).to(vertices.device).type(vertices.dtype)
    sampled_pts = torch.from_numpy(poisson_samples_scaled).to(vertices.device).type(vertices.dtype)

    return vertices_scaled, scale_factor, sampled_pts

def poisson_disk_sample_from_points(points: np.ndarray, num_samples: int):
    """
    Compute Poisson disk sampling from a point cloud using MeshLab.
    """
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        np.savetxt(f.name, points, fmt='%.6f')
        temp_path = f.name

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(temp_path)
    ms.apply_filter('generate_sampling_poisson_disk', samplenum=num_samples, subsample=True)
    sampled = ms.current_mesh()
    os.remove(temp_path)
    return np.array(sampled.vertex_matrix())

import time
def processmesh_with_pointcloud(vertices: torch.Tensor, faces: torch.Tensor):
    """
    Sample mesh with 8K points, compute Poisson samples from those, and scale+center the mesh and samples.
    
    Args:
        vertices (torch.Tensor): (N, 3)
        faces (torch.Tensor): (M, 3)
    
    Returns:
        vertices_scaled (torch.Tensor): (N, 3)
        scale_factor (float)
        pointcloud_scaled (torch.Tensor): (8000, 3)
    """
    # Convert to numpy
    verts_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    # Create Trimesh
    
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

    # Area-weighted sample 8000 points
    samples_8k = mesh.sample(8000)
    s = time.time()
    # Compute 500 Poisson disk samples from the 8K point cloud, n
    # Notice we only use point clouds to normalize, which means our codes work with point clouds inputs. 
    # The mesh vertices is transformed and returned as we need them for computing geodesic errors after query
    poisson_samples = poisson_disk_sample_from_points(samples_8k, num_samples=500)

    # Compute average NN distance
    nn_mean = compute_avg_nearest_neighbor(poisson_samples)
    target_dist =0.03835#0.12121
    scale_factor = target_dist / nn_mean

    # Center using 8K point cloud mean
    center = samples_8k.mean(axis=0)
    verts_centered = verts_np - center
    samples_8k_centered = samples_8k[:8000] - center

    # Apply scaling
    verts_scaled = verts_centered * scale_factor
    samples_8k_scaled = samples_8k_centered * scale_factor

    # Convert back to torch
    vertices_scaled = torch.from_numpy(verts_scaled).to(vertices.device).type(vertices.dtype)
    pointcloud_scaled = torch.from_numpy(samples_8k_scaled).to(vertices.device).type(vertices.dtype)

    return vertices_scaled, scale_factor, pointcloud_scaled, time.time() - s