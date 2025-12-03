import torch
import numpy as np
import time
from PointCloudUDFConstruction import sample_mesh, preprocess_pointcloud,preprocess_pointcloud_batch,findUDF_from_PointCloud,pca_transform_torch
import trimesh
import os
from CoordMLP import CoordMLP, save_model, load_model
# Load
import pickle
import argparse
parser = argparse.ArgumentParser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('-ck', '--checkpointpath',nargs='+',  # Accept one or more values
    type=str,   # Each item is a string
    help='List of checkpoint paths'
)
parser.add_argument('-bs', '--batch_size', type=int, default=96)
parser.add_argument('-p', '--points', type=int, default=2000)
args = parser.parse_args()
modelname = args.checkpointpath[0]

model = CoordMLP(200, embedding_size=240).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)

startepoch, model, optimizer,scheduler = load_model(model, optimizer, scheduler,modelname, device)
model = model.cuda()
model.eval()


#pcadimension = 240
pcadimension = 240
with open('pca_model_PointCloudUDF_shapematch_SMAL_0_16fil.pkl', 'rb') as f:
    pca = pickle.load(f)
    pca_mean = torch.from_numpy(pca.mean_).float().to(device)[None]         # shape: (1,D,)
    pca_components = torch.from_numpy(pca.components_).float().to(device)[:pcadimension]  # shape: (K, D)

faces = trimesh.load("MeshTestSample" + os.sep + "meshtest_6_remesh.off", process=False).faces

device = 'cuda'
# 1. Load fixed voxel reference points (M, 3)
important_points = np.load("VoxelsCenterNearSurfaceHighstd)PointCloudUDF_0_16fil.npy")  # shape: (M, 3)
important_points = torch.from_numpy(important_points).float().to(device)

# 2. Sample 100 point clouds from meshverticesori
sampled_list = []
testdata = "UDFTestRemeshedShapeMatch.npz"#"TestDataShapeMatchSMALRemeshed.npz"
meshverticesori = np.load(testdata)['testmeshvertices']
for verts in meshverticesori:  # meshverticesori is a list of (V, 3) tensors
    sampled_pts = sample_mesh(torch.from_numpy(verts), torch.from_numpy(faces), sample_count=args.points)  # (8000, 3)
    sampled_list.append(sampled_pts)

# 3. Stack into batch
B = args.batch_size
sampled_batch = torch.stack(sampled_list).float().cuda()[:B]  # (B=100, 8000, 3)

# 4. Expand important points once
B = sampled_batch.shape[0]
M = important_points.shape[0]
sampled_srcs = torch.randn(B,3).float().cuda()
sampled_dests = torch.randn(B,3).float().cuda()
important_points_batch = important_points.unsqueeze(0).expand(B, -1, -1)  # (B, M, 3)

torch.cuda.synchronize()

# Warm-up: do 3-5 iterations
for _ in range(5):
    processed_pts, _ = preprocess_pointcloud_batch(sampled_batch)
    udf = findUDF_from_PointCloud(processed_pts, important_points_batch)
    udf_pca = pca_transform_torch(udf, pca_mean, pca_components)
    with torch.no_grad():
        model(sampled_srcs, sampled_dests, udf_pca, udf_pca)

times = []
#for _ in range(5):
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
before_pca_allocated = torch.cuda.memory_allocated()
torch.cuda.synchronize()
start = time.perf_counter()

processed_pts, _ = preprocess_pointcloud_batch(sampled_batch)
udf = findUDF_from_PointCloud(processed_pts, important_points_batch)
udf_pca = pca_transform_torch(udf,pca_mean, pca_components)
with torch.no_grad():
    model(sampled_srcs, sampled_dests, udf_pca, udf_pca)
torch.cuda.synchronize()
end = time.perf_counter()
times.append(end - start)

print(f"Mean: {np.mean(times):.4f}s | Min: {np.min(times):.4f}s | Std: {np.std(times):.4f}s")
pca_peak_memory = torch.cuda.max_memory_allocated()
pca_end_memory = torch.cuda.memory_allocated()

# Step 6: Calculate memory usage
pca_memory_used = pca_peak_memory - before_pca_allocated
pca_memory_MB = pca_memory_used / (1024 ** 2)
pca_endmem_used = pca_end_memory - before_pca_allocated
pca_endmem_MB = pca_endmem_used / (1024**2)

print(f"PCA Peak memory used: {pca_memory_MB:.2f} MB")
print(f"PCA End memory used: {pca_endmem_MB:.2f} MB")


