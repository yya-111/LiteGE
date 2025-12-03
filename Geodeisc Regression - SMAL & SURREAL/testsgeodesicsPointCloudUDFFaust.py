import torch
import torch.nn.functional as F
import sys, os; original_cwd = os.getcwd(); module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.append(module_path)
from CoordMLP import CoordMLP, save_model, load_model

from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import argparse
import os
from torch.utils.data import Dataset, DataLoader

from PointCloudUDFConstruction import sample_mesh, slice_point_cloud_middle_tensor, preprocess_pointcloud,preprocess_pointcloud_batch,findUDF_from_PointCloud,pca_transform_torch
sys.path.remove(module_path); os.chdir(original_cwd)


import lzma
#from buildvertexcache import buildcache, locate_corresponding_vertex_in_Y
#from buildvertexcache import fast_locate_corresponding_vertex_in_Y, fast_locate_multi_vertex_in_Y
import numpy as np
from sklearn import metrics
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pcadimension = 240
with open('pca_model_PointCloudUDF_SMALSURREAL_0_12fil.pkl', 'rb') as f:
    pca = pickle.load(f)
    pca_mean = torch.from_numpy(pca.mean_).float().to(device)[None]         # shape: (1,D,)
    pca_components = torch.from_numpy(pca.components_).float().to(device)[:pcadimension]  # shape: (K, D)

# Generate random rotation matrix using QR decomposition
def random_rotation_matrix(device='cuda'):
    A = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    
    # Ensure right-handed coordinate system (det(R) = +1)
    if torch.det(Q) < 0:
        Q[:, 2] *= -1
    return Q  # shape: (3, 3)
def auc_at_threshold(errors, max_threshold=0.1, num_points=40):
    errors = np.asarray(errors)
    thresholds = np.linspace(0.0, max_threshold, num_points)

    # Compute recall at each threshold
    recall = [(errors <= t).mean() for t in thresholds]

    # Integrate using trapezoidal rule
    auc = metrics.auc(thresholds, recall)

    # Normalize by max_threshold to scale AUC into [0,1]
    return auc / max_threshold

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir')
# Put custom arguments here
parser.add_argument('-n', '--num_epoch', type=int, default=50)
parser.add_argument('-neu', '--num_neurons', type=int,default = 200)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-c', '--continue_training', action='store_true')
parser.add_argument('-ck', '--checkpointpath',nargs='+',  # Accept one or more values
    type=str,   # Each item is a string
    help='List of checkpoint paths'
)
parser.add_argument('-d', '--delta', type=float,default = 0.035)
args = parser.parse_args()
#model = CoordMLP()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



startepoch = 0
allmodels = []
for modelname in args.checkpointpath:
    model = CoordMLP(neurons=args.num_neurons, embedding_size=240).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=1e-4)

    startepoch, model, optimizer,scheduler = load_model(model, optimizer, scheduler,modelname, device)
    model = model.cuda()
    model.eval()
    allmodels.append(model)

import gzip
import pickle
# Load
file_path = "/notebooks/Faust APSP Data" + os.sep + "100MeshFaustDistances.pkl" + os.sep + "100MeshFaustDistances.pkl"
with open(file_path, 'rb') as f:
    gtdata = pickle.load(f)
    for i in range(len(gtdata)):
        gtdata[i] = 1.42*gtdata[i]

print("Len gt data:",len(gtdata))
meshdir = "/notebooks/Faust APSP Data" + os.sep + "FaustMeshes" + os.sep + "off"





mean_coord = np.array([ 8.4521987e-02, -8.0156110e-02 ,-3.9249542e-05] )[None,:]
std_coord = np.array([0.31262797, 0.18953875, 0.07636771])[None,:]
#meshvertices = (np.load(meshvertices)['arr_0'][np.load(testidxname)]- mean_coord)/std_coord


#print("Meshvertices shape:",meshverticesori.shape)

###2. Sample Random Pair of Meshes and vertex index and test.
sources = np.load("/notebooks/Faust APSP Data" + os.sep + "SourcesFAUST.npy")
import trimesh
import os


pointclouds_test = []
pointclouds_testori = []
#torch.manual_seed(41)
meshverticesori = []
from scipy.spatial.transform import Rotation as R
for i in range(100):
    # Suppose meshverticesori[i] is your (N, 3) point cloud
    mesh_i = trimesh.load(meshdir + os.sep + f"tr_reg_{i:03d}.off",process=False)
    
    points = mesh_i.vertices #meshverticesori[i]  # shape (N, 3)

    # Generate a random 3D rotation
    rotation = R.random()  # Random rotation in 3D
    rotated_points = rotation.apply(points)  # Apply the rotation

    # Optional: store it back
    vertices = rotated_points
    
    vertices = torch.from_numpy(vertices).float().cuda()
    #noise_std = 0.01  # Adjust as needed
    #vertices = vertices + torch.randn_like(vertices) * noise_std
    
    sampled_pts = sample_mesh(vertices, torch.from_numpy(mesh_i.faces).cuda(), sample_count = 700)
    with torch.no_grad():
        pointclouds_testori.append(sampled_pts.clone())
    #sampled_pts = slice_point_cloud_middle_tensor(sampled_pts,keep_ratio=0.85)
    #noise_std = 0.01  # Adjust as needed
    #sampled_pts = sampled_pts + torch.randn_like(sampled_pts) * noise_std
    sampled_pts, vertices, scale = preprocess_pointcloud(vertices, sampled_pts, )
    pointclouds_test.append(sampled_pts.float())
    meshverticesori.append(vertices.detach().cpu().numpy())
    gtdata[i] *= scale
all_values = np.concatenate([arr.ravel() for arr in gtdata])

# Calculate mean
mean_val = all_values.mean()

print("Gt Data mean:", mean_val)

pointclouds_testori = torch.stack(pointclouds_testori)
print(pointclouds_testori.shape)
pointclouds_testori, _ = preprocess_pointcloud_batch(pointclouds_testori.float())

l2_diffs = []
max_diffs = []
for i in range(len(meshverticesori)):
    processed_batch = pointclouds_testori[i]
    processed_single = pointclouds_test[i]
    l2_diff = torch.norm(processed_single - processed_batch).item()
    max_diff = (processed_single - processed_batch).abs().max().item()

    l2_diffs.append(l2_diff)
    max_diffs.append(max_diff)

    if l2_diff > 1e-3 or max_diff > 1e-3:
        print(f"[{i}] Warning: High diff detected → L2: {l2_diff:.6f}, Max: {max_diff:.6f}")

# --- Summary
print("\n=== Summary over 100 samples ===")
print(f"Max L2 diff: {max(l2_diffs):.8f}")
print(f"Max Abs diff: {max(max_diffs):.8f}")
print(f"Avg L2 diff: {sum(l2_diffs)/len(l2_diffs):.8f}")
print(f"Avg Max diff: {sum(max_diffs)/len(max_diffs):.8f}")


#
#print(pointclouds_testori.shape)
important_points_batch =  np.load("VoxelsCenterNearHumanSMALSurfaceHighstd)PointCloudUDF_0_12fil.npy")

important_points_batch = torch.from_numpy(important_points_batch).float().cuda()
M = important_points_batch.shape[0]
B = pointclouds_testori.shape[0]
# Expand query points to each batch without memory duplication
important_points_batch = important_points_batch.unsqueeze(0).expand(B, M, 3)  # (B, M, 3)
meshvertices = [(meshverticesori[i]- mean_coord)/std_coord for i in range(len(meshverticesori))]


UDFreps = findUDF_from_PointCloud(pointclouds_testori.float(), important_points_batch) 
pcaUDFrep = pca_transform_torch(UDFreps,  pca_mean, pca_components)
print(pcaUDFrep.shape)
#print()
meshpca = pcaUDFrep.detach().cpu().numpy()    
    

K = 20000
NTEST = (100,K)
rng = np.random.default_rng()
mesh1_indices  = np.empty(NTEST[0], dtype=np.int32)
vertex1_indices = np.empty(NTEST, dtype=np.int32)
vertex2_indices  = np.empty(NTEST, dtype=np.int32)

print("No Vertex:", meshvertices[0].shape[0])
print("preparing test data: ", NTEST, " samples")
for b in range(NTEST[0]):
    m1 = b#np.full((1, K), b, dtype=np.int32)
    v1 = rng.integers(0, sources.shape[0], size=(1,K))
    v2 = rng.integers(0, meshvertices[b].shape[0], size=(1,K))
    
    start, end = b, (b+1)
    mesh1_indices [start:end] = m1
    vertex2_indices [start:end] = v2
    vertex1_indices[start:end] = v1
resultpred = []
resulttarget = []
error = 0
with torch.no_grad():
    for i in tqdm(range(NTEST[0])):
        #batch = loader[i]
        #vertex1 = batch['vertex1'].cuda()
        #vertex2 = batch['vertex2'].cuda()
        #pca1 =  batch['pca1'].cuda()
        #pca2 = batch['pca2'].cuda()
        # batch['pca1'], batch['pca2']: (B, 400)
        # batch['vertex1'], batch['vertex2']: (B, 3)
        # but vertex1 is constant within the batch
        batch_i = i   # 0 for first batch, 1 for second, …
        mesh1_idx  = mesh1_indices[batch_i]
        vertex2_idx  = vertex2_indices[batch_i]
        vertex1_idx= sources[vertex1_indices[batch_i]]
        
        pca1    = meshpca[mesh1_idx]             # (400,)
        pca1 = pca1[None, :] #(1,400)
        pca1 = np.repeat(pca1, K, axis=0)  # shape (N, 400)
        vertex2 = meshvertices[mesh1_idx][vertex2_idx]       # (N,3,)
        vertex1 = meshvertices[mesh1_idx][vertex1_idx]    # (N,3,)
        target = gtdata[mesh1_idx][ vertex1_indices[batch_i], vertex2_idx] # N,
        #Compute euclidean
        vertex2ori = meshverticesori[mesh1_idx][vertex2_idx]       # (N,3,)
        vertex1ori = meshverticesori[mesh1_idx][vertex1_idx]    # (N,3,)
        
        distances = 1.42*np.linalg.norm( vertex2ori  -  vertex1ori , axis=1)  # shape: (N,)
        #target -= distances
        print(np.min(target-distances))
        target = target[:,None] #(N,1)
        
        pca1 =  torch.from_numpy(pca1).float().cuda()
        vertex2=   torch.from_numpy(vertex2).float().cuda()
        vertex1= torch.from_numpy(vertex1).float().cuda()
        #print(meshvertices[mesh2_idx])
        
        preddist = allmodels[0](vertex1, vertex2, pca1, pca1)
        preddist = preddist.cpu().numpy() + distances[:,None]
        resultpred.extend(preddist)
        resulttarget.extend(target)
        #print(np.mean(np.abs(preddist-target)))
        
        
        #initial_memory = torch.cuda.memory_allocated(device)
        # run the model on all B candidate pairs at once
        # pred_dist: (B,) predicted geodesic distance for each (vertex1, vertex2) pair
        
        #after_inference_memory = torch.cuda.memory_allocated(device)

        # Get the peak memory allocated during the operation
        #peak_memory = torch.cuda.max_memory_allocated(device)

        #print(f"Memory allocated before inference: {initial_memory / 1024**2:.2f} MB")
        #print(f"Memory allocated after inference: {after_inference_memory / 1024**2:.2f} MB")
        #print(f"Peak memory allocated during inference: {peak_memory / 1024**2:.2f} MB")
        

        # record the matching vertex2 index
        
error = 0.0
resultpred = np.array(resultpred)
resulttarget = np.array(resulttarget)

mederror = np.median(np.abs(resultpred - resulttarget))

print("Median:", mederror)

print("Error below 0.10:", (np.abs(resultpred - resulttarget) < 0.1).mean())






