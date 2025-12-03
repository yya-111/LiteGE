import torch
from CoordMLP import CoordMLP, save_model, load_model
from PointCloudUDFConstruction import findUDF_from_PointCloud, pca_transform_torch
from datatest import MeshCorrespondenceDataset
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
import lzma
from buildvertexcache import buildcache, locate_corresponding_vertex_in_Y,buildcache_3tiers
from buildvertexcache import fast_locate_corresponding_vertex_in_Y, fast_locate_multi_vertex_in_Y, fast_locate_multi_vertex_in_Y_3tiered
import numpy as np
from sklearn import metrics
import trimesh
def createanisomesh(vertices,faces):
#vertices = mesh.vertices.tolist()
#faces = mesh.faces.copy()
    new_faces = []
    
    # Decide percentage of triangles to split
    num = int(0.5 * len(faces))  # 50% triangles split
    chosen = np.random.choice(len(faces), size=num, replace=False)
    
    for idx, face in enumerate(faces):
        if idx in chosen:
            v_idx = face
            v = [np.array(vertices[i]) for i in v_idx]
    
            # Randomly choose edge (v0-v1, v1-v2, v2-v0)
            edge_id = np.random.choice(3)
            v0, v1 = v[edge_id], v[(edge_id + 1) % 3]
            v2_idx = v_idx[(edge_id + 2) % 3]
    
            # Anisotropic split: near v0 or v1
            t = np.random.uniform(0.03, 0.25) if np.random.rand() < 0.5 else np.random.uniform(0.8, 0.97)
            p = (1 - t) * v0 + t * v1
            new_idx = len(vertices)
            vertices.append(p.tolist())
    
            # Two new triangles from split
            new_faces.append([v_idx[edge_id], new_idx, v2_idx])
            new_faces.append([new_idx, v_idx[(edge_id + 1) % 3], v2_idx])
        else:
            new_faces.append(face.tolist())
    
    # Build new mesh
    new_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(new_faces))
    return new_mesh

def auc_at_threshold(errors, max_threshold=0.1, num_points=40):
    errors = np.asarray(errors)
    thresholds = np.linspace(0.0, max_threshold, num_points)

    # Compute recall at each threshold
    recall = [(errors <= t).mean() for t in thresholds]

    # Integrate using trapezoidal rule
    auc = metrics.auc(thresholds, recall)

    # Normalize by max_threshold to scale AUC into [0,1]
    return auc / max_threshold
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

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir')
# Put custom arguments here
parser.add_argument('-n', '--num_epoch', type=int, default=50)
parser.add_argument('-neu', '--num_neurons', type=int,default = 400)
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
    model = CoordMLP(neurons=args.num_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=1e-4)

    startepoch, model, optimizer,scheduler = load_model(model, optimizer, scheduler,modelname, device)
    model = model.cuda()
    model.eval()
    allmodels.append(model)
import trimesh
import os
gtdata = decompressed_bytes_array1 = np.load("distancestestremesh.npz")['arr_0']/1.42

#scalegt = np.load("ScaleFactorEachMesh.npy")
#print(scalegt.shape)
#print(scalegt[:,None,None].shape)
#gtdata = gtdata*scalegt[:,None,None]
#np.save("distances.npy",gtdata)
normalization = np.max(gtdata.reshape(gtdata.shape[0], -1), axis=1)[:,None,None]
print("Normalization using Princeton protocol (not applied):", normalization.mean())
#gtdata = gtdata/normalization
meshpca = "Xpca_voxels_dense_rot_center_cleannoise_all_shapematch_0_25fil_norm.npy"
testdata = "TestDataShapeMatchSMALRemeshed.npz"#"VoxelsRepTestRemeshedShapeMatch.npz"#"UDFTestRemeshedShapeMatch.npz"#"
meshvertices = "MeshVerticesSMALProcessed.npz"
testidxname = "TestIndexPairs.npy"
#meshpca = np.load(meshpca)[np.load(testidxname)] 
#meshpca = np.load(testdata)['testpca']
from TNetModel import TNet, geodesic_loss, trace_loss, point_alignment_loss, gram_schmidt_orthogonalization
import torch
device = 'cuda'
tnet_model = TNet(k=3).cuda()
ckpt = torch.load(
    "/storage/tnet_model_weights_0.25699647267659503_0.0598757229745388_std_0.11098886281251907_2k.pth",
    map_location=device
)
tnet_model.load_state_dict(ckpt)
tnet_model.eval()


mean_coord =  np.array([ 1.1345062e-01 , 7.7768716e-05, -8.0925278e-02])[None,None,:]
#mean_coord =  np.array([ 0.09839194 ,-0.1014386  ,-0.00022806])[None,None,:]

#std_coord = np.array([0.32054427, 0.16493152, 0.0724903 ])[None,None,:]
std_coord = np.array([0.27175936, 0.10233479 ,0.204286])[None,None,:]

#mean_coord =  np.array([ 1.13937326e-01 , 5.98807965e-05 ,-8.12632143e-02])[None,None,:]
#mean_coord =  np.array([ 0.09839194 ,-0.1014386  ,-0.00022806])[None,None,:]

#std_coord = np.array([0.32054427, 0.16493152, 0.0724903 ])[None,None,:]
#std_coord = np.array([0.27238855, 0.10268788, 0.2048397 ])[None,None,:]

#np.array([0.34042523 ,0.1923675 , 0.07791027])[None,None,:]
#meshvertices = (np.load(meshvertices)['arr_0'][np.load(testidxname)]- mean_coord)/std_coord
meshvertices = np.load(testdata)['testmeshvertices']# - mean_coord)/std_coord
faces =  trimesh.load("MeshTestSample" + os.sep + "meshtest_6_remesh.off", process=False).faces
import utils
voxels_centers= np.load("/storage/VoxelsCenters_all_121resKaolin_TNetalign.npy")
indexinclude = np.load("/storage/IndexInclude_0_14stdfil_kaolin_TNetalign.npy")
meshpca = []
vox_center_include = voxels_centers[indexinclude][None]
vox_center_include_tensor = torch.from_numpy(voxels_centers[indexinclude][None]).float().cuda()
print(vox_center_include.shape)
import pickle
from sklearn.decomposition import PCA # Import PCA to have the class available
import random
# Define the path to your .pkl file
file_path = '/storage/pcaUDF_model_shapematch__0_14fil__TNetalign.pkl'

# Load the PCA model
try:
    with open(file_path, 'rb') as f:
        loaded_pca_model = pickle.load(f)
    print("PCA model loaded successfully!")
    print(f"Type of loaded object: {type(loaded_pca_model)}")

    # You can now use the loaded_pca_model
    # For example, to check some attributes
    if isinstance(loaded_pca_model, PCA):
        print(f"Number of components: {loaded_pca_model.n_components_}")
        #print(f"Explained variance ratio: {loaded_pca_model.explained_variance_ratio_}")
        pca_mean_np = loaded_pca_model.mean_
        pca_components_np = loaded_pca_model.components_

        pca_mean = torch.from_numpy(pca_mean_np).float().cuda()
        pca_components = torch.from_numpy(pca_components_np).float().cuda()
    else:
        print("The loaded object is not a scikit-learn PCA model.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
allgeod = []

import time
time_process_all = 0

for i in range(meshvertices.shape[0]):
    #noise_std = 0.01  # Adjust as needed
    #meshvertices[i] = meshvertices[i] + np.random.randn(*meshvertices[i].shape) * noise_std
    
    mesh_i = trimesh.Trimesh(vertices = meshvertices[i], faces = faces)
    #mesh_i_aniso = createanisomesh(list(mesh_i.vertices), list(mesh_i.faces))
    #verts_aniso = np.array(mesh_i_aniso.vertices)
    #faces_aniso = np.array(mesh_i_aniso.faces)
    #mesh_i.export("MeshTestSample" + os.sep + f"remeshed_befrotate_{i}.obj")
    vertices = torch.from_numpy(meshvertices[i]).float().cuda()
    faces_torch = torch.from_numpy(faces).cuda()
    num_faces = faces_torch.shape[0]

    # Randomly choose a proportion between 0 and 0.06
    remove_ratio = 0.4#random.uniform(0.0, 0.30)
    num_remove = int(remove_ratio * num_faces)

    # Randomly select indices to keep
    if num_remove == 0:
        faces_torch = faces_torch
    else:
        keep_indices = torch.randperm(num_faces, device=faces_torch.device)[num_remove:]
        faces_torch = faces_torch[keep_indices]
    torch.cuda.synchronize()
    s = time.time()
    #meshvertices[i] = utils.processmesh(meshvertices[i], faces)[0]
    #print(meshvertices[i].shape)
    #vertices, scale_factor, pc_standard = utils.processmesh_torch(vertices, faces_torch)
    vertices, scale_factor, pc_standard = utils.processmeshbroken_torch(vertices, faces_torch)
    #verts_aniso_process, scale = utils.processmesh(verts_aniso, faces_aniso)
    #print(scale)
    #meshvertices[i] = verts_aniso_process[:meshvertices[i].shape[0]]
    
    #pc_standard = (utils.sample_mesh(torch.from_numpy(meshvertices[i]).float().cuda(), torch.from_numpy(faces).cuda(), sample_count=8000))
    torch.cuda.synchronize()
    time_process_all += time.time() - s
    meshvertices[i] = vertices.cpu().numpy()
    #print(time_process_all / (i+1))
    #pc_standard = (utils.sample_mesh(torch.tensor(verts_aniso_process).double(), torch.tensor(faces_aniso), sample_count=8000))
    with torch.no_grad():
        pc_rotated, R_gt = utils.rotate_pc_random_withgt(pc_standard.cpu().numpy())
        meshvertices[i] = meshvertices[i] @ R_gt.numpy().T
        #print(pc_rotated.shape)
        gt_rotations_batch = (R_gt).float().to(device)[None]   # Shape: (B, 3, 3)
        pc_canonical = pc_standard.float().cuda()
        pc_canonical = pc_canonical[None]
        pc_rotated = pc_rotated.float().cuda()[None]
        vertices = torch.from_numpy(meshvertices[i]).float().cuda()
        # Forward pass: Get the 6D output from T-Net
        torch.cuda.synchronize()
        s=time.time()
        with torch.no_grad():
            raw_6d_output = tnet_model(pc_rotated) # Shape: (B, 6)

            # Convert 6D output to 3x3 rotation matrix using Gram-Schmidt
            predicted_rotations = gram_schmidt_orthogonalization(raw_6d_output) # Shape: (B, 3, 3)

            #point_clouds_batch =point_clouds_batch * std_point_clouds
            # Calculate geodesic loss
            #geodloss = geodesic_loss(predicted_rotations, gt_rotations_batch, L2=False,reduction=None)
            #print(geodloss)
            P_aligned = torch.bmm(pc_rotated, predicted_rotations)  # (B, N, 3)

            vertices = vertices @ predicted_rotations[0]

            #mesh_i = trimesh.Trimesh(vertices = meshvertices[i], faces = faces)
            #mesh_i.export("MeshTestSample" + os.sep + f"remeshed_{i}.obj")
            #P_aligned = P_aligned.cpu().numpy()
            #pc_i = trimesh.points.PointCloud(P_aligned[0])
            #pc_i.export("MeshTestSample" + os.sep + f"remeshed_{i}_pc.obj")
            #udf = utils.compute_udf(P_aligned[0], vox_center_include[0])
            udf = findUDF_from_PointCloud(P_aligned, vox_center_include_tensor)

            #meshpca.append(loaded_pca_model.transform(udf[None])[0])
            #meshpca.append(loaded_pca_model.transform(udf)[0])
            pca = pca_transform_torch(udf, pca_mean, pca_components)
        torch.cuda.synchronize()
        time_process_all += time.time() - s
        #print(time_process_all/(i+1))
        meshpca.append(pca.cpu().numpy()[0])
        meshvertices[i] = vertices.cpu().numpy()
        if(i < 50):
            time_process_all = 0 #Warm up period
        #allgeod.append(geodloss.cpu().numpy())
#print("Alignment mistake:",np.mean(allgeod), np.std(allgeod))
print("Time processing for one shape:" ,time_process_all/(meshvertices.shape[0] - 50))
meshpca = np.array(meshpca)
print(meshpca.shape)
#np.save("/storage/pcaUDF_Remeshed_TNetalign.npy", meshpca)
meshvertices = (meshvertices - mean_coord)/std_coord
#meshpca_ori = np.load("/storage/pcaUDFVertProcess_all_TNetalign.npy")[np.load(testidxname)]
#print("Cosine similarity with original PCA:", utils.computecosinesim(meshpca, meshpca_ori))
                       
        #aligned_pointcloud.append(P_aligned)
    

print("Meshvertices shape:",meshvertices.shape)
print("Building caches..")
caches = []
import time
####1. Build cache of data for quick access
s_cache = time.time()
for i in range(meshvertices.shape[0]):
    #caches.append(buildcache_3tiers(meshvertices[i],  tier1sample=60, tier1neigh=50, tier2sample=500, tier2neigh=56))
    caches.append(buildcache_3tiers(meshvertices[i],  tier1sample=60, tier1neigh=65, tier2sample=650, tier2neigh=60))
print("Time for one cache :", (time.time() - s_cache)/meshvertices.shape[0])
    


###2. Sample Random Pair of Meshes and vertex index and test.


matches = []   # will store (mesh1_idx, mesh2_idx, vertex1_idx, best_vertex2_idx) per batch
sources = np.load("SourcesGTRemeshed.npy")

import torch
NTEST = 10000
novertex = 100
rng = np.random.default_rng()
mesh1_indices  = np.empty(NTEST, dtype=np.int32)
mesh2_indices  = np.empty(NTEST, dtype=np.int32)
vertex1_indices = np.empty((NTEST,novertex), dtype=np.int32)
print("No Vertex:", meshvertices[0].shape[0])
print("preparing test data: ", NTEST, " samples")
for b in range(NTEST):
    m1 = b//len(meshpca)#rng.integers(0, len(meshpca))
    m2 = b%len(meshpca)#rng.integers(0, len(meshpca))
    v1 = rng.integers(low=0, high=sources.shape[0], size=novertex)
    start, end = b, (b+1)
    mesh1_indices [start:end] = m1
    mesh2_indices [start:end] = m2
    vertex1_indices[start:end] = v1

totaltime = totalmem = 0
with torch.no_grad():
    for i in tqdm(range(NTEST), mininterval=30):
        
        batch_i = i   # 0 for first batch, 1 for second, …
        mesh1_idx  = mesh1_indices[batch_i]
        mesh2_idx  = mesh2_indices[batch_i]
        vertex1_idx= sources[vertex1_indices[batch_i]]
        if(mesh1_idx == mesh2_idx):
            
            continue
        
        vertex1_indices_src = vertex1_indices[batch_i]
        #print(vertex1_indices_src.shape)
        pca1    = meshpca[mesh1_idx][None]             # (1,400,)
        pca2    = meshpca[mesh2_idx][None]             # (1,400,)
        vertex1 = meshvertices[mesh1_idx][vertex1_idx]    # (N,3,)
        #print(vertex1.shape)
        
        pca1 =  torch.from_numpy(pca1).float().cuda()
        pca2=   torch.from_numpy(pca2).float().cuda()
        vertex1= torch.from_numpy(vertex1).float().cuda()
        #print(meshvertices[mesh2_idx])
        initial_memory = torch.cuda.memory_allocated(device)
        torch.cuda.synchronize()
        s = time.time()
        best_idx = fast_locate_multi_vertex_in_Y_3tiered(
            vertex1, # (N, 3)
            pca1,         # (1, 400)
            pca2,         # (1, 400)
            caches[mesh2_idx][0],      #(N1, 3)
            caches[mesh2_idx][1], # (N2, 3)
            caches[mesh2_idx][2], # (N3, 3)
            caches[mesh2_idx][3],   #(N1, M1)
            caches[mesh2_idx][4], # (N2, M2)
            
            allmodels
        )
        torch.cuda.synchronize()
        totaltime += time.time() - s
        
        
        

       
        # find the index of the minimum predicted distance
        best_idx = best_idx.cpu().numpy()   # integer in [0, 3889]

        
        

        # record the matching vertex2 index
        matches.append((mesh1_idx,
                        mesh2_idx,
                        vertex1_indices_src,
                        best_idx))
error = 0.0
allerror = []
for mesh1_idx, mesh2_idx,vertex1_indices_src, best_idx in matches:
    error += np.mean(gtdata[mesh2_idx,vertex1_indices_src,best_idx])
    allerror.extend( gtdata[mesh2_idx,vertex1_indices_src,best_idx] )
allerror = np.array(allerror)
print("Error : ", error/len(matches))
print(np.sum(allerror < 0.2)/(novertex*len(matches)))
allerror = np.array(allerror, dtype=np.float64)
print("Std: ", np.std(allerror))
print("Time:",totaltime/len(matches))

auc = auc_at_threshold(allerror,max_threshold=0.2)
print("AUC below 0.2:",auc)








