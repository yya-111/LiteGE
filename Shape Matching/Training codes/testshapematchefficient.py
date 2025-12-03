import torch
from CoordMLP import CoordMLP, save_model, load_model
#from datatest import MeshCorrespondenceDataset
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
import lzma
from buildvertexcache import buildcache, locate_corresponding_vertex_in_Y
from buildvertexcache import fast_locate_corresponding_vertex_in_Y, fast_locate_multi_vertex_in_Y
import numpy as np
from sklearn import metrics

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

gtdata = decompressed_bytes_array1 = decompress_lzma_to_numpy("distances.npy.lzma") #"distancestestremesh.npy.lzma")
if decompressed_bytes_array1:
#     # Assuming original dtype was float16 and shape (101, 2000, 6890)
    original_dtype_array1 = np.float16
    original_shape_array1 = (100, 2000, 3889)#(100,2700,5356)
    gtdata = np.frombuffer(gtdata, dtype=original_dtype_array1).reshape(original_shape_array1)

    print(f"Successfully reconstructed gtdata with shape {gtdata.shape}")

#scalegt = np.load("ScaleFactorEachMesh.npy")
#print(scalegt.shape)
#print(scalegt[:,None,None].shape)
#gtdata = gtdata*scalegt[:,None,None]
#np.save("distances.npy",gtdata)
normalization = np.max(gtdata.reshape(gtdata.shape[0], -1), axis=1)[:,None,None]
print("Normalization using Princeton protocol (not applied):", normalization.mean())
#gtdata = gtdata/normalization
#meshpca = "/storage/pcaUDFMeshVertProcess_all_notnorm.npy" #"/storage/pca_std_0_25fil_rot_center_scale_voxels_norm.npy"
meshpca = "/storage/pcaUDFVertProcess_all_TNetalign.npy"
testdata = "VoxelsRepTestRemeshedShapeMatch.npz"#"UDFTestRemeshedShapeMatch.npz"#"TestDataShapeMatchSMALRemeshed.npz"
meshvertices = "/storage/MeshVerticesAligned.npz"#"MeshVerticesSMALProcessed.npz"
testidxname = "TestIndexPairs.npy"
meshpca = np.load(meshpca)[np.load(testidxname)] 
#meshpca = np.load(testdata)['testpca']

mean_coord =  np.array([ 1.1345062e-01 , 7.7768716e-05, -8.0925278e-02])[None,None,:]
#mean_coord =  np.array([ 0.09839194 ,-0.1014386  ,-0.00022806])[None,None,:]

#std_coord = np.array([0.32054427, 0.16493152, 0.0724903 ])[None,None,:]
std_coord = np.array([0.27175936, 0.10233479 ,0.204286])[None,None,:]

meshvertices = (np.load(meshvertices)['arr_0'][np.load(testidxname)]- mean_coord)/std_coord
#meshvertices = (np.load(testdata)['testmeshvertices'] - mean_coord)/std_coord
print("Meshvertices shape:",meshvertices.shape)
print("Building caches..")
caches = []
for i in range(meshvertices.shape[0]):
    caches.append(buildcache(meshvertices[i]))
    
####1. Build cache of data for quick access

###2. Sample Random Pair of Meshes and vertex index and test.


matches = []   # will store (mesh1_idx, mesh2_idx, vertex1_idx, best_vertex2_idx) per batch
sources = np.load("SourcesinGTTest.npy")#np.load("SourcesGTRemeshed.npy")

NTEST = 100000
rng = np.random.default_rng()
mesh1_indices  = np.empty(NTEST, dtype=np.int32)
mesh2_indices  = np.empty(NTEST, dtype=np.int32)
vertex1_indices = np.empty(NTEST, dtype=np.int32)
print("No Vertex:", meshvertices[0].shape[0])
print("preparing test data: ", NTEST, " samples")
for b in range(NTEST):
    m1 = rng.integers(0, len(meshpca))
    m2 = rng.integers(0, len(meshpca))
    v1 = rng.integers(0, meshvertices[0].shape[0])
    start, end = b, (b+1)
    mesh1_indices [start:end] = m1
    mesh2_indices [start:end] = m2
    vertex1_indices[start:end] = v1

with torch.no_grad():
    for i in tqdm(range(NTEST), mininterval=30):
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
        mesh2_idx  = mesh2_indices[batch_i]
        vertex1_idx= vertex1_indices[batch_i]
        if(vertex1_idx not in sources or mesh1_idx == mesh2_idx):
            #print(vertex1_idx)
            continue
        
        vertex1_indices_src = np.where(sources == vertex1_idx)[0]
        #print(vertex1_indices_src)
        pca1    = meshpca[mesh1_idx][None]             # (1,400,)
        pca2    = meshpca[mesh2_idx][None]             # (1,400,)
        vertex1 = meshvertices[mesh1_idx, vertex1_idx][None]    # (1,3,)
        
        pca1 =  torch.from_numpy(pca1).float().cuda()
        pca2=   torch.from_numpy(pca2).float().cuda()
        vertex1= torch.from_numpy(vertex1).float().cuda().repeat(10,1)
        #print(meshvertices[mesh2_idx])
        
        best_idx = fast_locate_multi_vertex_in_Y(
            vertex1, # (1, 3)
            pca1,         # (1, 400)
            pca2,         # (1, 400)
            caches[mesh2_idx][0],      # (200, 3)
            caches[mesh2_idx][1], # (200, 100, 3)
            caches[mesh2_idx][2], # (200, 100)
            allmodels
        )[0]
        
        # find the index of the minimum predicted distance
        best_idx = best_idx.cpu().item()   # integer in [0, 3889]

        
        

        # record the matching vertex2 index
        matches.append((mesh1_idx,
                        mesh2_idx,
                        vertex1_indices_src,
                        best_idx))
error = 0.0
allerror = []
for mesh1_idx, mesh2_idx,vertex1_indices_src, best_idx in matches:
    error += gtdata[mesh2_idx][vertex1_indices_src[0]][best_idx] 
    allerror.append( gtdata[mesh2_idx][vertex1_indices_src[0]][best_idx] )
allerror = np.array(allerror)
print("Error : ", error/len(matches))
print(np.sum(allerror < 0.2)/len(matches))
print("Std: ", np.std(allerror))
auc = auc_at_threshold(allerror)
print("AUC below 0.1:",auc)






