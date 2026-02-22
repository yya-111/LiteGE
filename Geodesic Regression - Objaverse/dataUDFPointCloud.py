import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

import numpy as np
import random
import os
from tqdm import tqdm
class MCAGeodesicData(Dataset):
    def _compute_euclidean(self, sources, dests, chunk_size=100000):
        print("compute euclidean")
        distances = []
        for i in tqdm(range(0, len(sources), chunk_size)):
            chunk = np.linalg.norm(sources[i:i+chunk_size] - dests[i:i+chunk_size], axis=1)
            distances.append(chunk)
        distances = np.concatenate(distances)[:, np.newaxis]
        print(distances.dtype)
        #np.savez_compressed(name, distances.astype(np.float32))
        return distances

    

    def _load_and_concatenate_data(self, npz_data_path):
        """
        Loads data from specified .npz files and concatenates arrays.
        """
        # Ensure npz_data_path is always a list for consistent iteration
        if isinstance(npz_data_path, str):
            npz_data_path = [npz_data_path]
        elif not isinstance(npz_data_path, list):
            print("Error: npz_data_path must be a string or a list of strings. No data loaded.")
            return

        # Initialize lists to hold data from multiple files before concatenation
        temp_sources = []
        temp_dests = []
        temp_dist_on_a = []
        temp_dist_on_b = []
        temp_mesh_a = []
        temp_mesh_b = []

        # Flags to ensure mesh_a and mesh_b are loaded only once (assuming they are static)
        mesh_a_loaded = False
        mesh_b_loaded = False

        for path_idx, path in enumerate(npz_data_path):
            if not os.path.exists(path):
                print(f"Warning: File not found at '{path}'. Skipping this file.")
                continue

            try:
                with np.load(path) as data:
                    print(f"\n--- Loading data from: '{path}' ---")

                    # Load mesh_a (only once from the first file)
                    if 'mesh_a' in data:
                        temp_mesh_a.append(data['mesh_a'])
                        #if not mesh_a_loaded:
                            #self.mesh_a = data['mesh_a']
                            #mesh_a_loaded = True
                            #print(f"Loaded 'mesh_a' with shape: {self.mesh_a.shape}")
                        # Optional: Add a check here if you want to verify consistency of mesh_a across files
                    else:
                        print("'mesh_a' not found in the .npz file.")

                    # Load mesh_b (only once from the first file)
                    

                    # Process and append 'source' data
                    if 'source' in data:
                        # Note: Corrected to use 'source' key as per your description.
                        temp_sources.append((data['source'] ) )
                        print(f"Appended 'source' from '{path}' with shape: {data['source'].shape}")
                    else:
                        print("'source' not found in the .npz file.")

                    # Process and append 'dest' data
                    if 'dest' in data:
                        # Note: Corrected to use 'dest' key as per your description.
                        temp_dests.append(data['dest'] )
                        print(f"Appended 'dest' from '{path}' with shape: {data['dest'].shape}")
                    else:
                        print("'dest' not found in the .npz file.")

                    # Process and append 'dist_on_a' data
                    if 'dist_on_a' in data:
                        # Note: Corrected to use 'dist_on_a' key as per your description.
                        temp_dist_on_a.append(1.42 * data['dist_on_a'][:, None])
                        #print(np.mean(data['dist_on_a']))
                        #print(np.mean(1.42 * data['dist_on_a']))
                        print(f"Appended 'dist_on_a' from '{path}' with shape: {data['dist_on_a'].shape}")
                    else:
                        print("'dist_on_a' not found in the .npz file.")

                    

            except Exception as e:
                print(f"Error loading data from '{path}': {e}. Skipping this file.")
                continue

        # Concatenate all collected data if lists are not empty
        print("\n--- Concatenating collected data ---")
        if temp_mesh_a:
            self.mesh_a = np.concatenate(temp_mesh_a, axis=0)
            print(f"Concatenated 'mesh_a' total shape: {self.mesh_a.shape}, {self.mesh_a.dtype}")
        else:
            print("No 'mesh_a' data found across all files.")
        
            
        if temp_sources:
            self.sources = np.concatenate(temp_sources, axis=0)
            print(f"Concatenated 'sources' total shape: {self.sources.shape}, {self.sources.dtype}")
        else:
            print("No 'sources' data found across all files.")

        if temp_dests:
            self.dests = np.concatenate(temp_dests, axis=0)
            print(f"Concatenated 'dests' total shape: {self.dests.shape}, {self.dests.dtype}")
        else:
            print("No 'dests' data found across all files.")

        if temp_dist_on_a:
            self.dist_on_a = np.concatenate(temp_dist_on_a, axis=0)
            print(f"Concatenated 'dist_on_a' total shape: {self.dist_on_a.shape}, {self.dist_on_a.dtype}")
            
        else:
            print("No 'dist_on_a' data found across all files.")

        
        del temp_dist_on_a, temp_dests, temp_sources,temp_mesh_a

    def __init__(self, npz_meshvertices_scale, npz_data_path, pca_path):
        """
        Initializes the data loader and loads/concatenates data from NPZ files.

        Args:
            npz_data_path (str or list): A single path string or a list of path strings to the .npz files.
            mean_coord (numpy.ndarray): Mean coordinate for normalization.
            std_coord (numpy.ndarray): Standard deviation coordinate for normalization.
        """
        #meanpca = np.load("MEANPCAUDF.npy")
        #stdpca = np.load("STDPCAUDF.npy")
        self.mesh_rep = (np.load(pca_path))#*stdpca + meanpca
        self.mesh_rep = self.mesh_rep[:,:240]
        #trainindex = np.load("TrainIndexPairs.npy")
        #trainindex = trainindex[np.where(trainindex< 12000)]
        #print("Mean PCA features:",self.mesh_rep[trainindex].mean())
        #print("STD PCA features:",self.mesh_rep[trainindex].std())
        #Normalize whole data PCA, the mean is already 0.0
        #self.mesh_rep /= 0.49869213
        
        #mean_coord = np.array([ 0.16173247, -0.05248954,  0.00203621])[None,None,:]
         
        
        mean_coord = np.array([  0.00076804, -0.00342962,  0.00045894] )[None,None,:]
        std_coord = np.array([0.34962583, 0.19903275, 0.11451308])[None,None,:]
        #std_coord = np.array([0.34033716, 0.19236997, 0.07788278])[None,None,:]
        
        self.mesh_a = None
        #self.mesh_b = None
        self.sources = None
        self.dests = None
        self.dist_on_a = None
        #self.dist_on_b = None
        
        self._load_and_concatenate_data(npz_data_path)
        
        self.valid_indices_inallarray = np.load(npz_meshvertices_scale, allow_pickle=True)['valid_indices'] 
        orig_to_valid = {orig_idx: i for i, orig_idx in enumerate(self.valid_indices_inallarray)}

        scalefactor = np.load(npz_meshvertices_scale, allow_pickle=True)['scale']
        
        #scalefactorfromunitmesh = np.load(scalefactorfromunitmesh_path)
        # Build mapping original -> valid
        #orig_to_valid = {orig: i for i, orig in enumerate(valid_indices)}

        # Identify which entries are valid
        mask_valid = np.array([x in orig_to_valid for x in self.mesh_a])

        # Filter everything consistently
        self.mesh_a = self.mesh_a[mask_valid]
        self.dist_on_a = self.dist_on_a[mask_valid]
        self.sources = self.sources[mask_valid]
        self.dests = self.dests[mask_valid]

        # Remap
        self.mesh_a = np.array([orig_to_valid[x] for x in self.mesh_a])
        scaling_dist = scalefactor[self.mesh_a].astype(np.float32)
        print(scaling_dist.shape)
        print(self.dist_on_a.shape, self.dist_on_a.dtype)
        self.dist_on_a = self.dist_on_a * scaling_dist[:,None]

        
        self.vertices = np.load(npz_meshvertices_scale, allow_pickle=True)['verts_ori']
        
        
        #mesh_a_sources = self.vertices[self.mesh_a, self.sources]
        #mesh_a_dests = self.vertices[self.mesh_a, self.dests]
        verts_all = self.vertices  # list of (Ki,3)
        mesh_idxs = self.mesh_a
        source_idxs = self.sources
        dest_idxs = self.dests

        mesh_a_sources = np.empty((len(self.mesh_a), 3), dtype=np.float32)
        mesh_a_dests   = np.empty((len(self.mesh_a), 3), dtype=np.float32)

        for i, (m, s, d) in enumerate(zip(self.mesh_a, self.sources, self.dests)):
            verts = verts_all[m]           # local reference → faster
            mesh_a_sources[i] = verts[s]
            mesh_a_dests[i]   = verts[d]

        mesh_a_euclid = 1.42 * self._compute_euclidean(mesh_a_sources, mesh_a_dests)
        print(mesh_a_euclid.shape)
        del mesh_a_sources, mesh_a_dests
        finite_mask = np.isfinite(self.dist_on_a)
        mean_dist = np.mean(self.dist_on_a[finite_mask])
        median_dist = np.median(self.dist_on_a[finite_mask])
        
        print("Mean distance :", mean_dist)
        print("Median distance :", median_dist)
        
        self.dist_on_a -= mesh_a_euclid
        print("Strange distance below euclidean:", np.sum(self.dist_on_a < -0.01))
        
        #del mesh_a_euclid
        
        
        print("Minimum distances on mesh a:",np.min(self.dist_on_a), self.dist_on_a.shape)
        finite_mask = np.isfinite(self.dist_on_a)
        mean_dist = np.mean(self.dist_on_a[finite_mask])
        median_dist = np.median(self.dist_on_a[finite_mask])
        #print("Mean distance after subtracting euclidean distance:", mean_dist)
        #print("Median distance after subtracting euclidean distance:", median_dist)

        #print("Minimum distances on mesh b:",np.min(self.dist_on_b), self.dist_on_b.shape)
        self.dist_on_a = np.clip(self.dist_on_a , a_min=0, a_max=None)
        #self.dist_on_b = np.clip(self.dist_on_b , a_min=0, a_max=None)
        
        
        
        
        #self._compute_source_dest_statistics()
        #print(self.mean_coord.shape)
        #print(self.std_coord.shape)
        print(self.vertices.shape)
        mean = mean_coord.reshape(3,)
        std = std_coord.reshape(3,)

        #Euclid substract
        #self.vertices = (self.vertices - mean_coord)/std_coord
        for i in range(len(self.vertices)):
            self.vertices[i] = (self.vertices[i] - mean) / std


        
        #self.getstats()
        

        
        #self.offset = offset
    def __len__(self):
        """
        Your code here
        """
        #print(self.sourcespoints.shape[0]*self.sourcespoints.shape[1])
        return self.mesh_a.shape[0]#(self.sourcespoints.shape[0]*self.sourcespoints.shape[1])

    def __getitem__(self, idx):
        #idx = np.random.randint(0, self.dist_on_a.shape[0])
        dist = self.dist_on_a[idx].astype(np.float32)

        while not np.isfinite(dist):
            idx = np.random.randint(0, self.dist_on_a.shape[0])
            dist = self.dist_on_a[idx].astype(np.float32)

        # Now idx is valid
        mesh_id = self.mesh_a[idx]

        source = self.vertices[mesh_id][self.sources[idx]].astype(np.float32)
        dest   = self.vertices[mesh_id][self.dests[idx]].astype(np.float32)

        return (
            self.mesh_rep[mesh_id].astype(np.float32),
            source,
            dest,
            dist
        )
        
        
        #source = self.vertices[self.mesh_a[idx]][self.sources[idx]].astype(np.float32)
        #dest = self.vertices[self.mesh_a[idx]][self.dests[idx]].astype(np.float32)
        #dist = self.dist_on_a[idx].astype(np.float32)  
        
        #return self.mesh_rep[self.mesh_a[idx]].astype(np.float32), source.astype(np.float32),dest.astype(np.float32) , self.dist_on_a[idx].astype(np.float32)    
        
        
    
    def _compute_source_dest_statistics(self, N_samples=10000000):
        """
        Computes the mean and standard deviation of source and destination coordinates
        by randomly sampling N_samples from the loaded data.

        Args:
            N_samples (int): Number of random samples to use for statistics computation.
                             Defaults to 10,000.
        """
        # Check if all necessary data is available
        if self.sources is None or self.dests is None or \
           self.mesh_a is None or self.mesh_b is None or \
           self.vertices is None:
            print("Cannot compute statistics: Required data (sources, dests, mesh_a, mesh_b, vertices) is not fully loaded or provided.")
            return

        total_samples = self.sources.shape[0]
        if total_samples == 0:
            print("No source/destination samples available to compute statistics.")
            return

        # Ensure N_samples does not exceed the total available samples
        N_samples = min(N_samples, total_samples)
        
        print(f"\n--- Computing statistics on {N_samples} randomly sampled coordinates ---")

        # Generate N_samples random indices
        # 'replace=False' ensures unique samples
        sampled_indices = np.random.choice(total_samples, N_samples, replace=False)

        sampled_coords_list = []
        #sampled_dest_coords_list = []

        for idx in sampled_indices:
            try:
                # Get the mesh index for this sample
                mesh_a_idx = self.mesh_a[idx]
                mesh_b_idx = self.mesh_b[idx]

                # Get the vertex index within that mesh for this sample
                source_vertex_idx = self.sources[idx]
                dest_vertex_idx = self.dests[idx]

                # --- Safety checks for valid indices ---
                if not (0 <= mesh_a_idx < len(self.vertices)):
                    print(f"Warning: mesh_a_idx {mesh_a_idx} for sample {idx} out of bounds for self.vertices (max {len(self.vertices)-1}). Skipping source coordinate for this sample.")
                    continue # Skip this sample for source
                if not (0 <= mesh_b_idx < len(self.vertices)):
                    print(f"Warning: mesh_b_idx {mesh_b_idx} for sample {idx} out of bounds for self.vertices (max {len(self.vertices)-1}). Skipping dest coordinate for this sample.")
                    continue # Skip this sample for destination

                # Get the actual vertex coordinates using the provided indexing:
                # self.vertices[mesh_index][vertex_index_within_mesh]
                source_coord = self.vertices[mesh_a_idx][source_vertex_idx]
                dest_coord = self.vertices[mesh_b_idx][dest_vertex_idx]

                sampled_coords_list.append(source_coord)
                sampled_coords_list.append(dest_coord)

            except IndexError as e:
                print(f"Error indexing for sample {idx}: {e}. This might mean vertex indices in .npz files are out of bounds for the selected mesh's vertex array. Skipping this sample.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred for sample {idx}: {e}. Skipping this sample.")
                continue

        

        sampled_source_coords = np.array(sampled_coords_list)
        

        # Compute mean and standard deviation across all sampled coordinates (axis=0 for features)
        self.mean_coord = np.mean(sampled_source_coords, axis=0)
        self.std_coord = np.std(sampled_source_coords, axis=0)
        #self.mean_dest_coord = np.mean(sampled_dest_coords, axis=0)
        #self.std_dest_coord = np.std(sampled_dest_coords, axis=0)

        print(f"Computed Mean Coordinates: {self.mean_coord}")
        print(f"Computed Std Coordinates: {self.std_coord}")
        #print(f"Computed Mean Destination Coordinates: {self.mean_dest_coord}")
        #print(f"Computed Std Destination Coordinates: {self.std_dest_coord}")

    
    

def load_data(npz_meshvertices, npz_data_path, pca_path,  num_workers=0, batch_size=1024, offset=0,**kwargs):
    dataset = MCAGeodesicData(npz_meshvertices, npz_data_path, pca_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

