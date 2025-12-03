import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

import numpy as np
import random
import os
class MCAGeodesicData(Dataset):
    

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
                    if 'mesh_b' in data:
                        temp_mesh_b.append(data['mesh_b'])
                        #if not mesh_b_loaded:
                        #    self.mesh_b = data['mesh_b']
                        #    mesh_b_loaded = True
                        #    print(f"Loaded 'mesh_b' with shape: {self.mesh_b.shape}")
                        # Optional: Add a check here if you want to verify consistency of mesh_b across files
                    else:
                        print("'mesh_b' not found in the .npz file.")

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
                        print(np.mean(1.42 * data['dist_on_a'][:, None] < 0.17))
                        print(f"Appended 'dist_on_a' from '{path}' with shape: {data['dist_on_a'].shape}")
                    else:
                        print("'dist_on_a' not found in the .npz file.")

                    # Process and append 'dist_on_b' data
                    if 'dist_on_b' in data:
                        # Note: Corrected to use 'dist_on_b' key as per your description.
                        temp_dist_on_b.append(1.42 * data['dist_on_b'][:, None])
                        print(f"Appended 'dist_on_b' from '{path}' with shape: {data['dist_on_b'].shape}")
                    else:
                        print("'dist_on_b' not found in the .npz file.")

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
        if temp_mesh_b:
            self.mesh_b = np.concatenate(temp_mesh_b, axis=0)
            print(f"Concatenated 'mesh_b' total shape: {self.mesh_b.shape}, {self.mesh_b.dtype}")
        else:
            print("No 'mesh_b' data found across all files.")
            
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

        if temp_dist_on_b:
            self.dist_on_b = np.concatenate(temp_dist_on_b, axis=0)
            print(f"Concatenated 'dist_on_b' total shape: {self.dist_on_b.shape}, {self.dist_on_b.dtype}")
        else:
            print("No 'dist_on_b' data found across all files.")
        del temp_dist_on_a, temp_dist_on_b, temp_dests, temp_sources,temp_mesh_a,temp_mesh_b

    def __init__(self, npz_meshvertices, npz_data_path, pca_path, scalefactor_path, forbidden_vert_index=["/storage/failed_above_0.66.npy"]):
        """
        Initializes the data loader and loads/concatenates data from NPZ files.

        Args:
            npz_data_path (str or list): A single path string or a list of path strings to the .npz files.
            mean_coord (numpy.ndarray): Mean coordinate for normalization.
            std_coord (numpy.ndarray): Standard deviation coordinate for normalization.
        """
        
        mean_coord =  np.array([ 1.1345062e-01 , 7.7768716e-05, -8.0925278e-02])[None,None,:]
        #mean_coord =  np.array([ 0.09839194 ,-0.1014386  ,-0.00022806])[None,None,:]

        #std_coord = np.array([0.32054427, 0.16493152, 0.0724903 ])[None,None,:]
        std_coord = np.array([0.27175936, 0.10233479 ,0.204286])[None,None,:]
        
        self.mesh_a = None
        self.mesh_b = None
        self.sources = None
        self.dests = None
        self.dist_on_a = None
        self.dist_on_b = None
        self.mesh_idx_allow = np.ones(12000).astype(bool)
        for index_file in forbidden_vert_index:
            forbidindex = np.load(index_file)
            forbidindex = forbidindex[np.where(forbidindex < 12000)]
            print("Forbidden index:", forbidindex.shape)
            self.mesh_idx_allow[forbidindex] = False
        print("Total allowed index:", np.sum(self.mesh_idx_allow))
        
        self._load_and_concatenate_data(npz_data_path)
        
        
        scalefactor = np.load(scalefactor_path)

        scale_a  = scalefactor[self.mesh_a]
        scale_b = scalefactor[self.mesh_b]
        print("Scale a shape:",scale_a.shape)
        self.dist_on_a = self.dist_on_a *scale_a[:,None]
        self.dist_on_b = self.dist_on_b*scale_b[:,None]
        print("Distance mean:",np.mean(self.dist_on_a), np.mean(self.dist_on_b))

        self.vertices = np.load(npz_meshvertices)['arr_0']
        #self._compute_source_dest_statistics()
        #print(self.mean_coord.shape)
        #print(self.std_coord.shape)
        print(self.vertices.shape)
        self.vertices = (self.vertices - mean_coord)/std_coord

        self.mesh_rep = (np.load(pca_path))
        #self.getstats()
        

        
        #self.offset = offset
    def __len__(self):
        """
        Your code here
        """
        #print(self.sourcespoints.shape[0]*self.sourcespoints.shape[1])
        return self.mesh_a.shape[0]#(self.sourcespoints.shape[0]*self.sourcespoints.shape[1])

    def __getitem__(self, idx):
        
        # return self.data[idx]
        #import random # This import is necessary for random.randint

# ... (rest of your existing code before this snippet) ...

        mesh_a_i = self.mesh_a[idx]
        mesh_b_i = self.mesh_b[idx]

        while(self.mesh_idx_allow[mesh_a_i] == False or self.mesh_idx_allow[mesh_b_i] == False):
            # Assuming self.mesh_a is a NumPy array or list,
            # its length will define the upper bound for the random index
            idx = random.randint(0, len(self.mesh_a) - 1)
            mesh_a_i = self.mesh_a[idx]
            mesh_b_i = self.mesh_b[idx]
        if(random.random() > 0.5):
            source = self.vertices[self.mesh_a[idx]][self.sources[idx]]
            dest = self.vertices[self.mesh_b[idx]][self.dests[idx]]
        else:
            source = self.vertices[self.mesh_a[idx]][self.dests[idx]]
            dest = self.vertices[self.mesh_b[idx]][self.sources[idx]]
        return self.mesh_rep[self.mesh_a[idx]].astype(np.float32), self.mesh_rep[self.mesh_b[idx]].astype(np.float32), \
        source.astype(np.float32),dest.astype(np.float32) , self.dist_on_a[idx].astype(np.float32), \
        self.dist_on_b[idx].astype(np.float32)
    
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

    
    

def load_data(npz_meshvertices, npz_data_path, pca_path, scalefactor_path ,num_workers=0, batch_size=1024, offset=0,**kwargs):
    dataset = MCAGeodesicData(npz_meshvertices, npz_data_path, pca_path, scalefactor_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

