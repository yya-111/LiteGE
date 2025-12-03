import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MeshCorrespondenceDataset(Dataset):
    def __init__(self,
                 meshpca: str,        # (101, 400)
                 
                 meshvertices: str,   # (101, 3889, 3)
                 valindex: str,
                 n_batches: int,
                 batch_size: int = 6890,
                 ):
        
        
        mean_coord = np.array([-2.3920309e-06 , 2.6345462e-05 , 2.0604899e-05])[None,None,:]
        std_coord = np.array([0.32054427, 0.16493152, 0.0724903 ])[None,None,:]
        print(mean_coord.shape, std_coord.shape)
        self.meshpca = (np.load(meshpca)[np.load(valindex)] )
        
        
        
        self.meshvertices = (np.load(meshvertices)['arr_0'][np.load(valindex)]- mean_coord)/std_coord
        print(self.meshvertices.shape)
        #self.meshvertices = self.meshvertices[np.load(valindex)]
        #self.meshvertices = (self.meshvertices 
        assert self.meshvertices.shape[1] == batch_size
        self.B = batch_size
        self.n_batches = n_batches
        self.N = self.B * self.n_batches

        # Precompute the “batch‑granular” indices:
        rng = np.random.default_rng()
        self.mesh1_indices  = np.empty(self.n_batches, dtype=np.int64)
        self.mesh2_indices  = np.empty(self.n_batches, dtype=np.int64)
        self.vertex1_indices = np.empty(self.n_batches, dtype=np.int64)

        for b in range(self.n_batches):
            m1 = rng.integers(0, len(self.meshpca))
            m2 = rng.integers(0, len(self.meshpca))
            v1 = rng.integers(0, self.B)
            start, end = b, (b+1)
            self.mesh1_indices [start:end] = m1
            self.mesh2_indices [start:end] = m2
            self.vertex1_indices[start:end] = v1
        print(self.mesh1_indices)
        print(self.mesh2_indices)
        print(self.vertex1_indices)
    def __len__(self):
        return self.N

    def __getitem__(self, i):
        # which batch‑slot within [0..B-1]?
        slot = i % self.B
        uniformidx = i // self.B
        m1 = self.mesh1_indices[uniformidx]
        m2 = self.mesh2_indices[uniformidx]
        v1 = self.vertex1_indices[uniformidx]
        v2 = slot                  # we want vertices 0..B-1 in order

        # fetch data
        pca1    = self.meshpca[m1]             # (400,)
        pca2    = self.meshpca[m2]             # (400,)
        vertex1 = self.meshvertices[m1, v1]    # (3,)
        vertex2 = self.meshvertices[m2, v2]    # (3,)

        return {
            'pca1':    torch.from_numpy(pca1).float(),
            'pca2':    torch.from_numpy(pca2).float(),
            'vertex1': torch.from_numpy(vertex1).float(),
            'vertex2': torch.from_numpy(vertex2).float(),
        }

# ——— Usage ———

# load your arrays:
# meshpca      = np.load("meshpca.npy")       # (101,400)
# meshvertices = np.load("meshvertices.npy")  # (101,6890,3)

#dataset = MeshCorrespondenceDataset(meshpca, meshvertices,
#                                    batch_size=6890,
#                                    n_batches=100)

#loader = DataLoader(dataset,
#                    batch_size=6890,   # yields exactly N_batches batches
#                    shuffle=False,     # keep our precomputed grouping
#                    num_workers=8)     # parallel workers

#for batch in loader:
    # batch['pca1'].shape   == (6890, 400)
    # batch['vertex2'].shape== (6890, 3)
    # and within this batch mesh1, mesh2, vertex1 are constant
    # … do your forward pass …
#    pass
