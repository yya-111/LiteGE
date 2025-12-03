import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, norm=False):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(nn.BatchNorm1d(hidden_dim))  # Keep BatchNorm1d
        layers.append(nn.ReLU())
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        self.norm = norm
            
        self.mlp = nn.Sequential(*layers)
    def normalizeinputpca(self, pca_vector):
        return F.normalize(pca_vector, dim=-1)
    
    def forward(self, x):
        if(self.norm):
            x = self.normalizeinputpca(x)
            return self.normalizeinputpca(self.mlp(x))
        else:
            return self.mlp(x)

class MLPNOBN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        #layers.append(nn.BatchNorm1d(hidden_dim))  # Keep BatchNorm1d
        layers.append(nn.ReLU())
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            #layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class InputMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(InputMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim,bias=False))
        layers.append(nn.BatchNorm1d(hidden_dim))  # Keep BatchNorm1d
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim,bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Keep BatchNorm1d
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        B, L, F = x.shape  # (Batch, 2, 3) → (B, 2, 3)
        x = x.view(B * L, F)  # Flatten across batch (B*2, 3)
        #print(x.shape)
        x = self.mlp(x)  # Pass through MLP (B*2, hidden_dim)
        x = x.view(B, L, -1)  # Reshape back to (B, 2, hidden_dim)
        return x
    def forward_onedim(self,x):
        return self.mlp(x)

    
class SplitMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SplitMLP, self).__init__()
        self.mlp1 = MLP(input_dim//4, hidden_dim, num_layers)
        self.mlp2 = MLP(input_dim//4, hidden_dim, num_layers)
        self.mlp3 = MLP(input_dim//4, hidden_dim, num_layers)
        self.mlp4 = MLP(input_dim//4, hidden_dim, num_layers)
        self.lambda_param_2 = nn.Parameter(torch.rand(1))
        self.lambda_param_3 = nn.Parameter(torch.rand(1))
        self.lambda_param_4 = nn.Parameter(torch.rand(1))
    
    def forward(self, v):
        #print(v1.shape)
        chunks = torch.chunk(v, 4, dim=1)
        out1 = self.mlp1(chunks[0])
        #print(out1.shape)
        #print(v2.shape)
        out2 = self.mlp2(chunks[1])
        out3 = self.mlp3(chunks[2])
        out4 = self.mlp4(chunks[3])
        lambda_val_2 = torch.sigmoid(self.lambda_param_2)
        lambda_val_3 = torch.sigmoid(self.lambda_param_3)
        lambda_val_4 = torch.sigmoid(self.lambda_param_4)
        return out1 + lambda_val_2 * out2 + lambda_val_3 * out3 + lambda_val_4 * out4

class CoordMLP(nn.Module):
    def __init__(self, neurons=200, no_layers=3, embedding_size=400):
        super(CoordMLP, self).__init__()
        self.mlp_coord = InputMLP(3, neurons, no_layers-1)
        self.mlp_pca = MLP(embedding_size, neurons, no_layers)#SplitMLP(embedding_size, neurons, no_layers)
        self.mlp_pcacoord = MLP(neurons, neurons,no_layers-1)
        self.project_down_pcacoord = torch.nn.Linear(2*neurons,neurons)
        self.project_embed = torch.nn.Linear(neurons,neurons)
        self.final_linear = torch.nn.Linear(2*neurons, 1)
        self.final_layers = MLP(neurons, 2*neurons,no_layers - 1)
        self.embedding_size = embedding_size
    
    def forward_distonly(self, sourcesembed, destsembed):
        final_layer = self.final_layers(sourcesembed - destsembed)
        output = self.final_linear(final_layer)
        return output.abs()
    
    def forward_multi_sources_alldest(self, sourcescoords, destscoords, meshsrcpca, meshdestpca):
        # shapes:
        # sourcescoords: [N, 3]
        # destscoords: [B, 3]
        # meshsrcpca, meshdestpca: [1, 400]
        # return BN x 3
        N = sourcescoords.shape[0]
        B = destscoords.shape[0]
        #meshsrcpca, meshdestpca = self.normalizeinputpca(meshsrcpca), self.normalizeinputpca(meshdestpca)
        

        
        
        expanded_meshsrcpca = meshsrcpca#.expand(sourcescoords.shape[0], -1) # [1, 400]
        # Concatenate all coordinates
        all_coords = torch.cat([sourcescoords, destscoords], dim=0) # [N + B, 3]
        all_pca_raw = torch.cat([expanded_meshsrcpca, meshdestpca], dim=0) # [1+1, 400]
        # === Joint Encoding Step ===
        all_coord_feat = self.mlp_coord.forward_onedim(all_coords) # [N + B, coord_dim]
        all_pca_feat_single = self.mlp_pca(all_pca_raw)                   # [1 + 1, pca_dim]
        all_pca_feat = all_pca_feat_single.new_empty(N + B, all_pca_feat_single.shape[1])
        all_pca_feat[:N] = all_pca_feat_single[0]
        all_pca_feat[N:] = all_pca_feat_single[1]
        
        # Fused features
        fused_all = torch.cat([all_pca_feat, all_coord_feat], dim=1) # [N + B, concat_dim]
        all_fused_feat = self.project_down_pcacoord(fused_all)       # [N + B, hidden]
        all_fused_feat = self.mlp_pcacoord(all_fused_feat)           # [N + B, hidden]
        all_fused_feat = self.project_embed(all_fused_feat)          # [N + B, embed_dim]

        # --- Separate results back ---
        src_fused_feat = all_fused_feat[:N][:,None,:] # [N, 1, embed_dim]
        dst_fused_feat = all_fused_feat[N:][None,:, :]    # [1,B, embed_dim]
        
        
    

        # === Step 4: Final prediction ===
        diff = src_fused_feat - dst_fused_feat                            # [N, B, embed_dim]
        #print(diff.shape)
        diff = diff.view(-1, diff.shape[2])
        final_feat = self.final_layers(diff)                              # [NB, hidden]
        output = self.final_linear(final_feat)                            # [NB, 1]
        return output.abs()
    
    def forward_multi_sources(self, sourcescoords, destscoords, meshsrcpca, meshdestpca):
        # shapes:
        # sourcescoords: [N, 3]
        # destscoords: [BN, 3]
        # meshsrcpca, meshdestpca: [1, 400]
        N = sourcescoords.shape[0]
        B = destscoords.shape[0] // N
        BN = B * N
        #meshsrcpca, meshdestpca = self.normalizeinputpca(meshsrcpca), self.normalizeinputpca(meshdestpca)

        
        
        expanded_meshsrcpca = meshsrcpca#.expand(sourcescoords.shape[0], -1) # [1, 400]
        # Concatenate all coordinates
        all_coords = torch.cat([sourcescoords, destscoords], dim=0) # [N + BN, 3]
        all_pca_raw = torch.cat([expanded_meshsrcpca, meshdestpca], dim=0) # [1+1, 400]
        # === Joint Encoding Step ===
        all_coord_feat = self.mlp_coord.forward_onedim(all_coords) # [N + BN, coord_dim]
        all_pca_feat_single = self.mlp_pca(all_pca_raw)                   # [1 + 1, pca_dim]
        all_pca_feat = all_pca_feat_single.new_empty(N + BN, all_pca_feat_single.shape[1])
        all_pca_feat[:N] = all_pca_feat_single[0]
        all_pca_feat[N:] = all_pca_feat_single[1]
        
        # Fused features
        fused_all = torch.cat([all_pca_feat, all_coord_feat], dim=1) # [N + BN, concat_dim]
        all_fused_feat = self.project_down_pcacoord(fused_all)       # [N + BN, hidden]
        all_fused_feat = self.mlp_pcacoord(all_fused_feat)           # [N + BN, hidden]
        all_fused_feat = self.project_embed(all_fused_feat)          # [N + BN, embed_dim]

        # --- Separate results back ---
        src_fused_feat = all_fused_feat[:N] # [N, embed_dim]
        dst_fused_feat = all_fused_feat[N:]  # [BN, embed_dim]
        
        

        # === Step 3: Expand source embeddings to match BN destinations ===
        # For each source i ∈ [0, N-1], we tile its embedding B times
        #src_fused_feat = src_fused_feat.repeat_interleave(B, dim=0)       # [BN, embed_dim]
        # === Step 3: Expand source embeddings efficiently ===
        idx = torch.arange(N, device=src_fused_feat.device).repeat_interleave(B)  # [BN]
        src_fused_feat = src_fused_feat.index_select(0, idx)  # [BN, embed_dim]


        # === Step 4: Final prediction ===
        diff = src_fused_feat - dst_fused_feat                            # [BN, embed_dim]
        final_feat = self.final_layers(diff)                              # [BN, hidden]
        output = self.final_linear(final_feat)                            # [BN, 1]
        return output.abs()

    
    def forward_samepca_samesources(self, sourcescoords, destscoords, meshsrcpca, meshdestpca):
        # shapes:
        # sourcescoords: [1, 3]
        # destscoords: [B, 3]
        # meshsrcpca, meshdestpca: [1, 400]

        B = destscoords.shape[0]
        #meshsrcpca, meshdestpca = self.normalizeinputpca(meshsrcpca), self.normalizeinputpca(meshdestpca)

        expanded_meshsrcpca = meshsrcpca#.expand(sourcescoords.shape[0], -1) # [1, 400]
        # Concatenate all coordinates
        all_coords = torch.cat([sourcescoords, destscoords], dim=0) # [1 + B, 3]
        all_pca_raw = torch.cat([expanded_meshsrcpca, meshdestpca], dim=0) # [2, 400]
        # === Joint Encoding Step ===
        all_coord_feat = self.mlp_coord.forward_onedim(all_coords) # [1 + B, coord_dim]
        all_pca_feat_single = self.mlp_pca(all_pca_raw)                   # [1 + 1, pca_dim]
        all_pca_feat = all_pca_feat_single.new_empty(1 + B, all_pca_feat_single.shape[1])
        all_pca_feat[0] = all_pca_feat_single[0]
        all_pca_feat[1:] = all_pca_feat_single[1]
        
        # Fused features
        fused_all = torch.cat([all_pca_feat, all_coord_feat], dim=1) # [1 + B, concat_dim]
        all_fused_feat = self.project_down_pcacoord(fused_all)       # [1 + B, hidden]
        all_fused_feat = self.mlp_pcacoord(all_fused_feat)           # [1 + B, hidden]
        all_fused_feat = self.project_embed(all_fused_feat)          # [1 + B, embed_dim]

        # --- Separate results back ---
        src_fused_feat = all_fused_feat[0:1] # [1, embed_dim]
        dst_fused_feat = all_fused_feat[1:]  # [B, embed_dim]


        # === Step 3: Expand source and compute final output ===
        src_fused_feat = src_fused_feat.expand(B, -1)            # [B, embed_dim]
        diff = src_fused_feat - dst_fused_feat                   # [B, embed_dim]
        final_feat = self.final_layers(diff)                     # [B, hidden]
        output = self.final_linear(final_feat)                   # [B, 1]
        return output.abs()
    
    def forward(self, source, destination, embedding1, embedding2):
        #embedding1, embedding2= self.normalizeinputpca(embedding1), self.normalizeinputpca(embedding2)
        # 1) coords → features
        new_input = torch.cat([source.unsqueeze(1), destination.unsqueeze(1)], dim=1)
        coordinput = self.mlp_coord(new_input)               # [B,2,neurons]

        # 2) PCA embeddings separately
        pca1 = self.mlp_pca(embedding1)                      # [B,neurons]
        pca2 = self.mlp_pca(embedding2)                      # [B,neurons]

        # 3) fuse each PCA with its matching coord
        coord_src, coord_dst = coordinput[:,0,:], coordinput[:,1,:]
        fused_src = torch.cat([pca1, coord_src], dim=1)      # [B,2*neurons]
        fused_dst = torch.cat([pca2, coord_dst], dim=1)      # [B,2*neurons]

        # 4) project & MLP
        pcacoord = torch.stack([fused_src, fused_dst], dim=1)   # [B,2,2*neurons]
        B = pcacoord.shape[0]
        pcacoord = pcacoord.view(B*2, -1)                      # [B*2,2*neurons]
        pcacoord = self.project_down_pcacoord(pcacoord)
        pcacoord = self.mlp_pcacoord(pcacoord)
        pcacoord = self.project_embed(pcacoord)
        pcacoord = pcacoord.view(-1,2,pcacoord.shape[1])
        #Calculate Predicted Distance
        #hidden_lay = pcacoord[:, 0] - pcacoord[:, 1]
        #print("embed shape")
        #print(hidden_lay.shape)
        final_layer = self.final_layers(pcacoord[:, 0] - pcacoord[:, 1])
        output = self.final_linear(final_layer)
        return output.abs()

model_factory = {
    'coord_mlp': CoordMLP,
    
}


import torch
import os

def save_model(model, optimizer, scheduler,epoch, model_name="model"):
    """ Saves the model, optimizer state, and epoch number. """
    save_path = f"{model_name}.pth"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

def load_model(model, optimizer, scheduler, checkpoint_path, device="cpu"):
    """ Loads model and optimizer state from a checkpoint file. """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    
    model.train()  # Ensure the model is in training mode
    
    print(f"Model loaded from {checkpoint_path}, resuming from epoch {epoch}")
    return epoch, model, optimizer, scheduler




# Create a random tensor simulating input data (batch_size=2 for quick testing)
"""batch_size = 2
feature_dim = 1006
test_input_src = torch.randn(batch_size, 3)
test_input_dest = torch.randn(batch_size, 3)
test_input_embed = torch.randn(batch_size, 1000)

# Define model parameters
neurons = 128
no_layers = 3
embedding_size = 1000

# Initialize the model
model = CoordMLP(neurons, no_layers, embedding_size)

# Run a forward pass
output = model(test_input_src,test_input_dest,test_input_embed)

# Print output shape and values
print("Output Shape:", output.shape)  # Should be [batch_size, 1]
print("Output Values:", output)

#pcacoord = torch.randn(8,2,10)
#y = pcacoord.view(pcacoord.shape[0]*2,pcacoord.shape[2])
#print(y.shape)
#y = pcacoord.view(-1,2,y.shape[1])
#print(pcacoord.shape)
#print(y.shape)
#print((pcacoord - y).abs().sum())"""