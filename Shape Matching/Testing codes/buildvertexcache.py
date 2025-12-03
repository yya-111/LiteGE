import numpy as np
import pynanoflann
import pickle
import torch.nn as nn
# Simulated input: 7000 vertices in 3D
#all_vertices = np.random.rand(7000, 3).astype(np.float32)
import torch



def locate_corresponding_vertex_in_Y(
    query_vertex_X: torch.Tensor, # (1, 3)
    pca_X: torch.Tensor,         # (1, 400)
    pca_Y: torch.Tensor,         # (1, 400)
    points_A: torch.Tensor,      # (200, 3)
    points_A_nearest_neigh: torch.Tensor, # (200, 150, 3)
    points_A_nearest_neigh_idx: torch.Tensor, # (200, 150)
    models
):
    """
    Locates the corresponding vertex in Shape Y for a given query vertex from Shape X.

    Args:
        query_vertex_X: A (1, 3) tensor representing the query vertex from Shape X.
        pca_X: A (1, 400) tensor representing the PCA representation of Shape X.
        pca_Y: A (1, 400) tensor representing the PCA representation of Shape Y.
        points_A: A (200, 3) tensor of sparse sample points from Shape Y.
        points_A_nearest_neigh: A (200, 150, 3) tensor containing the 150 nearest
                                neighbors in shape Y for each point in points_A.
        points_A_nearest_neigh_idx: A (200, 150) tensor containing the indices of
                                    the 150 nearest neighbors in shape Y.
        model: Your PyTorch network model that predicts geodesic distance.

    Returns:
        A tuple containing:
        - corresponding_vertex_Y: The (1, 3) tensor representing the precisely
                                  located corresponding vertex in Shape Y.
        - corresponding_vertex_Y_idx: The index of the corresponding vertex
                                      in points_B.
    """
    # 1. Locate the closest vertex in points_A to the queried vertex Q from shape X
    # Predict geodesic distances from query_vertex_X to all points in points_A
    
    #distances_to_points_A = torch.cdist(query_vertex_X.unsqueeze(0), points_A.unsqueeze(0)).squeeze(0).squeeze(0)
    # Reshape query_vertex_X to (200, 3) by repeating it
    vertex1_batch = query_vertex_X.repeat(points_A.shape[0], 1) # (200, 3)
    #print(vertex1_batch)
    # points_A is already (200, 3)
    vertex2_batch = points_A

    # Reshape PCA representations to (200, 400) by repeating them
    pca1_batch = pca_X.repeat(points_A.shape[0], 1) # (200, 400)
    pca2_batch = pca_Y.repeat(points_A.shape[0], 1) # (200, 400)


    with torch.no_grad():
        outputs = [
            model(vertex1_batch, vertex2_batch, pca1_batch, pca2_batch)
            for model in models
        ]
        distances_to_points_A= torch.mean(torch.stack(outputs), dim=0).squeeze(-1)
        #distances_to_points_A = model(vertex1_batch,vertex2_batch,pca1_batch,pca2_batch).squeeze(-1) 
    # Get the index of the closest point in points_A
    closest_A_idx = torch.argmin(distances_to_points_A)
    #print(vertex2_batch)
    #return closest_A_idx
    #print(closest_A_idx)
    # Retrieve the closest vertex V from points_A
    #vertex_V = points_A[closest_A_idx].unsqueeze(0) # (1, 3)

    # 2. Retrieve the 150 nearest neighbors of vertex V from points_A_nearest_neigh
    # These are already sorted by distance in your precomputed array
    neighbors_of_V = points_A_nearest_neigh[closest_A_idx] # (150, 3)
    neighbors_of_V_idx = points_A_nearest_neigh_idx[closest_A_idx] # (150,)

    # 3. Prepare inputs for the network
    # We need to query the network for geodesic distances between query_vertex_X
    # and each of the 150 neighbors_of_V.
    
    # Reshape query_vertex_X to (150, 3) by repeating it
    vertex1_batch = vertex1_batch[:neighbors_of_V.shape[0]]# (150, 3)
    
    # neighbors_of_V is already (150, 3)
    vertex2_batch = neighbors_of_V # (150, 3)
    #print(vertex1_batch.shape, vertex2_batch.shape)
    # Reshape PCA representations to (150, 400) by repeating them
    pca1_batch = pca1_batch[:neighbors_of_V.shape[0]] # (150, 400)
    pca2_batch = pca2_batch[:neighbors_of_V.shape[0]] # (150, 400)

    # 4. Query the network for predicted geodesic distances
    with torch.no_grad():
        outputs = [
            model(vertex1_batch, vertex2_batch, pca1_batch, pca2_batch)
            for model in models
        ]
        predicted_distances = torch.mean(torch.stack(outputs), dim=0).squeeze(-1)
        
        #predicted_distances = model(
        #    vertex1_batch,
        #    vertex2_batch,
        #    pca1_batch,
        #    pca2_batch
        #).squeeze(-1) # (150,) - squeeze to remove the last dimension if output is (B, 1)

    # 5. Find the neighbor with the minimum predicted geodesic distance
    min_dist_idx_in_neighbors = torch.argmin(predicted_distances)

    # 6. Retrieve the precisely located corresponding vertex in Shape Y
    #corresponding_vertex_Y = neighbors_of_V[min_dist_idx_in_neighbors].unsqueeze(0) # (1, 3)
    corresponding_vertex_Y_idx = neighbors_of_V_idx[min_dist_idx_in_neighbors]

    return corresponding_vertex_Y_idx


def fast_locate_corresponding_vertex_in_Y(
    query_vertex_X: torch.Tensor, # (1, 3)
    pca_X: torch.Tensor,         # (1, 400)
    pca_Y: torch.Tensor,         # (1, 400)
    points_A: torch.Tensor,      # (200, 3)
    points_A_nearest_neigh: torch.Tensor, # (200, 150, 3)
    points_A_nearest_neigh_idx: torch.Tensor, # (200, 150)
    models
):
    """
    Locates the corresponding vertex in Shape Y for a given query vertex from Shape X.

    Args:
        query_vertex_X: A (1, 3) tensor representing the query vertex from Shape X.
        pca_X: A (1, 400) tensor representing the PCA representation of Shape X.
        pca_Y: A (1, 400) tensor representing the PCA representation of Shape Y.
        points_A: A (200, 3) tensor of sparse sample points from Shape Y.
        points_A_nearest_neigh: A (200, 100, 3) tensor containing the 150 nearest
                                neighbors in shape Y for each point in points_A.
        points_A_nearest_neigh_idx: A (200, 100) tensor containing the indices of
                                    the 100 nearest neighbors in shape Y.
        model: Your PyTorch network model that predicts geodesic distance.

    Returns:
        A tuple containing:
        - corresponding_vertex_Y: The (1, 3) tensor representing the precisely
                                  located corresponding vertex in Shape Y.
        - corresponding_vertex_Y_idx: The index of the corresponding vertex
                                      in points_B.
    """
    # 1. Locate the closest vertex in points_A to the queried vertex Q from shape X
    # Predict geodesic distances from query_vertex_X to all points in points_A
    
    
    vertex1_batch = query_vertex_X # (1, 3)
    
    # points_A is already (200, 3)
    vertex2_batch = points_A

    
    pca1_batch = pca_X
    pca2_batch = pca_Y


    with torch.no_grad():
        outputs = [
            model.forward_samepca_samesources(vertex1_batch, vertex2_batch, pca1_batch, pca2_batch)
            for model in models
        ]
        distances_to_points_A= torch.mean(torch.stack(outputs), dim=0).squeeze(-1)
        
    # Get the index of the closest point in points_A
    closest_A_idx = torch.argmin(distances_to_points_A)
    

    # 2. Retrieve the 100 nearest neighbors of vertex V from points_A_nearest_neigh
    # These are already sorted by distance in your precomputed array
    neighbors_of_V = points_A_nearest_neigh[closest_A_idx] # (100, 3)
    neighbors_of_V_idx = points_A_nearest_neigh_idx[closest_A_idx] # (100,)

    # 3. Prepare inputs for the network
    # We need to query the network for geodesic distances between query_vertex_X
    # and each of the 100 neighbors_of_V.
    
    # neighbors_of_V is already (100, 3)
    vertex2_batch = neighbors_of_V # (100, 3)
    

    # 4. Query the network for predicted geodesic distances
    with torch.no_grad():
        outputs = [
            model.forward_samepca_samesources(vertex1_batch, vertex2_batch, pca1_batch, pca2_batch)
            for model in models
        ]
        predicted_distances = torch.mean(torch.stack(outputs), dim=0).squeeze(-1)
        
        

    # 5. Find the neighbor with the minimum predicted geodesic distance
    min_dist_idx_in_neighbors = torch.argmin(predicted_distances)

    # 6. Retrieve the precisely located corresponding vertex in Shape Y
    corresponding_vertex_Y_idx = neighbors_of_V_idx[min_dist_idx_in_neighbors]

    return corresponding_vertex_Y_idx

def fast_locate_multi_vertex_in_Y(
    query_vertex_X: torch.Tensor, # (N, 3)
    pca_X: torch.Tensor,         # (1, 400)
    pca_Y: torch.Tensor,         # (1, 400)
    points_A: torch.Tensor,      # (200, 3)
    points_A_nearest_neigh: torch.Tensor, # (200, 100, 3)
    points_A_nearest_neigh_idx: torch.Tensor, # (200, 100)
    models
):
    """
    Locates the corresponding vertex in Shape Y for a given query vertex from Shape X.

    Args:
        query_vertex_X: A (N, 3) tensor representing the query vertex from Shape X.
        pca_X: A (1, 400) tensor representing the PCA representation of Shape X.
        pca_Y: A (1, 400) tensor representing the PCA representation of Shape Y.
        points_A: A (200, 3) tensor of sparse sample points from Shape Y.
        points_A_nearest_neigh: A (200, 150, 3) tensor containing the 150 nearest
                                neighbors in shape Y for each point in points_A.
        points_A_nearest_neigh_idx: A (200, 150) tensor containing the indices of
                                    the 150 nearest neighbors in shape Y.
        model: Your PyTorch network model that predicts geodesic distance.

    Returns:
        A tuple containing:
        - corresponding_vertex_Y_idx: The index of the corresponding vertices
                                      index in shape Y.
    """
    # 1. Locate the closest vertex in points_A to the queried vertex Q from shape X
    # Predict geodesic distances from N query_vertex_X to all points in points_A
    
    vertex1_batch = query_vertex_X # (N, 3)
    
    # points_A is already (200, 3) #200,3
    vertex2_batch = points_A

    
    pca1_batch = pca_X
    pca2_batch = pca_Y


    with torch.no_grad():
        outputs = [
            model.forward_multi_sources_alldest(vertex1_batch, vertex2_batch, pca1_batch, pca2_batch)
            for model in models
        ]
        distances_to_points_A= torch.mean(torch.stack(outputs), dim=0).squeeze(-1)
        #distances_to_points_A = model(vertex1_batch,vertex2_batch,pca1_batch,pca2_batch).squeeze(-1) 
    # Get the index of the closest point in points_A
    distances_to_points_A = distances_to_points_A.view(vertex1_batch.shape[0], vertex2_batch.shape[0])
    
    closest_A_idx = torch.argmin(distances_to_points_A, dim=1,keepdim=False)
    

    # 2. Retrieve the 200 nearest neighbors of the vertices from points_A_nearest_neigh
    # These are already sorted by distance in your precomputed array
    neighbors_of_V = points_A_nearest_neigh[closest_A_idx] # (N,100, 3)
    neighbors_of_V_idx = points_A_nearest_neigh_idx[closest_A_idx] # (N,100,)

    # 3. Prepare inputs for the network
    # We need to query the network for geodesic distances between N query_vertex_X
    # and each of the 100 neighbors_of_V for each quueried results before.
    
    
    
    # neighbors_of_V is not yet already (100N, 3)
    vertex2_batch = neighbors_of_V.view(-1, 3) # (100N, 3)
    

    # 4. Query the network for predicted geodesic distances
    with torch.no_grad():
        outputs = [
            model.forward_multi_sources(vertex1_batch, vertex2_batch, pca1_batch, pca2_batch)
            for model in models
        ]
        predicted_distances = torch.mean(torch.stack(outputs), dim=0).squeeze(-1)
        predicted_distances = predicted_distances.view(vertex1_batch.shape[0], -1) #(N, 100)
        
        
    # 5. Find the neighbor with the minimum predicted geodesic distance
    min_dist_idx_in_neighbors = torch.argmin(predicted_distances,  dim=1,keepdim=False) #N,

    # 6. Retrieve the precisely located corresponding vertices in Shape Y
    # It needs to be N x 1 to gather along dim=1 of N x 100
    index_tensor_for_gather = min_dist_idx_in_neighbors.unsqueeze(1) # Shape becomes (N, 1)
    corresponding_vertex_idx = torch.gather(neighbors_of_V_idx, 1, index_tensor_for_gather)

    return corresponding_vertex_idx.squeeze(-1)

def buildcache(all_vertices, tier1sample=190, tier2sample=142):
    # Sample sets
    #np.random.seed(42)
    indices_A = np.random.choice(all_vertices.shape[0], tier1sample, replace=False)
    #indices_B = np.random.choice(, 500, replace=False)
    #indices_B = np.arange(allvertices.shape[0])  # all
    #import time
    #s = time.time()
    points_A = all_vertices[indices_A]
    points_B = all_vertices
    #points_C = all_vertices[indices_C]

    # -----------------------------
    # 1. A → B: KNN (30 neighbors)
    # -----------------------------
    tree_B = pynanoflann.KDTree(n_neighbors=tier2sample, metric='L2', leaf_size=20)
    tree_B.fit(points_B)

    # Query
    _, nn_indices_A_to_B = tree_B.kneighbors(points_A, n_jobs=8)
    nn_indices_A_to_B = np.array(nn_indices_A_to_B, dtype = np.int32)
    
    return torch.from_numpy(points_A).float().to('cuda'), \
    torch.from_numpy(points_B[nn_indices_A_to_B]).float().to('cuda'), \
    torch.from_numpy(nn_indices_A_to_B).to('cuda')



    # Map back to global indices (since B is a subset)
    # Use advanced indexing to map local indices to global
    #knn_A_to_B_array = indices_B[nn_indices_A_to_B]  # shape (200, 150), dtype=int

    # -----------------------------
    # 2. B → C: KNN (30 neighbors)
    # -----------------------------
    #tree_C = pynanoflann.KDTree(n_neighbors=30, metric='L2', leaf_size=20)
    #tree_C.fit(points_C)

    # Query
    #_, nn_indices_B_to_C = tree_C.kneighbors(points_B, n_jobs=12)

    #knn_B_to_C_array = indices_C[nn_indices_B_to_C]  # shape (30, 30), dtype=int
    #print(time.time() - s)
    #return points_A, knn_A_to_B_array

def buildcache_3tiers(all_vertices, tier1sample=45, tier1neigh=45, tier2sample=500, tier2neigh=45):
    """
    Implements a 3-tiered Nearest Neighbor Cache building.

    Args:
        all_vertices (np.array): All mesh vertices (P_3(S)).
        tier1sample (int): Number of points for the first tier (P_1(S)).
        tier1neigh (int): Number of neighbors to connect from Tier 1 to Tier 2 (M_1).
        tier2sample (int): Number of points for the second tier (P_2(S)).
        tier2neigh (int): Number of neighbors to connect from Tier 2 to Tier 3 (M_2).

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Points for Tier 1 (P_1(S)) on CUDA.
            - torch.Tensor: Points for Tier 2 (P_2(S)) on CUDA.
            - torch.Tensor: Points for Tier 3 (P_3(S)) on CUDA.
            - torch.Tensor: Nearest neighbor indices from Tier 1 to Tier 2 on CUDA.
            - torch.Tensor: Nearest neighbor indices from Tier 2 to Tier 3 on CUDA.
    """

    # Ensure valid sample sizes for the tiers: N1 < N2 < N3
    if not (0 < tier1sample < tier2sample < all_vertices.shape[0]):
        raise ValueError("Sample sizes must satisfy 0 < tier1sample < tier2sample < all_vertices.shape[0]")

    # Sample sets for each tier
    indices_P1 = np.random.choice(all_vertices.shape[0], tier1sample, replace=False)
    # Ensure tier2sample includes indices_P1 or is completely separate.
    # For simplicity, we'll sample tier2sample independently here, which might
    # result in some overlap or completely distinct sets.
    # A more robust approach might ensure P1 is a subset of P2, or sample P2
    # from the remaining vertices after P1 is chosen.
    indices_P2 = np.random.choice(all_vertices.shape[0], tier2sample, replace=False)
    
    # Sort indices to maintain consistency if needed (not strictly required by algorithm)
    # indices_P1.sort()
    # indices_P2.sort()

    points_P1 = all_vertices[indices_P1]  # P_1(S)
    points_P2 = all_vertices[indices_P2]  # P_2(S)
    points_P3 = all_vertices             # P_3(S) - all mesh vertices

    # Initialize cache (conceptually, we'll return the mappings)
    # C(S) in the algorithm will be represented by the nn_indices tensors

    # -----------------------------
    # Tier 1 (P_1) -> Tier 2 (P_2): KNN (tier1neigh neighbors)
    # -----------------------------
    # Build KD-Tree on P_2(S)
    tree_P2 = pynanoflann.KDTree(n_neighbors=tier1neigh, metric='L2', leaf_size=20)
    tree_P2.fit(points_P2)

    # Query from P_1(S) to P_2(S)
    _, nn_indices_P1_to_P2 = tree_P2.kneighbors(points_P1, n_jobs=8)
    nn_indices_P1_to_P2 = np.array(nn_indices_P1_to_P2, dtype=np.int64)
    # Store mapping: P1_point_idx -> NN_indices_in_P2

    # -----------------------------
    # Tier 2 (P_2) -> Tier 3 (P_3): KNN (tier2neigh neighbors)
    # -----------------------------
    # Build KD-Tree on P_3(S)
    tree_P3 = pynanoflann.KDTree(n_neighbors=tier2neigh, metric='L2', leaf_size=20)
    tree_P3.fit(points_P3)

    # Query from P_2(S) to P_3(S)
    _, nn_indices_P2_to_P3 = tree_P3.kneighbors(points_P2, n_jobs=8)
    nn_indices_P2_to_P3 = np.array(nn_indices_P2_to_P3, dtype=np.int64)
    # Store mapping: P2_point_idx -> NN_indices_in_P3
    #print(nn_indices_P1_to_P2.dtype, nn_indices_P1_to_P2.shape)
    # Return results as torch tensors on CUDA
    return (
        torch.from_numpy(points_P1).float().to('cuda'),
        torch.from_numpy(points_P2).float().to('cuda'),
        torch.from_numpy(points_P3).float().to('cuda'), # P_3(S) might not be strictly needed for connections, but good for context
        torch.from_numpy(nn_indices_P1_to_P2).to('cuda'),
        torch.from_numpy(nn_indices_P2_to_P3).to('cuda'),
        
    )




def fast_locate_multi_vertex_in_Y_3tiered(
    query_vertex_X: torch.Tensor,       # (N, 3) - Batch of query vertices from Shape X
    pca_X: torch.Tensor,                # (1, 400) - PCA representation of Shape X
    pca_Y: torch.Tensor,                # (1, 400) - PCA representation of Shape Y
    points_P1: torch.Tensor,            # (N1, 3) - Tier 1 cache points (P_1(Y))
    points_P2: torch.Tensor,            # (N2, 3) - Tier 2 cache points (P_2(Y))
    points_P3: torch.Tensor,            # (N3, 3) - Tier 3 cache points (P_3(Y) - all vertices)
    nn_indices_P1_to_P2: torch.Tensor,  # (N1, M1) - Indices of P1's neighbors in P2
    nn_indices_P2_to_P3: torch.Tensor,  # (N2, M2) - Indices of P2's neighbors in P3
    models # A list of PyTorch network models that predict geodesic distance
):
    """
    Locates the corresponding vertex in Shape Y for a given batch of query vertices from Shape X,
    using a 3-tiered nearest neighbor cache and a coarse-to-fine search.

    Args:
        query_vertex_X (torch.Tensor): A (N, 3) tensor representing the batch of query vertices from Shape X.
        pca_X (torch.Tensor): A (1, 400) tensor representing the PCA representation of Shape X.
        pca_Y (torch.Tensor): A (1, 400) tensor representing the PCA representation of Shape Y.
        points_P1 (torch.Tensor): A (N1, 3) tensor of sparse sample points from Shape Y (Tier 1 cache).
        points_P2 (torch.Tensor): A (N2, 3) tensor of intermediate sample points from Shape Y (Tier 2 cache).
        points_P3 (torch.Tensor): A (N3, 3) tensor of all vertices from Shape Y (Tier 3 cache).
        nn_indices_P1_to_P2 (torch.Tensor): A (N1, M1) tensor containing the indices of
                                           P1's M1 nearest neighbors in P2.
        nn_indices_P2_to_P3 (torch.Tensor): A (N2, M2) tensor containing the indices of
                                           P2's M2 nearest neighbors in P3.
        models (list): A list of PyTorch network models that predict geodesic distance.

    Returns:
        torch.Tensor: A (N,) tensor containing the final indices of the corresponding vertices
                      in Shape Y (relative to points_P3, i.e., all_vertices).
    """
    N = query_vertex_X.shape[0] # Batch size of query vertices
    #print(query_)

    # --- Tier 1 Search: Locate closest point in P1(Y) ---
    # Q_X (N,3) vs P_1(Y) (N1,3) -> distances (N, N1)
    
    # vertex1_batch (N, 3) is query_vertex_X
    # vertex2_batch (N1, 3) is points_P1
    
    with torch.no_grad():
        outputs_tier1 = [
            model.forward_multi_sources_alldest(query_vertex_X, points_P1, pca_X, pca_Y)
            for model in models
        ]
        distances_tier1 = torch.mean(torch.stack(outputs_tier1), dim=0).squeeze(-1)
        distances_tier1 = distances_tier1.view(N, points_P1.shape[0]) # (N, N1)

    # Find the best match q_1Y (local index in points_P1)
    closest_P1_local_idx = torch.argmin(distances_tier1, dim=1, keepdim=False) # (N,)

    # --- Tier 2 Search: Refine search within P2(Y) neighbors of closest P1 point ---
    # Retrieve the neighbors in P2 for the closest P1 points
    # `nn_indices_P1_to_P2` stores indices relative to `points_P2`
    
    # For each query, we have a chosen P1 point, and for that P1 point,
    # we have M1 neighbors in P2. So, for N queries, we have N * M1 candidates in P2.
    candidates_P2_indices = nn_indices_P1_to_P2[closest_P1_local_idx] # (N, M1)
    
    # Get the actual points from P2 using these indices
    # This creates a (N, M1, 3) tensor of candidate points for each query vertex.
    current_candidate_points_P2 = points_P2[candidates_P2_indices] # (N, M1, 3)

    # Prepare inputs for network:
    # vertex1_batch (N, 3) is query_vertex_X
    # vertex2_batch (N * M1, 3) is the flattened candidates from P2
    
    vertex2_batch_tier2 = current_candidate_points_P2.view(-1, 3) # (N * M1, 3)
    
    # Replicate pca_X and pca_Y to match the N*M1 batch size for vertex2
    #pca_X_repeated_tier2 = pca_X.repeat(N, 1) # N times for each query vertex
    #pca_Y_repeated_tier2 = pca_Y.repeat(N, 1) # N times for each query vertex
    
    # The model expects query_vertex_X of shape (N,3) if vertex2_batch_tier2 is (N*M1,3).
    # This implies that the model's forward_multi_sources expects the first argument to broadcast
    # over the repeated second argument (which is the case for your previous example).
    
    with torch.no_grad():
        outputs_tier2 = [
            # The model is expected to handle the broadcasting of query_vertex_X to N*M1 candidates
            # if query_vertex_X remains (N,3). If forward_multi_sources expects
            # (N*M, 3) for both inputs, we need to replicate query_vertex_X.
            # Your original example used query_vertex_X as (N,3) and vertex2_batch as (N*100,3)
            # and it worked, meaning the model implicitly repeats query_vertex_X.
            # Let's stick to that pattern.
            model.forward_multi_sources(query_vertex_X, vertex2_batch_tier2, pca_X, pca_Y)
            for model in models
        ]
        predicted_distances_to_P2 = torch.mean(torch.stack(outputs_tier2), dim=0).squeeze(-1)
        predicted_distances_to_P2 = predicted_distances_to_P2.view(N, -1) # Reshape to (N, M1)

    # Find the best match q_2Y (local index in the M1 neighbors)
    min_dist_idx_in_P2_neighbors = torch.argmin(predicted_distances_to_P2, dim=1, keepdim=False) # (N,)

    # Get the actual index in P2 for the best match (its global index in points_P2)
    index_tensor_for_gather_P2 = min_dist_idx_in_P2_neighbors.unsqueeze(1) # (N, 1)
    closest_P2_global_idx = torch.gather(candidates_P2_indices, 1, index_tensor_for_gather_P2).squeeze(1) # (N,) -> index in points_P2

    # --- Tier 3 Search: Refine search within P3(Y) neighbors of closest P2 point (Final Match) ---
    # Retrieve M2 nearest neighbors of q_2Y from P_3(Y) (all vertices) using C(Y)
    
    # For each query, we have a chosen P2 point, and for that P2 point,
    # we have M2 neighbors in P3. So, for N queries, we have N * M2 candidates in P3.
    final_candidates_P3_indices = nn_indices_P2_to_P3[closest_P2_global_idx] # (N, M2)
    
    # Get the actual points from P3 using these indices
    # This creates a (N, M2, 3) tensor of final candidate points for each query vertex.
    current_candidate_points_P3 = points_P3[final_candidates_P3_indices] # (N, M2, 3)

    # Prepare inputs for network:
    # vertex1_batch (N, 3) is query_vertex_X
    # vertex2_batch (N * M2, 3) is the flattened final candidates from P3
    
    vertex2_batch_tier3 = current_candidate_points_P3.view(-1, 3) # (N * M2, 3)

    with torch.no_grad():
        outputs_tier3 = [
            # Again, assuming model.forward_multi_sources handles broadcasting of query_vertex_X
            model.forward_multi_sources(query_vertex_X, vertex2_batch_tier3, pca_X, pca_Y)
            for model in models
        ]
        predicted_distances_to_P3 = torch.mean(torch.stack(outputs_tier3), dim=0).squeeze(-1)
        predicted_distances_to_P3 = predicted_distances_to_P3.view(N, -1) # Reshape to (N, M2)

    # Find the best match q_3Y (local index in the M2 final neighbors)
    min_dist_idx_in_final_neighbors = torch.argmin(predicted_distances_to_P3, dim=1, keepdim=False) # (N,)

    # Retrieve the precisely located corresponding vertices in Shape Y (global indices in points_P3)
    index_tensor_for_gather_final = min_dist_idx_in_final_neighbors.unsqueeze(1) # (N, 1)
    corresponding_vertex_Y_idx = torch.gather(final_candidates_P3_indices, 1, index_tensor_for_gather_final).squeeze(1) # (N,)

    return corresponding_vertex_Y_idx