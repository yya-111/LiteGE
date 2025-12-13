<img width="7043" height="554" alt="image" src="https://github.com/user-attachments/assets/5403482c-2b27-4771-828e-9ab4d6d24480" />
This repository stores the codes and dataset links for LiteGE - Lightweight Representation for Geodesic Regression and Non-Isometric Shape Matching (AAAI 2026).

# Background

Geodesic : shortest curve between two points on 3D surface as illustrated below. (many applications in 3D graphics & vision.)

<img width="255" height="182" alt="image" src="https://github.com/user-attachments/assets/03465d43-ef63-4ba5-8896-c3c99bea22d8" />


We introduced LiteGE (300x faster than current SOTA & supports sparse point cloud with 300 points) for geodesic prediction & shape matching.
<img width="1200" height="250" alt="image" src="https://github.com/user-attachments/assets/3a5c54fe-9a34-46ba-98bd-44d1e9db7e53" />

Idea : LiteGE represent shapes efficiently using 50-400 features UDF-PCA vector. 

3 Stages Process:

1. Shapes in one dataset are canonicalized (centered, scaled and aligned)

2. Shapes are turned into 3D grid occupancy (0/1) voxels, indicating whether each 3D grid voxel is inside or outside the shape. Most voxels are constant 0/1s. So, informative voxels (voxels with variance larger than a threshold across the dataset) are extracted.

3. Shapes are represented using unsigned distance (UDF), measured from the  informative voxels to the shape. Using PCA, this representation is reduced to 50-400 features UDF-PCA vector.

This representation is then used to predict geodesic distances in a 3D shape using the network architecture explained in our paper: 

<img width="362" height="193" alt="image" src="https://github.com/user-attachments/assets/262b6c90-79ae-4145-be01-12c877ad59d8" />

1. Source & destination points are embedded using Coord-MLP

2. UDF-PCA vector is  encoded into a shape embedding.

3. Point embeddings and shape embedding are fused via MLP.

4. The difference between fused embeddings is passed to an MLP to predict geodesic distance.

5. For shape matching: involves 2 points from 2 distinct shapes together with their UDF-PCA vectors. The distance predicted is average distance when the 2 points both lie on either one of the two shapes.

The predicted geodesic distance between 2 points on 2 distinct shapes can be used to perform shape matching using a coarse-to-fine strategy.

# Codes and Dataset Instructions

For instruction to run the codes, you can check the individual ReadMe files for each folder. The dataset used in this project can be downloaded from these links:

Put in /storage directory : https://www.kaggle.com/datasets/5ed40f3afdb35d0558adbad4b6a78dddf0e972bbd9b2d6fba1a0caafc4a67ea8

Put in the same directory as the train / test codes : https://www.kaggle.com/datasets/34eef62ee57d54eea9b8442341077ab565bb007414ef0e6addd1339a1cb02633 

These 2 datasets are used to train and test the Geodesic Regression and Shape Matching on the SMAL (4-legged animals) datasets. We do not plan to release the training data for SURREAL + SMAL geodesic regression. Instead, we will release the Objaverse XL dataset and codes soon to showcase how LiteGE can work with diverse dataset and generalize across it.

To run the test codes you need also a sample mesh, in MeshTestSample directory. The MeshTestSample directory here must be placed inside the same directory of the testing codes that you want to use. Note also, to run our shape matching testing codes, you can also place the tnet_model_weights_0.25699647267659503_0.0598757229745388_std_0.11098886281251907_2k.pth in the /storage directory. 








