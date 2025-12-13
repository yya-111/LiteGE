
This repository stores the codes and dataset links for LiteGE - Lightweight Representation for Geodesic Regression and Non-Isometric Shape Matching (AAAI 2026).

Geodesic : shortest curve between two points on 3D surface as illustrated below. (many applications in 3D graphics & vision.)
<img width="255" height="182" alt="image" src="https://github.com/user-attachments/assets/03465d43-ef63-4ba5-8896-c3c99bea22d8" />


We introduced LiteGE (300x faster than current SOTA & supports sparse point cloud with 300 points) for geodesic prediction & shape matching.
<img width="1200" height="250" alt="image" src="https://github.com/user-attachments/assets/3a5c54fe-9a34-46ba-98bd-44d1e9db7e53" />

Idea : LiteGE represent shapes efficiently using 50-400 features UDF-PCA vector. 
3 Stages Process:
Shapes in one dataset are canonicalized (centered, scaled and aligned)
Shapes are turned into occupancy (0/1) voxels. Most voxels are constant 0/1s. So, informative voxels are extracted.
Shapes are represented using unsigned distance (UDF), measured from the  informative voxels to the shape. Using PCA, this representation is reduced to 50-400 features UDF-PCA vector.

For instruction to run the codes, you can check the individual ReadMe files for each folder. The dataset used in this project can be downloaded from these links:
Put in /storage directory : https://www.kaggle.com/datasets/5ed40f3afdb35d0558adbad4b6a78dddf0e972bbd9b2d6fba1a0caafc4a67ea8
Put in the same directory as the train / test codes : https://www.kaggle.com/datasets/34eef62ee57d54eea9b8442341077ab565bb007414ef0e6addd1339a1cb02633 

These 2 datasets are used to train and test the Geodesic Regression and Shape Matching SMAL module on the SMAL datasets. We do not plan to release the training data for SURREAL + SMAL geodesic regression. Instead, we will release the Objaverse XL dataset and codes soon to showcase how LiteGE can work with diverse dataset and generalize across it.








