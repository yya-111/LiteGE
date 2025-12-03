This is Geodesic Ground Truth dataset computation codes for SMAL & FAUST dataset. Note that you need to access our dataset first to be able to ru n it. We provided the codes for transparency.

1. We used ComputeDGG.ipynb first to precompute DGG for each mesh in the SMAL dataset.
2. Then we run ComputeGeodesicMeshPairs.ipynb to create 1 train dataset. There're 2 train datasets in our uploaded dataset. We used the same logic in ComputeGeodesicMeshPairs.ipynb to create the other train dataset and the validation dataset.

3. There's also ComputeDGGFAUST.ipynb to compute geodesics distances ground truth on FAUST dataset using 0.1% accuracy control parameter that we used to test generalization ability our Geodesic Regression Module for SMAL + SURREAL data.

We can modify these ipynb easily to create geodesic distances ground truth dataset for other types of dataset using the desired accuracy control parameter. The dataset is created in Windows environment. One can also run it in Linux using the Wine software to run FastDGG executables.
