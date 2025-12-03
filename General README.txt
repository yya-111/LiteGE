Here, we provided the codes used during the training and testing of our models. 
This is a general readme file. For more details you can refer to the detailed ReadMe for each experiment in their own folders :

1. Dataset Details : To maintain reviews anonymity and compliance with the SMAL, SURREAL and FAUST datasets licences used here, we can not share all our dataset currently. We will make the dataset available upon request (to people that have accepted the relevant datasets licence) after our paper gets accepted. Our README here assumes that you have accessed our dataset.


2. Shape Matching Task : LiteGE Shape Matching is done on the SMAL dataset using our sampled models. You can check the training & testing codes, other than the UDF-PCA computation codes in Shape Matching folder. Please check the ReadMe for clarity.

3. Geodesic Regression - SMAL : LiteGE geodesic regression experiment codes, UDF-PCA creation codes on SMAL dataset and Readme are provided here.

4. Geodesic Regression - SMAL & SURREAL : LiteGE geodesic regression experiment codes, UDF-PCA creation codes on SMAL & SURREAL dataset and Readme are provided here. We also provided testing codes for test on FAUST models.

5. You can create our environment for testing & training LiteGE using "./create_env_UDFGeod.sh". For training, you can use almost any Pytorch version with Pytorch 1.12 - Pytorch 2.6, you don't need our environment. But, for testing, you might need our environment since we used Pytorch3D, pymeshlab, trimesh, scikit-learn, pynanoflann among others.

6. We provided also our geodesic ground truths dataset computation codes on SMAL model in "Geodesics Ground Truth Train Data Computation". You can re-use it for creating geodesics dataset using other dataset (FAUST, SURREAL, cleaned meshes from Thingi10K, Objaverse XL). It runs in Windows. Note also, our ipynb for creating FAUST geodesics ground truths data are also provided in the same folder.

Remark : Due to time constraints, we have not ensured that all of our codes here work flawlessly, but you can check the general idea implementation and training methods.


