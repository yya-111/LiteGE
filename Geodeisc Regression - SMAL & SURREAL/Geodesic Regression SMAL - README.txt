Here, we provided the codes to do training and testing geodesic regression task in SMAL dataset. The pretrained model is :

1. pcamodel_ep_5_l_0.028829665793713485_std_0.07522301562869257.pth : LiteGE geodesic module on SMAL + SURREAL data.

The codes are structured as follows.

2. UDF-PCA Data Creation folder contain codes for creating UDF-PCA. We used ProcessUDFPointCloudSurrealSMAL.ipynb to create UDF-PCA. Here, you need to download SURREAL dataset from 3D CODED first (https://github.com/ThibaultGROUEIX/3D-CODED). We used the first 8K meshes of SURREAL training dataset. Then, you may need to run the UDF-PCA creation codes in Geodesic Regression-SMAL first before using this code. This code used some precomputed data from the UDF-PCA creation ipynb in Geodesic Regression-SMAL. We provided it for transparency. But, you don't need to run it just to run our training and testing codes.

3. For training, you can use the following after downloading dataset. 
	python trainUDFPointCloud.py -bs 3072 --num_epoch 10            (we used batch size = 3072 and 10 epochs)

We used validation loss (after scaling them so that the g.t. mean is 100) when reporting error in SMAL + SURREAL dataset. We have not had time to compute separate test data. The train & validation data are computed using FastDGG with accuracy control of 0.1-0.3%. Note in isotropic models, Fast-DGG error is typically 3x lower than the accuracy control. So, the true mean error of our validation data is likely < 0.1%. 

4. For testing on FAUST dataset, we can use the following. Please download the dataset first.
Use : - python testsgeodesicsPointCloudUDFFaust.py -ck [checkpoint_name] for testing L1 error.

For installing the testing environment, you can use ./create_env_UDFGeod.sh and please refer to General ReadMe. The training environment used pytorch 1.12 - pytorch 2.5. You need to download our datasets first and setting up proper directory names in the codes before running any codes. Some directories names in the codes are not used and can be ignored. 

