Here, we provided the codes to do training and testing geodesic regression task in SMAL dataset. The pretrained model is :

1. pcamodel_ep_3_l_0.02094968554851589_std_0.03780007733523726_geod.pth : LiteGE geodesic module on SMAL data.

The codes are structured as follows. Please check Dataset Links & Details ReadMe.txt file to download our dataset first before using. Note that our codes for training and testing are tied to our system storage system before. So, you might need to change it a bit.


2. UDF-PCA Data Creation folder contain codes for creating UDF-PCA data after you have access to our dataset. We used ProcessUDFPointCloud.ipynb to create UDF-PCA from MeshSMALVerticesRandom.npz. You need the triangle faces of SMAL dataset to run it. You can get one in MeshTestSample/meshtest_4_ori.off or just download the SMAL pkl file from https://smal.is.tue.mpg.de/.

3. Training codes folder contain the train codes. You can run it using the following after downloading dataset : 
	python trainUDFPointCloud.py -bs 3072 --num_epoch 10            (we used batch size = 3072 and 10 epochs)

4. Testing codes folder contain test codes for point clouds sampled from remeshed 5K models. Please download the dataset first.
Use : - python testsgeodesicsPointCloudUDF_error.py -ck [checkpoint_name] for testing L1 error.
You can change line 182, num_samples = ... , to change the number of point cloud samples. The error reported is not yet scaled with so that the geodesic ground truth is 100. We used the train data mean to scale it.

5. Testing codes for memory and rruntime is in testing codes folder too. You can run it:
python testPointCloudUDFRunTimeMem.py -ck [checkpoint name] --batch_size [Number of point clouds queried] --num_points [number of points in each point cloud (default to 2K)]

Remark : Note that we can also use the validation loss to test our method. But, we used the testing set to avoid overfitting. Moreover, the testing set is computed with FastDGG using 0.1% accuracy control parameter that is more accurate and representative of our method accuracy. The train and validation data used FastDGG 0.3% - 1% accuracy control to compute them. We didn't use higher accuracy to save time.

For installing the testing environment, you can use ./create_env_UDFGeod.sh and please refer to General ReadMe. The training environment used pytorch 1.12 - pytorch 2.5. You need to download our datasets first and setting up proper directory names in the codes before running any codes. Some directories names in the codes are not used and can be ignored. 
