Here, we provided the codes to do training and testing shape matching task in SMAL dataset. The pretrained model is :

1. pcamodel_ep_7_l_0.026928224170116737_std_0.07962740425484817.pth : LiteGE shape matching module 
2. tnet_model_weights_0.25699647267659503_0.0598757229745388_std_0.11098886281251907_2k.pth : TNet pretrained model used in making the train dataset and aligning models during testing with random rotation (360 degrees).

The codes are structured as follows. 

1. TNet Training Codes folder contain ipynb for training TNet, please download the dataset first. You can use Pytorch 2.0 - Pytorch 2.5.
2. UDF-PCA Data Creation folder contain codes for creating UDF-PCA data after you're done training TNet and downloading dataset. For this, note we used Kaolin NVIDIA library. Please refer to KaolinInstall.ipynb for installing it. We used ProcessSMAL_TNetModel.ipynb to create UDF-PCA. Note here, we need the one template model from SMAL dataset to get the triangle faces. 

3. Training codes folder contain the train codes. There're 2 phases of training, one where we train for 1 epoch for 8x. This is done in ./runrepeatSM.sh. The next phase, we pick the 1-2 best models and continue the training using : 
	python train.py -c -ck [checkpoint_name] -bs 3072 --num_epoch 10            (we used batch size = 3072 and 10 epochs)

4. Testing codes folder contain test codes for remeshed 5K models, template models, broken models, anisotropic remeshed models, & point clouds. 
Use : - python testshapematchefficient_multivertex_3tier.py -ck [checkpoint_name] for testing on template models.
      - python testshapematchefficientremeshed_multivertex_3tier.py -ck [checkpoint_name] for testing on remeshed 5K models.
      - python testshapematchefficientremeshed_multivertex_3tier_broken.py -ck [checkpoint_name] for testing on Broken 5K models.
      - python testshapematchefficientremeshed_multivertex_3tier_aniso.py -ck [checkpoint_name] for testing on Anisotropic models.
      - python testshapematchefficientremeshed_multivertex_3tier_pointcloud.py -ck [checkpoint_name] for testing on point cloud models.
Note : For point cloud, it is currently set to test 8K point clouds, for 500 point clouds, you may need to modify processmesh_with_pointcloud() function in utils.py to adjust the setting for 500 samples point cloud using our setting in the Appendix.

For installing the testing environment, you can use ./create_env_UDFGeod.sh and please refer to General ReadMe. The training environment used pytorch 1.12 - pytorch 2.5. You need to download our datasets first and setting up proper directory names in the codes before running any codes. Some directories names in the codes are not used and can be ignored. 
