#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
cd /root
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

$HOME/miniconda/bin/conda init
INSTALL_PATH="$HOME/miniconda"
source ~/.bashrc

source $INSTALL_PATH/etc/profile.d/conda.sh
cd /notebooks
# Create a new conda environment with Python 3.9 (-y automatically confirms)
echo "Creating conda environment 'GeodUDF' with Python 3.9..."

conda env create -f environmentGeodUDF.yml -y
conda activate GeodUDF
pip install pynanoflann
pip install pymeshlab

#python testsgeodesicsPointCloudUDF.py -ck pca_model.pth #Testing codes

