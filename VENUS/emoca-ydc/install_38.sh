#!/bin/bash

# module swap cuda/ cuda/11.7 (cuda 11.7 12.1 ok)
module swap gnu/5.4.0 gnu9/9.4.0 # (gnu7/7.3.0 ok)

echo "downloading weight assets"
pip install --upgrade gdown # (should be 5.0.0)
mkdir assets
cd assets
gdown 'https://drive.google.com/uc?id=1btEXr5VTEC1kYW68ERgdSEF2DzHGFcyj'  # https://drive.google.com/file/d/1btEXr5VTEC1kYW68ERgdSEF2DzHGFcyj/view?usp=sharing
unzip assets.zip
cd ..

# no pull submo -> pull submodules in the top of the repo
# echo "Pulling submodules"
# bash pull_submodules.sh

echo "Installing mamba"
conda install mamba -n base -c conda-forge
if ! command -v mamba &> /dev/null
then
    echo "mamba could not be found. Please install mamba before running this script"
    exit
fi
echo "Creating conda environment"
mamba create -n emoca-ydc python=3.8 
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate emoca-ydc
if echo $CONDA_PREFIX | grep emoca-ydc
then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script"
    exit
fi
echo "Installing conda packages"
mamba env update -n emoca-ydc --file conda-environment_py38_cu11_ubuntu.yml 
echo "Installing other requirements"
pip install -r requirements38_issue.txt

mamba install -c 1adrianb face_alignment==1.4.0
pip install pandas==1.4.2 
pip install numpy==1.20.3 
pip install scikit-video==1.1.11


pip install Cython==0.29
echo "Making sure Pytorch3D installed correctly"
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
echo "Installing GDL"
pip install -e . 
echo "Installing Pytorch"
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
echo "pytorch3d"
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

pip install --upgrade transformers
conda install pytorch3d -c pytorch3d

echo "Installation finished"

# sbatch --qos=cpu_qos --partition=dell_cpu --job-name=ins38 install_38.sh
