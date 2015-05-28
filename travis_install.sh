#! /bin/bash

set -e

apt-get install python-dev python-numpy python-matplotlib python-nose python-h5py python-pip

# install CUDA
CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
#CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_6.5-14_amd64.deb
CUDA_FILE=/tmp/cuda_install.deb
curl $CUDA_URL -o $CUDA_FILE
dpkg -i $CUDA_FILE
rm -f $CUDA_FILE
apt-get -y update
# Install the minimal CUDA subpackages required to test Caffe build.
# For a full CUDA installation, add 'cuda' to the list of packages.
apt-get -y install cuda-core-7-0 cuda-cublas-7-0 cuda-cublas-dev-7-0 cuda-cudart-7-0 cuda-cudart-dev-7-0 cuda-curand-7-0 cuda-curand-dev-7-0
# Create CUDA symlink at /usr/local/cuda
# (This would normally be created by the CUDA installer, but we create it
# manually since we did a partial installation.)
ln -s /usr/local/cuda-7.0 /usr/local/cuda

# install cuDNN, once I figure out how to do so on Travis

# Uncomment these if I ever figure out how to install cuDNN on Travis
# CUDNN_PATH=/usr/local/share/cudnn
# export LIBRARY_PATH=$CUDNN_PATH:$LIBRARY_PATH  # for the static libs
# export CPATH=$CUDNN_PATH:$CPATH  # for the .h file


# install Theano
pip install -H -q --no-deps git+git://github.com/Theano/Theano.git

# install Pylearn2, for comparison tests
pip install -H -q --no-deps git+git://github.com/lisa-lab/pylearn2.git

mkdir ./datasets
export SIMPLELEARN_DATA_PATH=./data/datasets

# set up PYTHONPATH to point to Simplelearn
export PYTHONPATH=$PYTHONPATH:./

# Some unit tests don't run on Travis. They check for this env. variable to
# determine whether to run.
export TRAVIS_TEST
