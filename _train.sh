#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Comment if using GPU
export CUDA_VISIBLE_DEVICES=-1

# For MuJoCo
# NOTE Below MuJoCo path, Nvidia driver version, and GLEW path 
# may differ depends on a computer setting
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Begin experiment
python3.6 main.py
