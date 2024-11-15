import torch
import torch.nn as nn
import numpy as np


############################################
## This code is for initializing the system dimension
## and training (NN) parameters
###########################################

############################################
# set default data type to double; for GPU
# training use float
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)

FINE_TUNE = 1 # set to 1 for fine-tuning a pre-trained model

############################################
# set the network architecture
############################################
N_H_B = 1 # the number of hidden layers for the barrier
D_H_B = 20 # the number of neurons of each hidden layer for the barrier

###########################################
#Barrier function conditions
#########################################

gamma=0; #first condition <= 0
lamda=0.00001; #this is required for strict inequality >= lambda

############################################
# set loss function definition
############################################
TOL_SAFE = 0.0   # tolerance for safe and unsafe conditions
TOL_UNSAFE = 0.000

TOL_LIE = 0.000 #tolerance for the last condition

TOL_DATA_GEN = 1e-16 #for data generation


############################################
#Lipschitz bound for training
lip_h= 1
lip_dh= 1
lip_d2h= 2
############################################
# number of training epochs
############################################
EPOCHS = 100

############################################
############################################
ALPHA=0.1
BETA = 0 # if beta equals 0 then constant rate = alpha
GAMMA = 0 # when beta is nonzero, larger gamma gives faster drop of rate

#weights for loss function

DECAY_LIE = 0 # decay of lie weight 0.1 works, 1 does not work
DECAY_SAFE = 1
DECAY_UNSAFE = 1
DECAY_HUMAN = 1

############################################
REF_POINTS = [[0.0, 0.0]]
RELATIVE_THRESHOLD = 1.2

