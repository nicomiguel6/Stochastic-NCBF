import torch
import torch.nn as nn
import sys1
import train
import time

system = 'ip'

if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")

    import torch

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def barr_nn(system, human=False):
    # generate training data
    # sys.sys_data(sys)
    data, prob = sys1.system_data(system)
    # print('eps',data.eps )
    time_start_data = time.time()
    batches_safe, batches_unsafe, batches_domain = data.gen_batch_data()
    time_end_data = time.time()
    
    ############################################
    # number of mini_batches
    ############################################
    BATCHES_S = len(batches_safe)
    BATCHES_U = len(batches_unsafe)
    BATCHES_D = len(batches_domain)
    BATCHES = max(BATCHES_S, BATCHES_U, BATCHES_D)
    NUM_BATCHES = [BATCHES_S, BATCHES_U, BATCHES_D, BATCHES]
    
    # train and return the learned model
    time_start_train = time.time()
    res = train.train(batches_safe, batches_unsafe, batches_domain, NUM_BATCHES, system) 
    time_end_train = time.time()
    
    print("\nData generation totally costs:", time_end_data - time_start_data)
    print("Training totally costs:", time_end_train - time_start_train)
    print("-------------------------------------------------------------------------")
        
    return barr_nn


if __name__ =="__main__":

    ## Generate data
    system = 'ip'
    data, prob = sys1.system_data(system)
    # print('eps',data.eps )
    time_start_data = time.time()
    batches_safe, batches_unsafe, batches_domain = data.gen_batch_data()
    time_end_data = time.time()

    ############################################
    # number of mini_batches
    ############################################
    BATCHES_S = len(batches_safe)
    BATCHES_U = len(batches_unsafe)
    BATCHES_D = len(batches_domain)
    BATCHES = max(BATCHES_S, BATCHES_U, BATCHES_D)
    NUM_BATCHES = [BATCHES_S, BATCHES_U, BATCHES_D, BATCHES]

     
    ## First, we want to set the training for non-human reference system, starting with the warm start
    time_start_train = time.time()
    res = train.train(batches_safe, batches_unsafe, batches_domain, NUM_BATCHES, system) 
    time_end_train = time.time()

    print("\nData generation for non-human full totally costs:", time_end_data - time_start_data)
    
    ## Second, we want to set the training for human reference system
    time_start_train = time.time()
    res = train.train(batches_safe, batches_unsafe, batches_domain, NUM_BATCHES, system, human=True) 
    time_end_train = time.time()

    print("\nData generation for human full totally costs:", time_end_data - time_start_data)

    #  torch.save(barr_nn,'saved_weights/barr_nn')

