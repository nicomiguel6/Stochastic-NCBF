import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import hyper_para as hyp # parameters
import loss # computing loss
import lrate
import os
import time

from deep_differential_network.differential_hessian_network import DifferentialNetwork

from utils.logger import DataLog
from utils.make_train_plots import make_train_plots

import sys1
import main

from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(batches, batch_size, num_workers):
    concatenated_batches = torch.cat(batches, dim=0)
    dataset = TensorDataset(concatenated_batches)  # Convert to TensorDataset if batches are tensors
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader


data, prob = sys1.system_data(main.system)


LOAD_MODEL = False
RENDER = True
SAVE_MODEL = True
SAVE_PLOT = False


torch.set_printoptions(precision=7)

#################################################
# iterative training: the most important function
# it relies on three assistant functions:
#################################################

# used for initialization and restart

def initialize_parameters(n_h_b, d_h_b):
    #initialize the eta variable for scenario verification
    lambda_h=Variable(torch.normal(mean=10*torch.ones(n_h_b*d_h_b),std=0.001*torch.ones(n_h_b*d_h_b)), requires_grad=True)
    lambda_dh=Variable(torch.normal(mean=10*torch.ones(n_h_b*d_h_b),std=0.001*torch.ones(n_h_b*d_h_b)), requires_grad=True)
    lambda_d2h=Variable(torch.normal(mean=10*torch.ones(n_h_b*d_h_b),std=0.001*torch.ones(n_h_b*d_h_b)), requires_grad=True)
    print("Initialize eta")
    eta=Variable(torch.normal(mean=torch.tensor([-0.003]), std=torch.tensor([0.00001])), requires_grad=True)
    return lambda_h, lambda_dh, lambda_d2h, eta

    
def initialize_nn(num_batches, eta, lambda_h, lambda_dh, human = False):    
    print("Initialize nn parameters!")
    cuda_flag = True
    filename = f"barr_nn"
    n_dof = data.DIM_S
    # Construct Hyperparameters:
    # Activation must be in ['ReLu', 'SoftPlus']
    hyper = {'n_width': hyp.D_H_B,
             'n_depth': hyp.N_H_B,
             'learning_rate': 1.0e-03,
             'weight_decay': 1.e-6,
             'activation': "SoftPlus"}

    # Load existing model parameters:
    if LOAD_MODEL:
        if human:
            barr_nn = torch.load('experiments/ip_w_eta_human_WS/iterations/barr_nn_99')
        else:
            barr_nn = torch.load('experiments/ip_w_eta_WS/iterations/barr_nn_99')
        # load_file = f"./models/{filename}.torch"
        # state = torch.load(load_file, map_location='cpu')

        #barr_nn = torch.load('experiments/di_l12_0/iterations/barr_nn_1') #DifferentialNetwork(n_dof, **state['hyper'])
        # barr_nn.load_state_dict(state['state_dict'])

    else:
        barr_nn = DifferentialNetwork(n_dof, **hyper)
        for p in barr_nn.parameters():
            nn.init.normal_(p,0,0.1)

    if cuda_flag:
        barr_nn.cuda()
        
    # Generate & Initialize the Optimizer:
    t0_opt = time.perf_counter()
    optimizer = torch.optim.Adam([{'params':barr_nn.parameters()},{'params':[lambda_h,lambda_dh]}],
                                    lr=hyper["learning_rate"],
                                    weight_decay=hyper["weight_decay"],
                                    amsgrad=True)

    print("{0:30}: {1:05.2f}s".format("Initialize Optimizer", time.perf_counter() - t0_opt))
    scheduler = lrate.set_scheduler(optimizer, num_batches)

    return barr_nn, optimizer,scheduler

def train(batches_safe, batches_unsafe, batches_domain, NUM_BATCHES, system, human = False):
    # Initialize data loaders with num_workers
    # batch_size = 1600  # Set according to your memory capacity
    # num_workers = 16
    # dataloader_safe = create_dataloader(batches_safe, batch_size, num_workers)
    # dataloader_unsafe = create_dataloader(batches_unsafe, batch_size, num_workers)
    # dataloader_domain = create_dataloader(batches_domain, batch_size, num_workers)
    
    logger = DataLog()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not human:
        log_dir = "experiments/" + system+"_w_eta_WS_rich_data"
    else:
        log_dir = "experiments/" + system+"_w_eta_human_WS_rich_data"
    
    working_dir = os.getcwd()

    if os.path.isdir(log_dir) == False:
        os.mkdir(log_dir)

    previous_dir = os.getcwd()
    
    os.chdir(log_dir)
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') ==False: os.mkdir('logs')

    log_dir = os.getcwd()
    os.chdir(working_dir)
    num_restart = -1

    ############################## the main training loop ##################################################################
    while num_restart < 0:
        num_restart += 1
        
        # initialize nn models and optimizers and schedulers
        lambda_h, lambda_dh, lambda_d2h, eta = initialize_parameters(hyp.N_H_B, hyp.D_H_B)
        barr_nn, optimizer_barr, scheduler_barr = initialize_nn(NUM_BATCHES[3], eta, lambda_h, lambda_dh, human = human)
        optimizer_eta= torch.optim.SGD([{'params':[eta]}], lr=0.001, momentum=0)


        safe_list = np.arange(NUM_BATCHES[3]) % NUM_BATCHES[0]  # generate batch indices    # S
        unsafe_list = np.arange(NUM_BATCHES[3]) % NUM_BATCHES[1]                            # U
        domain_list = np.arange(NUM_BATCHES[3]) % NUM_BATCHES[2]                            # D

        for epoch in range(hyp.EPOCHS): # train for a number of epochs
            # initialize epoch
            epoch_loss = 0 # scalar
            lie_loss = 0
            lie_eta_loss = 0
            lmi_loss = 0 #scalar
            eta_loss = 0
            epoch_gradient_flag = True # gradient is within range
            hyp.CURR_MAX_GRAD = 0

            # mini-batches shuffle by shuffling batch indices
            np.random.shuffle(safe_list)
            np.random.shuffle(unsafe_list)
            np.random.shuffle(domain_list)
            print(NUM_BATCHES[3])

            # train mini-batches
            for batch_index in range(NUM_BATCHES[3]):
            # for batch_index, (batch_safe, batch_unsafe, batch_domain) in enumerate(zip(dataloader_safe, dataloader_unsafe, dataloader_domain)):
                # batch data selection
                batch_safe = batches_safe[safe_list[batch_index]].to(device)
                batch_unsafe = batches_unsafe[unsafe_list[batch_index]].to(device)
                batch_domain = batches_domain[domain_list[batch_index]].to(device)
                # batch_safe = batch_safe[0].to(device)
                # batch_unsafe = batch_unsafe[0].to(device)
                # batch_domain = batch_domain[0].to(device)
                ############################## mini-batch training ################################################
                optimizer_barr.zero_grad() # clear gradient of parameters
                optimizer_eta.zero_grad()

                sigma = 0.00*torch.eye(data.DIM_S)
                
                _, _, lie_batch_loss, lie_eta_batch_loss, curr_batch_loss = loss.calc_loss(barr_nn, batch_safe, batch_unsafe, batch_domain, epoch, batch_index,eta, hyp.lip_h, sigma, human = human)
                # batch_loss is a tensor, batch_gradient is a scalar
                curr_batch_loss.backward() # compute gradient using backward()
                # update weight and bias
                optimizer_barr.step() # gradient descent once
                   
                optimizer_barr.zero_grad()

                curr_lmi_loss= loss.calc_lmi_loss(barr_nn, lambda_h, lambda_dh, lambda_d2h, hyp.lip_h, hyp.lip_dh, hyp.lip_d2h, sigma)
                                
                if curr_lmi_loss >= -5000:
                    curr_lmi_loss.backward()
                    optimizer_barr.step()
                    optimizer_barr.zero_grad()
                
                optimizer_eta.zero_grad()
                
                curr_eta_loss=  loss.calc_eta_loss(eta, hyp.lip_h, hyp.lip_dh, hyp.lip_d2h)
                
                if curr_eta_loss > 0:
                    curr_eta_loss.backward()
                    optimizer_eta.step()
                
                # learning rate scheduling for each mini batch
                scheduler_barr.step() # re-schedule learning rate once

                # update epoch loss
                lie_loss += lie_batch_loss.item()
                lie_eta_loss += lie_eta_batch_loss.item()
                epoch_loss += curr_batch_loss.item()
                lmi_loss += curr_lmi_loss.item()
                eta_loss += curr_eta_loss.item()

                print("restart: %-2s" % num_restart, "epoch: %-3s" % epoch, "batch: %-5s" % batch_index, "batch_loss: %-25s" % curr_batch_loss.item(), \
                          "epoch_loss: %-25s" % epoch_loss,"lie_loss: %-25s" % lie_loss, "lmi loss: % 25s" %lmi_loss, "eta loss: % 25s" %eta_loss, "eta: % 25s" % eta,"lie_eta_loss: %-25s" % lie_eta_loss)

            logger.log_kv('epoch', epoch)
            logger.log_kv('epoch_loss', epoch_loss)
            logger.log_kv('lie_loss', lie_loss)
            logger.log_kv('lmi_loss', lmi_loss)
            logger.log_kv('eta_loss', eta_loss)

            logger.save_log(log_dir+"/logs")
            make_train_plots(log = logger.log, keys=['epoch', 'epoch_loss'], save_loc=log_dir+"/logs")
            make_train_plots(log = logger.log, keys=['epoch', 'lie_loss'], save_loc=log_dir+"/logs")
            torch.save(barr_nn,log_dir+'/iterations/barr_nn_'+str(epoch))


            if (epoch_loss <= 0) and (lmi_loss <= 0) and (eta_loss <= 0):
                print("The last epoch:", epoch, "of restart:", num_restart)
                return True # epoch success: end of epoch training

    return False


