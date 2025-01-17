# for the machine learning model
import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import pandas as pd
from utils import MNIST_partial, plot_training_metrics
from model import MnistModel, evaluate, evaluate_combined_2
from boson_sampler import BosonSampler
import perceval as pcvl
import numpy as np
#import perceval.providers.scaleway as scw  # Uncomment to allow running on scaleway

#### TRAINING LOOP ####
def fit(epochs, lr, model_temp, train_loader, val_loader, bs: BosonSampler, 
        opt_func = torch.optim.SGD, device='cpu', optimizer_temp=None,
        file_path="checkpoints mps classical"):
    history = []
    print("inside new version 2")
    print("device:", device)

    model = model_temp.to(device)
    print("model shifted to device:", device)

    if optimizer_temp is None:
        optimizer = opt_func(model.parameters(), lr)
        print("optimizer shifted to device:", optimizer)
    else:
        optimizer = optimizer_temp

    # train_loader = train_loader.to(device)
    # val_loader = val_loader.to(device)

    # creation of empty lists to store the training metrics
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for epoch in range(epochs):
        training_losses, training_accs = 0, 0
        step_counter_per_epoch = 0

        ## Training Phase
        for step, batch in enumerate(tqdm(train_loader)):
            # batch = batch.to(device)

            # embedding in the BS
            if model.embedding_size:
                # batch = np.array(batch)
                print("inside version for boson sampler:", type(batch[0]), type(batch[1]))

                images, labs = batch
                print("inside boson sampler; images and labs data:", images.shape, labs.shape)

                batch_size = images.shape[0]

                if batch_size == 1:
                    images = images.squeeze(0).squeeze(0)
                    print("shape of images after squeezed:", images.shape)

                    embs = bs.embed(images,1000)

                    t_s = time.time()
                    print("info about embs:", type(embs), embs.shape)

                    loss,acc = model.training_step(batch,emb=embs.unsqueeze(0))
                else:
                    output_tensors = []
                    for i in range(images.shape[0]):  # Iterate over the batch
                        single_image = images[i].squeeze(0)  # Remove the singleton dimension -> [28, 28]
                        processed_output = bs.embed(single_image, 1000)  # Apply the function
                        print("info about embs:", type(processed_output), processed_output.shape)
                        output_tensors.append(processed_output)

                    # Stack all outputs to create a tensor of shape [32, 435]
                    embs = torch.stack(output_tensors)
                    del output_tensors

                    t_s = time.time()
                    print("info about embs:", type(embs), embs.shape)

                    loss,acc = model.training_step(batch, emb=embs)

            else:
                loss,acc = model.training_step(batch)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_losses+=float(loss.detach())
            training_accs+=float(acc.detach())
            if model.embedding_size and step%100==0:
                try:
                    # torch.save({
                    #     'model_statedict': model.state_dict(),
                    #     'optimizer_statedict': optimizer.state_dict()
                    # }, f"./checkpoints mps/epoch_{epoch + 1}_{step_counter_per_epoch + 1}.pth")

                    print(f"fit function newer version; STEP {step}, Training-acc = {training_accs/(step+1)}, Training-losses = {training_losses/(step+1)}")

                except Exception as e:
                    print("Error in saving the checkpoint at path ", f"./checkpoints kk mps/epoch_{epoch + 1}_{step_counter_per_epoch + 1}.pth")
                    print("with error:", e)

                step_counter_per_epoch += 1
        
        ## Validation phase
        result = evaluate(model, val_loader, bs)
        validation_loss, validation_acc = result['val_loss'], result['val_acc']
        model.epoch_end(epoch, result)
        history.append(result)

        ## summing up all the training and validation metrics
        training_loss = training_losses/len(train_loader)
        training_accs = training_accs/len(train_loader)
        train_loss.append(training_loss)
        train_acc.append(training_accs)
        val_loss.append(validation_loss)
        val_acc.append(validation_acc)

        # plot training curves
        plot_training_metrics(train_acc,val_acc,train_loss,val_loss)

        # torch.save({
        #     'model_statedict': model.state_dict(),
        #     'optimizer_statedict': optimizer.state_dict()
        # }, f"./checkpoints mps classical/epoch_{epoch + 1}.pth")

        torch.save({
            'model_statedict': model.state_dict(),
            'optimizer_statedict': optimizer.state_dict()
        }, f"./{file_path}/epoch_{epoch + 1}.pth")
    return(history)

def fit_combined_2(epochs, lr, model_temp, train_loader, val_loader, bs: BosonSampler, 
        opt_func = torch.optim.SGD, device='cpu', optimizer_temp=None, file_path=""):
    history = []
    # print("inside new version 2")
    print("device:", device)

    model = model_temp.to(device)
    print("model shifted to device:", device)

    if optimizer_temp is None:
        optimizer = opt_func(model.parameters(), lr)
        print("optimizer shifted to device:", optimizer)
    else:
        optimizer = optimizer_temp

    # train_loader = train_loader.to(device)
    # val_loader = val_loader.to(device)

    # creation of empty lists to store the training metrics
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for epoch in range(epochs):
        training_losses, training_accs = 0, 0
        step_counter_per_epoch = 0

        ## Training Phase
        for step, batch in enumerate(tqdm(train_loader)):
            # batch = batch.to(device)
            # print("details about step, batch just enterred:", step, batch[0].shape, batch[1].shape, batch[2].shape)

            batch_PQK = batch[0]
            batch_normal = batch[1]
            labels = batch[2]

            # embedding in the BS
            if model.embedding_size:
                # batch = np.array(batch)
                print("inside version for boson sampler:", type(batch[0]), type(batch[1]))

                images, labs = batch
                print("inside boson sampler; images and labs data:", images.shape, labs.shape)

                batch_size = images.shape[0]

                if batch_size == 1:
                    images = images.squeeze(0).squeeze(0)
                    print("shape of images after squeezed:", images.shape)

                    embs = bs.embed(images,1000)

                    t_s = time.time()
                    print("info about embs:", type(embs), embs.shape)

                    loss,acc = model.training_step(batch,emb=embs.unsqueeze(0))
                else:
                    output_tensors = []
                    for i in range(images.shape[0]):  # Iterate over the batch
                        single_image = images[i].squeeze(0)  # Remove the singleton dimension -> [28, 28]
                        processed_output = bs.embed(single_image, 1000)  # Apply the function
                        print("info about embs:", type(processed_output), processed_output.shape)
                        output_tensors.append(processed_output)

                    # Stack all outputs to create a tensor of shape [32, 435]
                    embs = torch.stack(output_tensors)
                    del output_tensors

                    t_s = time.time()
                    print("info about embs:", type(embs), embs.shape)

                    loss,acc = model.training_step(batch_PQK, batch_normal, labels, emb=embs)

            else:
                loss,acc = model.training_step(batch_PQK, batch_normal, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_losses+=float(loss.detach())
            training_accs+=float(acc.detach())
            if model.embedding_size and step%100==0:
                try:
                    # torch.save({
                    #     'model_statedict': model.state_dict(),
                    #     'optimizer_statedict': optimizer.state_dict()
                    # }, f"./checkpoints mps/epoch_{epoch + 1}_{step_counter_per_epoch + 1}.pth")

                    print(f"fit function newer version; STEP {step}, Training-acc = {training_accs/(step+1)}, Training-losses = {training_losses/(step+1)}")

                except Exception as e:
                    print("Error in saving the checkpoint at path ", f"./checkpoints kk mps/epoch_{epoch + 1}_{step_counter_per_epoch + 1}.pth")
                    print("with error:", e)

                step_counter_per_epoch += 1
        
        ## Validation phase
        result = evaluate_combined_2(model, val_loader, bs)
        validation_loss, validation_acc = result['val_loss'], result['val_acc']
        model.epoch_end(epoch, result)
        history.append(result)

        ## summing up all the training and validation metrics
        training_loss = training_losses/len(train_loader)
        training_accs = training_accs/len(train_loader)
        train_loss.append(training_loss)
        train_acc.append(training_accs)
        val_loss.append(validation_loss)
        val_acc.append(validation_acc)

        # plot training curves
        plot_training_metrics(train_acc,val_acc,train_loss,val_loss)

        # torch.save({
        #     'model_statedict': model.state_dict(),
        #     'optimizer_statedict': optimizer.state_dict()
        # }, f"./checkpoints mps classical/epoch_{epoch + 1}.pth")

        torch.save({
            'model_statedict': model.state_dict(),
            'optimizer_statedict': optimizer.state_dict()
        }, f"./{file_path}/epoch_{epoch + 1}.pth")
    return(history)


# #### LOAD DATA ####
# # dataset from csv file, to use for the challenge
# train_dataset = MNIST_partial(data="/Users/soardr/Perceval 2/Percevel-main/data", split = 'train')
# val_dataset = MNIST_partial(data="/Users/soardr/Perceval 2/Percevel-main/data", split='val')

# # definition of the dataloader, to process the data in the model
# # here, we need a batch size of 1 to use the boson sampler
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
# val_loader = DataLoader(val_dataset, batch_size, shuffle = False)

# #### START SCALEWAY SESSION ####
# session = None
# # to run a remote session on Scaleway, uncomment the following and fill project_id and token
# # session = scw.Session(
# #                    platform="sim:sampling:p100",  # or sim:sampling:h100
# #                    project_id=""  # Your project id,
# #                    token=""  # Your personal API key
# #                    )

# # start session
# if session is not None:
#     session.start()

# #### BOSON SAMPLER DEFINITION ####
# # here, we use 30 photons and 2 modes
# bs = BosonSampler(30, 2, postselect = 2, session = session)
# print(f"Boson sampler defined with number of parameters = {bs.nb_parameters}, and embedding size = {bs.embedding_size}")
# #to display it
# pcvl.pdisplay(bs.create_circuit())

# #### RUNNING DEVICE ####
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else device

# print(f'DEVICE = {device}')

# #### MODEL ####
# # define the model and send it to the appropriate device
# # set embedding_size = bs.embedding_size if you want to use the boson sampler in input of the model
# model = MnistModel(device = device)
# # model = MnistModel(device = device, embedding_size = bs.embedding_size)
# model = model.to(device)
# # train the model with the chosen parameters
# experiment = fit(epochs = 20, lr = 0.005, model_temp=model, train_loader = train_loader, val_loader = val_loader, bs=bs, device=device)

# # end session if needed
# if session is not None:
#     session.stop()
