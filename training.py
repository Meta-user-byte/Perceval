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
from model import MnistModel, evaluate
from boson_sampler import BosonSampler
import perceval as pcvl
#import perceval.providers.scaleway as scw  # Uncomment to allow running on scaleway

#### TRAINING LOOP ####
def fit(epochs, lr, model, train_loader, val_loader, bs: BosonSampler, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    # creation of empty lists to store the training metrics
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for epoch in range(epochs):
        training_losses, training_accs = 0, 0
        ## Training Phase
        for step, batch in enumerate(tqdm(train_loader)):
            # embedding in the BS
            if model.embedding_size:
                images, labs = batch
                images = images.squeeze(0).squeeze(0)
                t_s = time.time()
                embs = bs.embed(images,1000)
                loss,acc = model.training_step(batch,emb = embs.unsqueeze(0))

            else:
                loss,acc = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_losses+=float(loss.detach())
            training_accs+=float(acc.detach())
            if model.embedding_size and step%100==0:
                print(f"STEP {step}, Training-acc = {training_accs/(step+1)}, Training-losses = {training_losses/(step+1)}")
        
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
    return(history)


#### LOAD DATA ####
# dataset from csv file, to use for the challenge
train_dataset = MNIST_partial(split = 'train')
val_dataset = MNIST_partial(split='val')

# definition of the dataloader, to process the data in the model
# here, we need a batch size of 1 to use the boson sampler
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size, shuffle = False)

#### START SCALEWAY SESSION ####
session = None
# to run a remote session on Scaleway, uncomment the following and fill project_id and token
# session = scw.Session(
#                    platform="sim:sampling:p100",  # or sim:sampling:h100
#                    project_id=""  # Your project id,
#                    token=""  # Your personal API key
#                    )

# start session
if session is not None:
    session.start()

#### BOSON SAMPLER DEFINITION ####
# here, we use 30 photons and 2 modes
bs = BosonSampler(30, 2, postselect = 2, session = session)
print(f"Boson sampler defined with number of parameters = {bs.nb_parameters}, and embedding size = {bs.embedding_size}")
#to display it
pcvl.pdisplay(bs.create_circuit())

#### RUNNING DEVICE ####
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE = {device}')

#### MODEL ####
# define the model and send it to the appropriate device
# set embedding_size = bs.embedding_size if you want to use the boson sampler in input of the model
model = MnistModel(device = device)
# model = MnistModel(device = device, embedding_size = bs.embedding_size)
model = model.to(device)
# train the model with the chosen parameters
experiment = fit(epochs = 5, lr = 0.001, model = model, train_loader = train_loader, val_loader = val_loader, bs=bs)

# end session if needed
if session is not None:
    session.stop()
