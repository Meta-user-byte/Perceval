import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import pandas as pd
from boson_sampler import BosonSampler
from utils import accuracy
import perceval as pcvl
import perceval.providers.scaleway as scw  # Uncomment to allow running on scaleway


class MnistModel(nn.Module):
    def __init__(self, device = 'cpu', embedding_size = 0):
        super().__init__()
        input_size = 28 * 28
        num_classes = 10
        self.device = device
        self.embedding_size = embedding_size
        if self.embedding_size:
            input_size += embedding_size #considering 30 photons and 2 modes
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, xb, emb = None):
        xb = xb.reshape(-1, 784)
        if self.embedding_size and emb is not None:
            # concatenation of the embeddings and the input images
            xb = torch.cat((xb,emb),dim=1)
        out = self.linear(xb)
        return(out)
    
    def training_step(self, batch, emb = None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device)) ## Generate predictions
        else:
            out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels) ## Calculate the loss
        acc = accuracy(out, labels)
        return loss, acc
    
    def validation_step(self, batch, emb =None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device)) ## Generate predictions
        else:
            out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return({'val_loss':loss, 'val_acc': acc})
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        return result['val_loss'], result['val_acc']


# evaluation of the model
def evaluate(model, val_loader, bs: BosonSampler = None):
    if model.embedding_size:
        outputs = []
        for step, batch in enumerate(tqdm(val_loader)):
            # embedding in the BS
            # outputs.append(model.validation_step(batch, emb=embs.unsqueeze(0)))


            images, labs = batch
            print("inside boson sampler; images and labs data:", images.shape, labs.shape)

            batch_size = images.shape[0]

            if batch_size == 1:
                images = images.squeeze(0).squeeze(0)
                print("shape of images after squeezed:", images.shape)

                embs = bs.embed(images,1000)

                # t_s = time.time()
                print("info about embs:", type(embs), embs.shape)

                # loss,acc = model.validation_step(batch,emb=embs.unsqueeze(0))
                outputs.append(model.validation_step(batch,emb=embs.unsqueeze(0)))
            else:
                output_tensors = []
                for i in range(images.shape[0]):  # Iterate over the batch
                    single_image = images[i].squeeze(0)  # Remove the singleton dimension -> [28, 28]
                    processed_output = bs.embed(single_image, 1000)  # Apply the function
                    output_tensors.append(processed_output)

                # Stack all outputs to create a tensor of shape [32, 435]
                embs = torch.stack(output_tensors)
                del output_tensors

                # t_s = time.time()/
                print("info about embs:", type(embs), embs.shape)

                # loss,acc = model.validation_step(batch, emb=embs)
                outputs.append(model.validation_step(batch, emb=embs))
    else:
        outputs = [model.validation_step(batch) for batch in val_loader]
    return(model.validation_epoch_end(outputs))

def evaluate_combined_2(model, val_loader, bs: BosonSampler = None):
    if model.embedding_size:
        outputs = []
        for step, batch in enumerate(tqdm(val_loader)):
            # embedding in the BS
            # outputs.append(model.validation_step(batch, emb=embs.unsqueeze(0)))


            images_PQK, images_normal, labs = batch
            print("inside boson sampler; images and labs data:", images_PQK.shape, images_normal.shape, labs.shape)

            batch_size = images_PQK.shape[0]

            if batch_size == 1:
                images = images.squeeze(0).squeeze(0)
                print("shape of images after squeezed:", images.shape)

                embs = bs.embed(images,1000)

                # t_s = time.time()
                print("info about embs:", type(embs), embs.shape)

                # loss,acc = model.validation_step(batch,emb=embs.unsqueeze(0))
                outputs.append(model.validation_step(batch,emb=embs.unsqueeze(0)))
            else:
                output_tensors = []
                for i in range(images.shape[0]):  # Iterate over the batch
                    single_image = images[i].squeeze(0)  # Remove the singleton dimension -> [28, 28]
                    processed_output = bs.embed(single_image, 1000)  # Apply the function
                    output_tensors.append(processed_output)

                # Stack all outputs to create a tensor of shape [32, 435]
                embs = torch.stack(output_tensors)
                del output_tensors

                # t_s = time.time()/
                print("info about embs:", type(embs), embs.shape)

                # loss,acc = model.validation_step(batch, emb=embs)
                outputs.append(model.validation_step(batch, emb=embs))
    else:
        outputs = [model.validation_step(item_cols[0], item_cols[1], item_cols[2]) for item_cols in val_loader]
    return(model.validation_epoch_end(outputs))
