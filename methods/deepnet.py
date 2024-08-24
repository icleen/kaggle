from tqdm import tqdm
import os, sys
import os.path as osp
import time
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
pd.options.mode.copy_on_write = True

# import sklearn as skl
# from sklearn import tree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader


class Mish(nn.Module):
    """
    Mish activation function from:
    "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    https://arxiv.org/abs/1908.08681v1
    x * tanh( softplus( x ) )
    """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh( F.softplus(x) )


class Sine(nn.Module):
    """
    Sine activation function from:
    "Sine: Implicit Neural Representations with Periodic Activation Functions"
    https://www.vincentsitzmann.com/siren/
    sine(x)
    """

    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)


activations = {'relu': nn.ReLU, 'mish': Mish, 'selu': nn.SELU, 'elu': nn.ELU, 'sine': Sine}


class FCNet(nn.Module):
    """Fully Connected Network"""

    def __init__(self, infeats=10, outfeats=10, layers=[15, 10, 15], activation='relu',
        lastivation=None, batchnorm=False, layernorm=False, dropout=0
    ):
        super(FCNet, self).__init__()
        self.infeats = infeats
        self.outfeats = outfeats
        self.layers = layers
        if isinstance(activation, str):
            activation = activations[activation]
        self.activation = activation
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.dropout = dropout

        model = []
        layers = [infeats] + layers
        for ix, layer_size in enumerate(layers[1:]):
            model.append(nn.Linear(layers[ix], layer_size))
            if layernorm:
                model.append(nn.LayerNorm(layer_size))
            model.append(self.activation())
            if batchnorm and ix < (len(layers)-1):
                model.append(nn.BatchNorm1d(layer_size))
            if dropout > 0 and ix < (len(layers)-1):
                model.append(nn.Dropout(dropout))
        model.append(nn.Linear(layers[-1], outfeats))
        self.model = nn.Sequential( *model )
        self.lastivation = lastivation

    def forward(self, xdata):
        ydata = self.model(xdata)
        if self.lastivation is not None:
            ydata = self.lastivation(ydata)
        return ydata
    
    def loss_fn(self, xdata, target):
        pred = self(xdata)
        return F.binary_cross_entropy_with_logits(pred, target)

    def evaluate(self, dataset):
        loss = 0
        acc = 0
        for d, t in dataset:
            d = d.to(self.get_device())
            t = t.to(self.get_device())
            pred = F.sigmoid(self(d))
            # import pdb; pdb.set_trace()
            acc += ((pred > 0.6) == t).sum().item() / len(d)
            loss += self.loss_fn(d, t).item()
        return loss / len(dataset), acc / len(dataset)
    
    def get_device(self):
        return next(self.parameters()).device
    

def train(model, dataset, evalset, total_epochs=10, eval_batch=25, save_batch=25, save_after=50, optim=None, clip_grad=None, lr_scheduler=None, save_folder='results', model_name='deepnet'):
    model.train()
    train_losses = []
    valid_losses = []
    valid_acces = []
    device = model.get_device()
    if optim is None:
        lr = 5e-4
        optim = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=5e-3
        )

    def save_model(epoch):
        checkpoint = {}
        checkpoint['model'] = model.state_dict()
        checkpoint['optim'] = optim.state_dict()
        checkpoint['train_losses'] = train_losses
        checkpoint['valid_losses'] = valid_losses
        checkpoint['valid_acces'] = valid_acces
        checkpoint['epoch'] = epoch
        filename = os.path.join(save_folder, '{}_e{}.torch'.format(model_name, best_epoch))
        torch.save(checkpoint, filename)
        plt.plot(train_losses)
        plt.plot(valid_losses)
        plt.plot(valid_acces)
        filename = os.path.join(save_folder, '{}_e{}.png'.format(model_name, best_epoch))
        plt.savefig(filename)
        plt.clf()
        print('saved to', filename)

    for epoch in range(0, total_epochs + 1):
        loss = 0.
        mean_loss = 0.
        mean_acc = 0.
        pbar = tqdm(total=len(dataset), file=sys.stdout)
        desc = f'Train Epoch {epoch}'
        pbar.set_description(desc)
        for iter, (batch, target) in enumerate(dataset):
            batch = batch.to(device)
            target = target.to(device)
            loss = model.loss_fn(batch, target)
            optim.zero_grad()
            loss.backward()
            if clip_grad is not None:
                clip_grad(model.parameters())
            optim.step()
            mean_loss += loss.item()

            ploss = mean_loss / (iter + 1)
            desc = f'Train Epoch {epoch}: loss={ploss:.3f}'
            pbar.set_description(desc)
            pbar.update(1)
        pbar.close()
        train_losses.append(mean_loss / len(dataset))

        if epoch % eval_batch == 0:
            eval_time = time.time()
            vloss, vacc = model.evaluate(evalset)
            valid_losses.append(vloss)
            valid_acces.append(vacc)
            print('eval time:', time.time() - eval_time)
            print('eval acc:', vacc)
            model.train()
        else:
            if len(valid_losses) > 0:
                valid_losses.append(valid_losses[-1])
                valid_acces.append(valid_acces[-1])
            else:
                valid_losses.append(train_losses[-1])
                valid_acces.append(0)

        if epoch >= save_after and epoch % save_batch == 0:
            save_model(epoch)

        if lr_scheduler is not None and (epoch == 100 or epoch == 200):
            lr_scheduler.step()

    best_epoch = np.argmax(valid_acces)
    print('best epoch:', best_epoch, valid_acces[best_epoch])
    filename = os.path.join(save_folder, '{}_e{}.torch'.format(model_name, best_epoch))
    model_info = torch.load(filename)
    torch.save(model_info, os.path.join(save_folder, 'best_model_{}.torch'.format(model_name)))

