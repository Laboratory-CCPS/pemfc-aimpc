import torch
from sklearn.metrics import r2_score
from torch.autograd import functional
from torch.func import vmap, jacrev, functional_call
from src.learning.neural_network import Scaled_MLP

# Function for training
def TrainLoop(Net,TrainLoader,lossfunc,optimizer):
    Net.train()

    running_loss = 0

    for batch in TrainLoader:
        features,labels = batch
        
        optimizer.zero_grad()

        pred = Net(features)

        loss = lossfunc(pred,labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.detach().cpu().numpy()

    return running_loss/len(TrainLoader)

# Function for training with scaling
def TrainLoopScaled(Net,TrainLoader,lossfunc,optimizer):
    Net.train()

    running_loss = 0

    for batch in TrainLoader:
        features,labels = batch
        
        optimizer.zero_grad()

        pred = Net(features)

        loss = lossfunc(pred, (labels - Net.label_mean) / Net.label_std)
        loss.backward()

        optimizer.step()

        running_loss += loss.detach().cpu().numpy()

    return running_loss/len(TrainLoader)


# Function for validation
def ValidLoop(Net,ValidLoader,lossfunc):
    Net.eval()

    running_loss = 0
    running_r2 = 0

    for batch in ValidLoader:
        features,labels = batch

        pred = Net(features)

        loss = lossfunc(pred,labels)

        running_loss += loss.detach().cpu().numpy()
        running_r2 += r2_score(labels.detach().cpu().numpy(),pred.detach().cpu().numpy())

    return running_loss/len(ValidLoader),running_r2/len(ValidLoader)

# Function for validation with scaling
def ValidLoopScaled(Net,ValidLoader,lossfunc):
    Net.eval()

    running_loss = 0
    running_r2 = 0

    for batch in ValidLoader:
        features,labels = batch

        pred = Net(features)

        loss = lossfunc(pred, (labels - Net.label_mean) / Net.label_std)

        running_loss += loss.detach().cpu().numpy()
        running_r2 += r2_score((labels.detach().cpu().numpy() - Net.label_mean.cpu().numpy()) / Net.label_std.cpu().numpy(),pred.detach().cpu().numpy())

    return running_loss/len(ValidLoader),running_r2/len(ValidLoader)


def get_jac_func(Net,features):
  params = dict(Net.named_parameters())
  def fmodel(params,features):
    return functional_call(Net, params, features)
  jac = vmap(jacrev(fmodel, argnums=(1)), in_dims=(None,0))(params, features)
  return jac.view(features.size()[0],-1)


def TrainLoopJacobian(Net,TrainLoader,lossfunc,optimizer):
    Net.train()

    running_loss = 0

    for batch in TrainLoader:
        features,labels = batch
        labels_u = labels[:,:1]
        labels_jacobians = labels[:,1:]
        
        optimizer.zero_grad()

        pred = Net(features)
        nn_jacobians = get_jac_func(Net=Net,features=features)

        loss = lossfunc(pred,labels_u,nn_jacobians,labels_jacobians)
        loss.backward()

        optimizer.step()

        running_loss += loss.detach().cpu().numpy()

    return running_loss/len(TrainLoader)


# Function for validation
def ValidLoopJacobian(Net,ValidLoader,lossfunc):
    Net.eval()

    running_loss = 0
    running_r2_u,running_r2_jac = 0,0

    for batch in ValidLoader:
        features,labels = batch
        labels_u = labels[:,:1]
        labels_jacobians = labels[:,1:]

        pred = Net(features)
        nn_jacobians = get_jac_func(Net=Net,features=features)

        loss = lossfunc(pred,labels_u,nn_jacobians,labels_jacobians)

        running_loss += loss.detach().cpu().numpy()
        running_r2_u += r2_score(labels_u.detach().cpu().numpy(),pred.detach().cpu().numpy())
        running_r2_jac+= r2_score(labels_jacobians.detach().cpu().numpy(),nn_jacobians.detach().cpu().numpy())

    return running_loss/len(ValidLoader),running_r2_u/len(ValidLoader),running_r2_jac/len(ValidLoader)