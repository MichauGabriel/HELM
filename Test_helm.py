"""
This file is subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.

Code for training and running the HELM as in
Michau, G., Hu, Y., Palmé, T., & Fink, O. (2020). Feature learning for fault detection in high-dimensional condition monitoring signals.
Proceedings of the Institution of Mechanical Engineers, Part O: Journal of Risk and Reliability, 234(1), 104-115.

Author: Dr. Gabriel Michau

Version: 30/10/2018
"""

import numpy as np
import random
import matplotlib.pyplot as plt

from lib import mapminmax as mmm
from lib import HELM


#%% Generating random data
train_x     = np.random.rand(6000,200)
vali_x      = np.random.rand(1000,200)
test_x      = np.random.rand(2000,200)
## Applying a simple simulated fault
fault_id    = random.sample(range(200), 5)
test_x[1001:2000,fault_id]=test_x[1001:2000,fault_id]+2

#%% Preprocessing:
## It is good practice to re-map all data/features between -1 and 1.

## The mapping is learned on the training set...
train_x,paramap = mmm.mapminmax(train_x, q=0, axis=1, mini=-1, maxi=1)
## ... and applied on the validation and test sets
vali_x,test_x  = mmm.mapminmax_a(paramap, vali_x, test_x)

#%% Hyperparameters of HELM
## cf related paper to get more details on the setting of the hyper-parameters
para={}
para['nhelm']         = 5                  # number of times HELM is trained and ran
para['neuron_number'] = np.array([20,100]) # HELM structure : 1 AE of size 20, one 1-class classifier of size 100
para['fista_weight']  = 1e-3               # weight for AE sparse regularization
para['fista_cv']      = 1e-5               # Number of iterations (if >1) or max RMSE on iterates (if <1)
para['ridge_weight']  = 1e-5               # weight for last layer regularization

#%% Train HELM
model=HELM.HELM(para,train_x)
#%% Run HELM on the three datasets
out=HELM.HELM_run(model,train=train_x,val=vali_x,test=test_x)

#%% Plot results

## Define a detection threshold based on the quantile of the validation set
quant = 99.9;
thr   = 1.5 * np.percentile(out['val']['Y'],quant)


fig = plt.figure(1)
fig.clf()
ax = fig.gca()
ax.plot(range(6000),      out['train']['Y']/thr, linestyle='None', marker='o', markerfacecolor='none')
ax.plot(range(6000,7000), out['val']['Y']  /thr, linestyle='None', marker='o', markerfacecolor='none')
ax.plot(range(7000,9000), out['test']['Y'] /thr, linestyle='None', marker='o', markerfacecolor='none')
ax.plot([0,9000],[1,1],color='k')
plt.show()

#%% Finding the number of neurons to use in the auto-encoder

## Number of neurons in the auto-encoding layer to test
nAElist = [1,2,4,8,12,16,20,24,32,64,128,256,350,450,550]
## Resulting mean residual placeholder
meanRes = np.zeros((nAElist.__len__()))

for i_exp,nAE in enumerate(nAElist):
    para['neuron_number'] = np.array([nAE,1])
    model=HELM.HELM(para,train_x)
    out=HELM.HELM_run(model,val=vali_x)
    meanRes[i_exp] = out['val']['Res'].mean()

fig = plt.figure(2)
fig.clf()
plt.plot(nAElist,meanRes)
## 256 should be in the elbow of the resulting L-curve, it's a good compromise between reconstruction and number of neurons
nAE = 256
#%% Finding the number of neuron to use in the 1-class classifier

## Number of neurons in the auto-encoding layer to test
n1Clist = [1,10,50,100,150,200,250,300,350,450,550]
## Resulting mean residual placeholder
meanAS = np.zeros((n1Clist.__len__()))

for i_exp,n1C in enumerate(n1Clist):
    para['neuron_number'] = np.array([nAE,n1C])
    model=HELM.HELM(para,train_x)
    out=HELM.HELM_run(model,val=vali_x)
    meanAS[i_exp] = out['val']['Y'].mean()

fig = plt.figure(3)
fig.clf()
plt.plot(n1Clist,meanAS)
## 200-250 should be in the elbow of the resulting L-curve, it's a good compromise between reconstruction and number of neurons
n1C = 250
#%% Runnning helm with these parameters (you should get a better separation faulty/non fautly)
## and finding which features are most likely faulty through the residual

para['neuron_number'] = np.array([nAE,n1C])

out=HELM.HELM_run(model,train=train_x,val=vali_x,test=test_x)

thr   = 1.5 * np.percentile(out['val']['Y'],quant)

fig = plt.figure(4)
fig.clf()
ax = fig.gca()
ax.plot(range(6000),      out['train']['Y']/thr, linestyle='None', marker='o', markerfacecolor='none')
ax.plot(range(6000,7000), out['val']['Y']  /thr, linestyle='None', marker='o', markerfacecolor='none')
ax.plot(range(7000,9000), out['test']['Y'] /thr, linestyle='None', marker='o', markerfacecolor='none')
ax.plot([0,9000],[1,1],color='k')
plt.show()


abnormal_samples = out['test']['Y'] / thr > 1

fig = plt.figure(5)
fig.clf()
ax = fig.gca()
ax.plot(range(test_x.shape[1]), out['test']['Res'][abnormal_samples.squeeze(),:].mean(axis=0), linestyle='None', marker='o', markerfacecolor='none')
## Plot against actually fault id (from the random draw in the data generation cell)
for i in range(5):
    ax.plot([fault_id[i],fault_id[i]],[0,4],'r-')
plt.show()

