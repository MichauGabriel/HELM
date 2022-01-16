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


#%%
train_x     = np.random.rand(6000,200)
vali_x      = np.random.rand(1000,200)
test_x      = np.random.rand(2000,200)
fault_id    = random.sample(range(200), 5)
test_x[1001:2000,fault_id]=test_x[1001:2000,fault_id]+2

train_x,paramap = mmm.mapminmax(train_x, q=0, axis=1, mini=-1, maxi=1)
vali_x,test_x  = mmm.mapminmax_a(paramap, vali_x, test_x)
# Hyperparameters of HELM
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

# Define a detection threshold based on the quantile of the validation set
quant = 99.9;
thr   = 1.5 * np.percentile(out['val']['Y'],quant)


f1 = plt.figure(1)
f1.clf()
ax = f1.gca()
ax.plot(range(6000),      out['train']['Y']/thr, linestyle='None', marker='o', markerfacecolor='none')
ax.plot(range(6000,7000), out['val']['Y']  /thr, linestyle='None', marker='o', markerfacecolor='none')
ax.plot(range(7000,9000), out['test']['Y'] /thr, linestyle='None', marker='o', markerfacecolor='none')
ax.plot([0,9000],[1,1],color='k')
plt.show()
