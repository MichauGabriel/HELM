"""
This file is subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.

Code for training and running the HELM as in
Michau, G., Hu, Y., Palmé, T., & Fink, O. (2020). Feature learning for fault detection in high-dimensional condition monitoring signals. 
Proceedings of the Institution of Mechanical Engineers, Part O: Journal of Risk and Reliability, 234(1), 104-115.

Author: Gabriel Michau

Version: 13/03/2019
"""

import numpy as np
from lib import mapminmax as mmm
from lib import FISTA_l2_l1 as fista

def HELM(para,train_x,train_y=1):
    """
    [model] = HELM(para,train_x,train_y)

    Train one or more HELM with input data train_x and output 1 and return a structure with all elements needed for running the model(s)

    INPUTS
     para : a dictionary with parameters for the HELM. Can be empty (default values will apply)
        para.neuron_number  : vector, each element is the number of neurones of each layer.
							  All layers are Auto-encoders but the last
                              default : [10 50]
        para.ridge_weight   : Ridge paramter C for regularisation in ELM layer
                              argmin_B  ||HB - T ||² + C ||B||²
                              default: 1e-5
        para.fista_weight   : weight factor of l1-norm in autoencoder layers
                              argmin_B  ||H*B - T||² + lam |B|
                              default: 1e-5
        para.fista_cv       : Criteria for FISTA convergence type.
                                 If >1 then represents number of iterations
                                 If <1 then represents threshold for RMSE between successive estimates. When reached, FISTA stops
                              default: 1e-5
        para.nhelm          : number of trained HELM. (useful to average results)
                              default: 5
     train_x : training data set, N by D (N samples, D dimensions)
	 train_y : training data label. For learning the one-class classifier, leave empty or set to 1.
                                    Otherwise of size N by M (N samples, M dimensions).


    OUTPUTS:
      model           : structure of size para.nhelm containing the HELM weights and neurons
      model{'type'}   : specify is HELM is used for 1-class classification (train_y=1) or regression (train_y of size N by M)
          model[k].beta[i]   : output weight of i-th layer of model k
          model[k].ps[i]     : normalization operation on neurons outputs of i-th layer of model k
          model[k].elmweight : input weight of the last layer (one class classifier ELM)
    """
    model={}

    if 'neuron_number' not in para:
        raise ValueError("Please specify network structure")

    if 'nhelm' not in para:
        print("Number of helm to run not specified. Continue with 5")
        para['nhelm'] = 5 # number of times HELM is trained and ran
    if 'fista_weight' not in para:
        print("Weight for l1 regularization not specified. Continue with 1e-3")
        para['fista_weight']  = 1e-5               # weight factor lambda
    if 'fista_cv' not in para:
        print("Convergence parameter for FISTA not specified. Continue with 1e-5")
        para['fista_cv']      =1e-5                       # Number of iterations or RMSE
    if 'ridge_weight' not in para:
        print("Weight for l2 regularization not specified. Continue with 1e-5")
        para['ridge_weight']  =1e-5

    if train_y==1:
        train_y=np.ones((train_x.shape[0],1))
        model['type']='1C'
    elif train_y.shape[0] != train_x.shape[0]:
        raise ValueError("Training data and output dimensions are not compatible")
    else:
        model['type']='Regr'

    number_of_AE_layers=para['neuron_number'].size-1

    for helm in range(para['nhelm']):  # Number of trained HELM
        # Auto-Encoder
        model[helm]={}
        model[helm]['beta']={}
        model[helm]['beta'][0]={}
        model[helm]['ps']={}
        model[helm]['ps'][0]={}
        x=np.copy(train_x)
        for i in range(number_of_AE_layers): # Number of AE-Layers
            x   = np.hstack((x, 0.1*np.ones((x.shape[0],1))))  # Add bias b
            w   = 2*np.random.rand(x.shape[1],para['neuron_number'][i])-1  # Generate random weights, uniform distribution between -1 and 1
            h,_ = mmm.mapminmax(np.tanh(x.dot(w)),axis=1) # Activation function and normalization, H hidden layer matrix
            model[helm]['beta'][i] = fista.FISTA_l2_l1(h, x, para['fista_weight'], para['fista_cv']) # argmin beta with FISTA
            x,model[helm]['ps'][i] = mmm.mapminmax(x.dot(model[helm]['beta'][i].T),axis=1)   # T^Train; save xmin and xmax

        # 1 class classifier ELM/ of ELM regressor
        n   = para['neuron_number'][len(para['neuron_number'])-1]
        x   = np.hstack((x, 0.1*np.ones((x.shape[0],1))))                                     # Add bias
        w,_ = np.linalg.qr(2*(np.random.rand(x.shape[1],n)).T-1) #sp.linalg.orth()  #Generate random weights
        w   = w.T
        if w.shape[0] < x.shape[1]:
            w = np.vstack((w,2*np.random.rand(x.shape[1]-w.shape[0],n)-1))
        h = np.tanh(x.dot(w))
        model[helm]['beta'][number_of_AE_layers] = np.linalg.solve((h.T.dot(h)+np.eye(h.shape[1])*para['ridge_weight']),h.T.dot(train_y))
        model[helm]['elmweight'] = w
    return model

def HELM_run(model,**kwargs):
    """
	out = HELM_run(model,**kwargs).

	Run HELM for model derived from [model] = HELM(para,train_x,train_y).

	INPUT:
	    model : HELM model(s) from function HELM
		**kwargs: as many datasets to run the model on as wished.
		          The name of the variable is used as key to the output dictionary
	OUTPUT:
		out : dictionary with average residuals and classifier output over nhelm (number of models).
			out{key}['Y']   : average output of the last layer for dataset with name 'key'. out{key}['Y'] is of size N by M.
			out{key}['Res'] : average residual of the first auto-encoder for dataset with name 'key'. out{key}['Res'] is of size N by D.
    """
    number_of_AE_layers=len(model[0]['beta'])-1
    nhelm=len(model)-1
    out={}


    for key in kwargs:
        out[key]         = {}
        out[key]['Y']    = np.zeros((kwargs[key].shape[0],model[0]['beta'][number_of_AE_layers].shape[1]))
        out[key]['Res']  = np.zeros(kwargs[key].shape)

        for helm in range(nhelm):
            # Auto-Encoder
            x=np.copy(kwargs[key])   # e.g. x=test_x
            for i in range(number_of_AE_layers):
                x = np.hstack((x, 0.1*np.ones((x.shape[0],1)))).dot(model[helm]['beta'][i].T)
                if i==0:
                    out[key]['Res']+=np.abs(x.dot(model[helm]['beta'][i][:,:-1])-kwargs[key])/nhelm
		x, = mmm.mapminmax_a(model[helm]['ps'][i], x)
            # 1 class classifier ELM
            x = np.hstack((x, 0.1*np.ones((x.shape[0],1))))
            h = (np.tanh(x.dot(model[helm]['elmweight']))).dot(model[helm]['beta'][number_of_AE_layers])
            if model['type']=='1C':
                out[key]['Y']+=np.abs(1-h)/nhelm
            else:
                out[key]['Y']+=h/nhelm
    return out
