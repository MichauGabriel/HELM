# -*- coding: utf-8 -*-
"""
This file is subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.

Code for training and running the HELM as in
Michau, G., Hu, Y., Palmé, T., & Fink, O. (2020). Feature learning for fault detection in high-dimensional condition monitoring signals. 
Proceedings of the Institution of Mechanical Engineers, Part O: Journal of Risk and Reliability, 234(1), 104-115.

Author: Gabriel Michau

Version: 30/10/2018
"""

import numpy as np

def mapminmax(x,q=0,axis=0,mini=-1,maxi=1):
    """
    nx, paramap = mapminmax(x,q=0,axis=0,mini=-1,maxi=1)

    Description:
        Apply the min-max transform to x such that for each row (if axis=0) or
        each column (if axis=1) the q and 1-q percentile are at min and maxi
    """
    if axis==0:
        xmin=np.percentile(x,q, axis=1, keepdims=True)
        xmax=np.percentile(x, 100-q, axis=1, keepdims=True)
    elif axis==1:
        xmin=np.percentile(x,q, axis=0, keepdims=True)
        xmax=np.percentile(x, 100-q,axis=0, keepdims=True)
    Dx=xmax-xmin
    dx=x-xmin
    if 0 in Dx:
        nx=mini+(maxi-mini)*np.divide(dx,Dx,out=np.zeros_like(dx), where=Dx!=0)
    else:
        nx=mini+(maxi-mini)*dx/Dx
    yield nx
    yield {'xmax':xmax, 'xmin':xmin, 'mini':mini, 'maxi':maxi}

##################################################################################
##################################################################################

def mapminmax_r(paramap,*args):
    """
    Reverse mapminmax operation specified by paramap  to datasets specified in *args
    """
    for ii in range(len(args)):  # Number of Inputs X_ii
        yield (args[ii]-paramap['mini'])/(paramap['maxi']-paramap['mini'])*(paramap['xmax']-paramap['xmin'])+paramap['xmin']

##################################################################################
##################################################################################

def mapminmax_a(paramap,*args):
    """
    Apply mapminmax operation specified by paramap to datasets specified in *args
    """
    Dx=paramap['xmax']-paramap['xmin']
    if 0 in Dx:
        for ii in range(len(args)):  # Number of Inputs X_ii
            yield paramap['mini']+(paramap['maxi']-paramap['mini'])*np.divide(args[ii]-paramap['xmin'],Dx,out=np.zeros_like(args[ii]), where=Dx!=0)
    else:
        for ii in range(len(args)):  # Number of Inputs X_ii
            yield paramap['mini']+(paramap['maxi']-paramap['mini'])*(args[ii]-paramap['xmin'])/Dx
