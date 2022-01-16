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

def FISTA_l2_l1(A,b,lam,itrs):
    """
    Solve argmin_x  ||A*x - b||² + lam |x|
    INPUT:
        A   : matrix of size p.m
        b   : matrix of size p.n
        lam : scalar
        itrs: number of iterations.
              If set to <1, algorithm look for covergence on iterates (if <1500)
              If negative, use default value 1e-5

    OUTPUT:
        x: matrix of size m.n
    """
    if not isinstance(A,np.ndarray):
        raise ValueError("First input should be of type numpy.ndarray (of dim 2)")
    if A.shape.__len__()<2:
        raise ValueError("First input should be of dimension 2")
    if not isinstance(b,np.ndarray):
        raise ValueError("Second input should be of type numpy.ndarray (of dim 2)")
    if b.shape.__len__()<2:
        raise ValueError("Second input should be of dimension 2")

    gamma=0.9/(1+np.linalg.norm(A.T.dot(A),ord=2))
    m=A.shape[1]
    n=b.shape[1]
    x=np.zeros((m,n))
    yk=x
    tk=1
    p1=2*(A.T).dot(A)
    p2=-2*(A.T).dot(b)

    if itrs<=1:
        if itrs<=0:
            itrs=1e-5
        itr_conv=True
        thrshld = itrs
        itrs=15000
    else:
        itr_conv=False

    for i in range(itrs):
        ck=yk-gamma*(p1.dot(yk)+p2)
        x1=np.maximum(abs(ck)-lam*gamma,0)*np.sign(ck)
        tk1=0.5+0.5*(1+4*tk**2)**0.5
        tt=(tk-1)/tk1
        yk=x1+tt*(x-x1)
        tk=tk1
        iter_diff=(1/x.size*np.sum((x1-x)**2))**0.5
        x=x1
        if itr_conv and iter_diff < thrshld:
            break
    return x
