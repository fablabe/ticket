# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:46:59 2021

@author: Yann
"""

import numpy as np

def gaussianHP(shape, sigma=0.5):
    [U, V]=np.ogrid[-0.5:0.5:1.0/shape[0],-0.5:0.5:1.0/shape[1]];
    H = 1 - np.exp(-(U*U+V*V)/(2*sigma**2))
    if H.shape[1]>shape[1]:
        H = H[:, :-1]
    return H

def ButterworthHP(shape, fc, order=1):
    [U, V]=np.ogrid[-0.5:0.5:1.0/shape[0],-0.5:0.5:1.0/shape[1]];
    H = 1/(1+(fc/(U*U+V*V+1e-7)**.5)**(2*order))
    if H.shape[0]>shape[0]:
        H = H[:-1, :]
    if H.shape[1]>shape[1]:
        H = H[:, :-1]
    H = np.fft.ifftshift(H)
    return H

def homofiltrer(im, fc, order=3):
    # step 1 : log
    lri = np.log(1e-7+im)
    
    #step 2 : FFT
    lRI = np.fft.fft2(lri)
    #lRI = sc.fftshift(lRI)
    
    # step 3 : HP filtering
    H = ButterworthHP(im.shape, fc, order)
    lR = H*lRI
    
    # step 4 : inverse FFT
    lr = np.fft.ifft2(lR)
    
    # step 5 : exp
    r = np.exp(lr)
    
    # step 6 : complex -> real
    r = np.real(r)
    
    # step 7 : -> [0, 1]
    M, m = np.max(r), np.min(r)
    if M!=m: r = (r-m)/(M-m)
    
    return r
