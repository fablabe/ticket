# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:46:59 2021

@author: Yann
"""

import numpy as np

def ButterworthHP(shape, fc, order=1):
    ''' 2D Butterworth high-pass filter '''
    [U, V]=np.ogrid[-0.5:0.5:1.0/shape[0],-0.5:0.5:1.0/shape[1]];
    H = 1/(1+(fc/(U*U+V*V+1e-7)**.5)**(2*order))
    if H.shape[0]>shape[0]:
        H = H[:-1, :]
    if H.shape[1]>shape[1]:
        H = H[:, :-1]
    H = np.fft.ifftshift(H)
    return H

def homomorphic_filtering(img, fc, order=3):
    
    img_log = np.log(img + 1e-24)
    img_log_fft = np.fft.fft2(img_log)
    H = 1.5*ButterworthHP(img.shape, fc, order) + 0.5
    prod = H*img_log_fft
    img_log_filtered = np.real(np.fft.ifft2(prod))
    img_filtered = np.exp(img_log_filtered)
    _m, _M = np.min(img_filtered), np.max(img_filtered)
    return (img_filtered - _m) / (_M - _m) if _m!=_M else np.zeros(img.shape, dtype=img.dtype)
