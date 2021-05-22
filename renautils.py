# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:00:03 2021

@author: Yann
"""
from math import floor, ceil
import sys
import matplotlib.pyplot as plt

def show(*args, nrows='auto', ncols='auto', title=None):
    ''' 
    Example :
        show(im1, "Image 1", im2, "Image 2", im3, im4, ..., nrows=2, ncols='auto', title="Mes images") 
    '''
    n = len(args)
    for arg in args:
        if type(arg) is str:
            n -= 1
    if nrows=='auto' and ncols=='auto':
        nrows = floor(n**.5)
        ncols = ceil(n**.5)
    elif nrows=='auto':
        nrows = ceil(n / ncols)
    elif ncols=='auto':
        ncols = ceil(n / nrows)
    elif nrows*ncols < n:
        print("show: Not enough cells: defaulting to auto.", file=sys.stderr)
        nrows = floor(n**.5)
        ncols = ceil(n**.5)
        
    if nrows*ncols < n: # occurs when n=3
        nrows+=1
    
    plt.figure()
    
    i=1
    for arg in args:
        if type(arg) is tuple :
            plt.subplot(nrows, ncols, i)
            plt.imshow(arg[0])
            i+=1
            plt.title(arg[1])
            
        elif type(arg) is str :
            plt.title(arg)
        
        else :
            plt.subplot(nrows, ncols, i)
            plt.imshow(arg)
            i+=1
    
    if title!=None:
        plt.suptitle(title)
    
    plt.show()