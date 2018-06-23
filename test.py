# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:36:43 2018

@author: marek
"""

__spec__ = None
import multiprocessing as mp

def f(n):
    return n*n

array = [0,1,2,3,4,5]


if __name__ == '__main__':
    p = mp.Pool()
    results = p.map(f, array)
    print(results)
    p.close()
    p.join()
    

