# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 22:35:28 2018

@author: marek
"""
__spec__ = None
import multiprocessing as mp, numpy
import random

def child(n):
    numpy.random.seed(n)
    m = numpy.random.randn(6)
    return m

if __name__ == '__main__':
    N = 20
    pool = mp.Pool()
    results = pool.map(child, range(N))
    for res in results:
        print (res)