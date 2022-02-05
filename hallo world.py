import collections
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def funktion(list,num):
    n = len(list)
    left = 0
    right = n-1
    m = (left+right)//2
    while left < right-1:
        if list[m] < num:
            left = m
        else:
            right = m
        m = (left+right)//2
        print(left , right)
    if list[left] >= num:
        m = left
    list = list[0:m+1]+[num]+list[m+1:]

    return list

a = [1,2,3,4,5]
num = 3
a = funktion(a,num)
#print(a)