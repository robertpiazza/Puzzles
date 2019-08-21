# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:19:35 2019

@author: 593787
"""

import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time 

big = cv.imread('gap.jpg')
tryList = ['IMG_0277_inside.jpg', 'IMG_0286_background.jpg', 'IMG_0286_inside.png', 'IMG_0226.jpeg']
small = cv.imread(tryList[0])
newBig = big.copy()
threshold = 5
for i, y  in enumerate(newBig):
    for j, x in enumerate(y):
        if np.min(np.max(np.abs(small-x), axis = 2)) > threshold:
            newBig[i, j] = [0, 0, 0]

plt.imshow(newBig)