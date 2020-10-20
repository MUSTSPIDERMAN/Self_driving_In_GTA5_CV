# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:51:55 2020

@author: 11037
"""

import pyautogui
import cv2
import numpy as np

while True:
    img = pyautogui.screenshot(region=[0, 0, 100, 100])  # x,y,w,h
    img.save('screenshot.png')
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("test",img)
    cv2.waitKey(1)
