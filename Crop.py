# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:38:14 2019

@author: pradeep
"""
import cv2

def crop():
    img = cv2.imread('53.jpg')
    i = 1000
    for y in range(0,3904,128):
        for x in range(0,2896,128):
            img2 = img[x:x+128,y:y+128]
            cv2.imwrite('test/'+str(i)+str('.jpg'), img2)
            i = i + 1
            
crop()