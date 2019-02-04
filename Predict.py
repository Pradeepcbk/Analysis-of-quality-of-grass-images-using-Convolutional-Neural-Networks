# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:53:33 2019

@author: pradeep
"""
import numpy as np
import os
import glob
import cv2
import xlsxwriter

coordinatePoints = []
l = []
def predict(network, dataSet):  
    img = cv2.imread('test.png')
    book = xlsxwriter.Workbook('prediction.xlsx')
    sheet = book.add_worksheet()
    column = 0
    xColumn = 1
    yColumn = 2
    valueColumn = 3
    row = 0
    itr = 0
    for x in range(0,1700,128):
        for y in range(0,1400,128):
            print('x: ' + str(x) + 'y' + str(y))
            row = row + 1
            img1 = img[x:x+128,y:y+128]
            img3 = np.expand_dims(img1, axis = 0)
            img2 = img3/255
            classes = network.predict_classes(img2)
            valueInExcel = [x,y,(int)(classes[0])]
            sheet.write(row, xColumn, x)
            sheet.write(row, yColumn, y)
            sheet.write(row,valueColumn, valueInExcel[2])
            itr = itr + 1
# =============================================================================
#         column = column + 3
#         xColumn = column + 1
#         yColumn = column + 2
#         valueColumn = column + 3
# =============================================================================
    book.close()
    
    var = 'Moss' if (classes[[0]] == 0) else 'Grass'

# =============================================================================
#     list_of_files = glob.glob('*.jpg') # * means all if need specific format then *.csv
#     latest_file = max(list_of_files, key=os.path.getctime)
#     print('The evaluated file is: ' + str(latest_file))
# =============================================================================
