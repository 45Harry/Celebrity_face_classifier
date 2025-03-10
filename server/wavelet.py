import numpy as np
import pywt
import cv2 as cv

def w2d(img, mode='haar' , level=1):
    imArray = img
    #datatype conversion
    #convert to gray scale
    imArray = cv.cvtColor(imArray,cv.COLOR_BGR2GRAY)
    #convert to float
    imArray = np.float32(imArray)
    imArray /=255
    #compute coefficients
    coeffs = pywt.wavedec2(imArray,mode,level=level)

    #process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    #reconstruction
    imArray_H = pywt.waverec2(coeffs_H,mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H