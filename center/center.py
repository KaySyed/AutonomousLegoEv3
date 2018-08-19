import cv2
import numpy as np
from PIL import Image

img = cv2.imread('test.jpg')
y, x = img.shape[:2]

roi = img[]

print('x = ' + str(x) + ' y = ' + str(y))