import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage import data
photo_data = misc.imread("./sunset.jpg")
x,y,z=photo_data.shape ## where z is the RGB dimension
### Method block begin
photo_data[:] = photo_data.mean(axis=-1,keepdims=1)
### Method Block ends
plt.figure(figsize=(10,20))
plt.imshow(photo_data)