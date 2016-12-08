import tensorflow as tf 
import numpy as np
from PIL import Image
from skimage import io
from skimage import data_dir

#im = Image.open('PTZfish.tiff')
#im=io.imread('PTZfish.tiff')
#imarray=np.array(im)

#print(imarray.shape)

#print(imarray.size)


#img = MultiImage(data_dir + 'PTZfish.tiff') 
#print(im.shape)

img = Image.open('PTZfish.tiff')
img.seek(0)
n=img.n_frames

x,y=(np.array(img)).shape
imarray=np.zeros((n,x,y))

for i in range(n):
    try:
        img.seek(i)
        #print img.getpixel( (1, 0))
        imarray[i]=np.array(img)
        #print(imarray)
    except EOFError:
        # Not enough frames in img
        break

print(imarray.shape)
print(imarray[1])