import cv2
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from square import get_corners,get_slice
import time

bgr = cv2.imread('roboview.jpg')
rgb = io.imread('roboview.jpg')

bgr = cv2.imread('roi.jpg')
rgb = io.imread('roi.jpg')





(height,width,_) = bgr.shape

rgb = cv2.resize(rgb,(width//5,height//5),interpolation= cv2.INTER_LINEAR)

lab = cv2.cvtColor(bgr,cv2.COLOR_BGR2Lab)

cv2.imshow('img',bgr)

before = time.time()
segments_slic = slic(rgb, n_segments=100, compactness=10, sigma=1,
                     start_label=0)
after = time.time()
print(after-before)

all_labels = np.unique(segments_slic)
# sample_labels = get_slice(segments_slic,top_left,bottom_right)
# sample_labels = np.unique(sample_labels)



def mean(label):
    return rgb[segments_slic==label].mean(dtype='uint8',axis=0)



#print(all_labels)


means = np.array([mean(label) for label in all_labels])
print('means',means.shape)



#print(means)


def doit(label):
    print(label)
    segments_slic[segments_slic == label,] = means[label]

result = np.take(means,segments_slic)

print('result',result.shape)
print('fuck')




cv2.imshow('result',result)



print(result.shape)


fig,ax = plt.subplots()
#
ax.imshow(mark_boundaries(rgb,segments_slic))
#
#
#
#
plt.show()
cv2.waitKey(0)