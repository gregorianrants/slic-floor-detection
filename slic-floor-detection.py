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

# bgr = cv2.imread('roi.jpg')
# rgb = io.imread('roi.jpg')

lab = cv2.cvtColor(rgb,cv2.COLOR_BGR2Lab)

(height,width,_) = lab.shape

#rgb = cv2.resize(rgb,(width//5,height//5),interpolation= cv2.INTER_LINEAR)

top_left,bottom_right = get_corners(rgb,(35,80),(65,100))

cv2.rectangle(bgr,top_left,bottom_right,3)

cv2.imshow('img',bgr)

before = time.time()
segments_slic = slic(rgb, n_segments=50, compactness=10, sigma=1,
                     start_label=1)
after = time.time()
print(after-before)

all_labels = np.unique(segments_slic)
sample_labels = get_slice(segments_slic,top_left,bottom_right)
sample_labels = np.unique(sample_labels)






fig,ax = plt.subplots()

ax.imshow(mark_boundaries(rgb,segments_slic))




plt.show()
cv2.waitKey(0)

