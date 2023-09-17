import cv2
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from square import get_corners,get_slice
import time
from helpers import normalise,mean,std,get_roi,get_contour_for_label,getContours,getWidth,getWidths,getHeights
from scipy.spatial import distance

# rgb = io.imread('roboview2.jpg')

cap = cv2.VideoCapture('video2.avi')
time.sleep(1)

resultWindow = cv2.namedWindow('result')

threshold = 0.2

def setThreshold(value):
  global threshold
  threshold = value/100
  print(threshold)


cv2.createTrackbar('%d','result',2,300,setThreshold)





while(cap.isOpened()):
  print('fuck')

  ret, bgr = cap.read()
  (height,width,_) = bgr.shape
  rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  rgb_small = cv2.resize(rgb,(width//5,height//5),interpolation= cv2.INTER_LINEAR)
  hsv_small = cv2.cvtColor(rgb_small,cv2.COLOR_RGB2HSV)
  threshold_image = hsv_small
  image = cv2.bilateralFilter(rgb_small, 9, 75, 75)
  segments_slic = slic(rgb_small, n_segments=50, compactness=10, sigma=1,start_label=0,convert2lab=True)


  #rgb_small = normalise(rgb_small)

  all_labels = np.unique(segments_slic)

  roi_labels = get_roi(threshold_image,segments_slic,(45,90),(65,99))


  unknown_labels = all_labels[roi_labels]

  means = np.array([mean(label,threshold_image,segments_slic) for label in all_labels])
  standard_deviations = np.array([std(label,threshold_image,segments_slic) for label in all_labels])

  contours = getContours(all_labels,segments_slic)
  widths = getWidths(contours)

  heights = getHeights(contours)


  segment_statistics = np.concatenate((means,0.5*widths,0.5*heights),axis=1)
  #segment_statistics = means


  segment_statistics = segment_statistics/segment_statistics.max(axis=0)
  roi_statistics = segment_statistics[roi_labels].mean(axis=0)
  # unknown_statistics = segment_statistics[unknown_labels]

  parameters = np.array([1,1,1,1,1])

  x = np.sum(((segment_statistics-roi_statistics)**2),axis=1)
  sd = x.std(axis=0)
  mu = np.exp(1/2*(x/sd)**2)
  sim =mu*x
  sim[roi_labels]=0
  # print(x.shape)
  # print(sd.shape)
  # print(mu.shape)
  # print(mu)

  # sum_of_squared_differences = np.sum(squared_differences,axis=1)
  # sum_of_squared_differences= sum_of_squared_differences/sum_of_squared_differences.max(axis=0)



  ssd_image = np.take(x,segments_slic)





  mask = np.zeros((ssd_image.shape[0],ssd_image.shape[1],3),dtype='uint8')

  mask[ssd_image<threshold]=np.array([0,0,250])


  mask =  cv2.resize(mask,(width,height),interpolation= cv2.INTER_LINEAR)

  result = cv2.addWeighted(bgr,0.9,mask,0.5,0.3)
  print('in loop')
  cv2.imshow('result',result)



  if cv2.waitKey(25) & 0xFF == ord('q'):
    plt.imshow(ssd_image)
    plt.colorbar()
    plt.show()
    break
  # plt.imshow(result)


  # plt.show()
















