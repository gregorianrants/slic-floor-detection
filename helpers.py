import cv2
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from square import get_corners,get_slice
import time



# def normalise(image_3_channels):
#   (width,height,_) = image_3_channels.shape
#   reshaped = image_3_channels.reshape((width*height),3)
#   return image_3_channels/reshaped.max(axis=0)

# def mean(label,lab_small,segments_slic):
#   return lab_small[segments_slic==label].mean(axis=0)

# def std(label,lab_small,segments_slic):
#     return lab_small[segments_slic==label].std(axis=0)

# def get_roi(image,segments,top_left,bottom_right):
#   (top_left,bottom_right) = get_corners(image,top_left,bottom_right)
#   roi_slice = get_slice(segments,top_left,bottom_right).copy()
#   height,width = segments.shape

#   labels = np.unique(roi_slice)
#   return labels

# def get_contour_for_label(label,segments):
#   mask = np.zeros((segments.shape))
#   mask[segments==label]=255
#   contours,_=cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#   return contours[0]

# def getContours(labels,segments):
#   return [get_contour_for_label(label,segments) for label in labels]

# def getWidth(contour):
#   x,y,w,h = cv2.boundingRect(contour)
#   return w

# def getHeight(contour):
#   x,y,w,h = cv2.boundingRect(contour)
#   return w

# def getWidths(contours):
#   result =  np.array([getWidth(contour) for contour in contours])
#   #result = result/result.max()
#   return np.expand_dims(result, axis=1)

# def getHeights(contours):
#   result =  np.array([getHeight(contour) for contour in contours])
#   #result = result/result.max()
#   return np.expand_dims(result, axis=1)


def normalise(image_3_channels):
  (width,height,_) = image_3_channels.shape
  reshaped = image_3_channels.reshape((width*height),3)
  return image_3_channels/(width*height)

def mean(label,lab_small,segments_slic):
  return lab_small[segments_slic==label].mean(axis=0)

def median(label,lab_small,segments_slic):
  return np.median(lab_small[segments_slic==label],axis=0)

def std(label,lab_small,segments_slic):
    return lab_small[segments_slic==label].std(axis=0)

def get_roi(image,segments,top_left,bottom_right):
  (top_left,bottom_right) = get_corners(image,top_left,bottom_right)
  roi_slice = get_slice(segments,top_left,bottom_right).copy()
  height,width = segments.shape

  labels = np.unique(roi_slice)
  return labels

def get_contour_for_label(label,segments):
  mask = np.zeros((segments.shape))
  mask[segments==label]=255
  contours,_=cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  return contours[0]

def getContours(labels,segments):
  return [get_contour_for_label(label,segments) for label in labels]

def getWidth(contour):
  x,y,w,h = cv2.boundingRect(contour)
  return w

def getHeight(contour):
  x,y,w,h = cv2.boundingRect(contour)
  return w

def getWidths(contours):
  result =  np.array([getWidth(contour) for contour in contours])
  #result = result/result.max()
  return np.expand_dims(result, axis=1)

def getHeights(contours):
  result =  np.array([getHeight(contour) for contour in contours])
  #result = result/result.max()
  return np.expand_dims(result, axis=1)

