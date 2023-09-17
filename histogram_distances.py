from scipy.spatial import distance
import cv2
import numpy as np

def get_pixel_count(labels,segments_slic):
  return np.count_nonzero(np.isin(segments_slic,labels))

def get_hist(label,segments_slic,img):
  mask = np.zeros(segments_slic.shape,dtype='uint8')
  mask[segments_slic==label]=255
  number_of_pixels = np.count_nonzero(mask)
  hist_1 = cv2.calcHist([img],[0],mask,[15],[0,256])
  hist_2 = cv2.calcHist([img],[1],mask,[15],[0,256])
  hist_3 = cv2.calcHist([img],[2],mask,[15],[0,256])
  hist = np.concatenate((hist_1,hist_2,hist_3))
  #hist = np.concatenate((hist_a,hist_b))
  hist = np.squeeze(hist)
  return hist/hist.sum()

def apply_to_pixels(labels,segments_slic,img,f):
  [f(label,segments_slic,img) for label in labels]





def get_roi_hist(roi_labels,segments_slic,img):
  hist = np.array([get_hist(label,segments_slic,img) for label in roi_labels])
  hist = hist.sum(axis=0)
  hist = np.squeeze(hist)
  return hist/len(roi_labels)

def create_hists(labels,segments_slic,img,create_hists):
  hists = np.array([create_hists(label,segments_slic,img) for label in labels])
  return hists



def combine_hists(labels,hists):
  return hists[labels].sum()



def get_distances(roi_hist,segments_hists):
  distances = [distance.cosine(hist,roi_hist) for hist in segments_hists]
  return distances


  

