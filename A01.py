###################################
#Author: 
# Michael Giardina
#Class:
#  CS470
#Lanuage Python 3.10
###################################

import numpy as np
import cv2
import gradio as gr
import math as m

#Create a unnormalized histogram (a histogram with the count of each pixel)
def create_unnormalized_hist(image):
    hist = np.zeros(256, dtype="float32")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i][j]
            hist[pixel_value] += 1
    return hist
   
#Create a normalized histogram (a histogram with the probabilities of each pixel)
def normalize_hist(hist):
    nhist = np.zeros(256, dtype="float32")
    total_pixels = np.sum(hist)
    for i in range(hist.shape[0]):
        nhist[i] = hist[i] / total_pixels
    return nhist    

#Create a CDF (a histogram where the next value is the probability of the previous value 
#       + the probability of the current value)
def create_cdf(nhist):
    cdf = np.zeros(256, dtype="float32")
    cdf[0] = nhist[0]
    for i in range(1, nhist.shape[0]):
        cdf[i] = cdf[i - 1] + nhist[i] 
    return cdf
    
#Get the histogram equalize transform (A function where you multiply the cdf
#       by the highest POSSIBLE value)
def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):    
    int_transform = np.zeros(256, dtype="float32")
    hist = create_unnormalized_hist(image)
    nhist = normalize_hist(hist)
    cdf = create_cdf(nhist)
    if do_stretching:
        #if the first value dominates the graph, the rest of the histogram will be 
        #   squeezed into a small output range, squeezing fixes this issue
        cdf = np.subtract(cdf, cdf[0])
        cdf = np.divide(cdf, max(cdf))
    for i in range(cdf.shape[0]):
        int_transform[i] = cdf[i] * 255.0
    int_transform = cv2.convertScaleAbs(int_transform)[:,0]
    return int_transform

#Apply the transform to the image    
def do_histogram_equalize(image, do_stretching):
    output = np.copy(image)
    int_transform = get_hist_equalize_transform(image, do_stretching)
    for i in range (image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i][j]
            new_value = int_transform[pixel_value]
            output[i][j] = new_value
    return output
    
    
#Everything below is mandatory, provided by the professor
def intensity_callback(input_img, do_stretching):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching)
    return output_img

def main():
    demo = gr.Interface(fn=intensity_callback,
    inputs=["image", "checkbox"],
    outputs=["image"])
    demo.launch()

if __name__ == "__main__":
    main()