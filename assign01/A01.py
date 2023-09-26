import numpy as np
import cv2
import gradio as gr

def create_unnormalized_hist(image):
    hist = np.zeros(256, dtype="float32")
    for i in range(len(image)):
        pixel_value = image[i]
        hist[pixel_value] += 1
    return hist
   
def normalize_hist(hist):
    nhist = np.zeros(256, dtype="float32")
    total_pixels = np.sum(hist)
    for i in range(len(hist)):
        nhist[i] = hist[i] / total_pixels
    return nhist    

def create_cdf(nhist):
    cdf = np.zeros(256, dtype="float32")
    for i in range(len(nhist)):
        cdf[i] = cdf[i - 1] + nhist[i] 
    return cdf
    
def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):    
    hist = create_unnormalized_hist(image)
    nhist = normalize_hist(hist)
    cdf = create_cdf(nhist)
    if do_stretching:
        first_val = cdf[0]
        last_val = cdf[255]
        for i in range(len(cdf)):
            cdf[i]  -= first_val 
            cdf[i] /= last_val
    for i in range(len(cdf)):
        cdf[i] *= 255
    int_transform = cv2.convertScaleAbs(int_transform)[:,0]
    return int_transform
    
def do_histogram_equalize(image, do_stretching):
    output = np.copy(image)
    int_transform = get_hist_equalize_transform(image, do_stretching)
    for i in range(len(image)):
        val = image[i]
        new_val = val * int_transform
        output[i] = new_val
    return output
    
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