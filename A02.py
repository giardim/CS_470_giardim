import gradio as gr
import cv2 
import numpy as np

def read_kernel_file(filepath):
    index = 2
    file = open(filepath, 'r')
    format = file.readline()
    file.close()
    format = format.split(" ")
    rowCnt = int(format[0])
    colCnt = int(format[1])
    kernel = np.zeros(shape=(rowCnt, colCnt))
    for i in range (0, rowCnt):
        for j in range(0, colCnt):
            kernel[i][j] = format[index]
            index += 1
    return kernel
    
    
def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    output = np.copy(image).astype("float64")
    image = image.astype("float64")
    kernel = kernel.astype("float64")
    np.flip(kernel, -1)
    PW = kernel.shape[0] // 2
    PH = kernel.shape[1] // 2
    paddedImage = cv2.copyMakeBorder(src=image, top=PH, bottom=PH, left=PW, right=PW, borderType=cv2.BORDER_CONSTANT, value=0)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            row = i + kernel.shape[0]
            col = j + kernel.shape[1]
            subImage = paddedImage[row][col]
            filterVals = subImage * kernel
            value = np.sum(filterVals)
            output[row, col] = value
    if (convert_uint8):
        output = cv2.convertScaleAbs(output)
    return output
            
    

def filtering_callback(input_img, filter_file, alpha_val, beta_val):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kernel = read_kernel_file(filter_file.name)
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val) 
    return output_img

def main():
    demo = gr.Interface(fn=filtering_callback, 
    inputs=["image", 
    "file", 
    gr.Number(value=0.125), 
    gr.Number(value=127)],
    outputs=["image"])
    demo.launch() 

# Later, at the bottom
if __name__ == "__main__": 
    main()
