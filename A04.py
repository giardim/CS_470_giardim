import numpy as np
import cv2 

def getOneLBPLabel(subimage, label_type):
    center = subimage[1][1]
    values = np.array([
        subimage[0][1],
        subimage[0][2],
        subimage[1][2],
        subimage[2][2],
        subimage[2][1],
        subimage[2][0],
        subimage[1][0],
        subimage[0][0]
    ])
    labelArray = np.zeros(8)
    label = 0
    jumps = 0

    for i in range(len(values)):
        if (values[i] > center):
            labelArray[i] = 1
        else:
            labelArray[i] = 0

    for i in range(len(labelArray) - 1):
        if (labelArray[i] != labelArray[i + 1]):
            jumps += 1
            if jumps > 2:
                return 9

    for i in range(len(values)):
        if (labelArray[i] == 1):
            label += 1
    
    return label

def getLBPImage(image, label_type):
    output = np.copy(image)
    subImage = np.zeros(shape=(4,4))
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            subImage = image[i:i+3, j:j+3]
            if (subImage.shape[0] < 3 or subImage.shape[1] < 3):
                continue
            output[i][j] = getOneLBPLabel(subImage, label_type="UNIFORM")
    return output


