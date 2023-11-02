import cv2
import skimage
import numpy as np

def find_WBC(image):
    #Step 1 get superpixel groups
    segments = skimage.segmentation.slic(image, n_segments=40, sigma=8, start_label=0)
    cnt = len(np.unique(segments))
    
    #Step 2 compute mean per superpixel
    group_means = np.zeros((cnt, 3), dtype="float32")
    for specific_group in range(cnt):
            mask_image = np.where(segments == specific_group, 255, 0).astype("uint8")
            mask_image = np.expand_dims(mask_image, axis=-1)
            group_means[specific_group] = cv2.mean(image, mask=mask_image)[0:3]
            
    #Step 3 Use k-means on group means colors to group them into 4 color groups
    _, bestLabels, centers = cv2.kmeans(data=group_means, K=4, bestLabels=None, 
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0),
                                                flags=cv2.KMEANS_RANDOM_CENTERS,
                                                attempts=10)
    
    #Step 4 Find the K-means groups closest to blue(255, 0, 0)
    blue = np.array([255, 0, 0])
    index = np.zeros(3)
    distances = np.zeros(1)
    distances = np.delete(distances, 0)
    for center in centers:
        distance = np.linalg.norm(blue-center)
        distances = np.append(distances, distance)

    minVal = np.min(distances)
    index = np.where(distances == minVal)
    distances = np.sort(distances, None)
    
    #Step 5 Set that k-means group to white and the rest to black
    for i in range(len(centers)):
        if (i == index[0][0]):
            centers[i] = [255, 255, 255]
        else:
            centers[i] = [0, 0, 0]

    #Step 6 Determine the new colors for each superpixel group
    centers = cv2.convertScaleAbs(centers)
    colors_per_clump = centers[bestLabels.flatten()]
    
    #Step 7 Recolor the superpixels with their new group colors (which are now just white and black)
    cell_mask = colors_per_clump[segments]
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
    
    #Step 8 Use cv2.connectedComponents to get disjoint blobs from cell_mask
    retval, labels = cv2.connectedComponents(cell_mask, None, 8, cv2.CV_32S)

    #Step 9 For each blob group (except 0, which is the background)
    SCALER = 20
    bounding_boxes = []
    for i in range(1, retval):
        coords = np.where(labels==i)
        if (len(coords[0]) != 0):
            ymin, xmin= np.min(coords, axis = 1)
            ymax, xmax = np.max(coords, axis = 1)
            bounding_box = [ymin - SCALER, xmin - SCALER, ymax + SCALER, xmax + SCALER]
            bounding_boxes.append(bounding_box)    
    
    return bounding_boxes