import cv2
import skimage
import numpy as np

def find_wbc(image):
    #Step 1 get superpixel groups
    segmented_image = skimage.segmentation.slic(image, n_segments=100, sigma=5, start_label=0)
    cnt = len(np.unique(segmented_image))
    
    #Step 2 compute mean per superpixel
    group_means = np.zeros((cnt, 3), dtype="float32")
    for specific_group in range(cnt):
            mask_image = np.where(segmented_image == specific_group, 255, 0).astype("uint8")
            mask_image = np.expand_dims()
            group_means[specific_group] = cv2.means(image, mask=mask_image)[0:3]
    
    #Step 3 Use k-means on group means colors to group them into 4 color groups
    _, bestLabels, centers = cv2.kmeans(data=group_means, K=5, bestLabels=None, 
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0),
                                                flags=cv2.KMEANS_RANDOM_CENTERS,
                                                attempts=10)
    
    #Step 4 Find the K-means groups closest to blue(255, 0, 0)
    blue = [255, 0, 0]
    background = [0, 0, 0]
    wbc = np.where(bestLabels == blue, bestLabels, background)
    
    #Step 5 Set that k-means group to white and the rest to black
    for center in centers:
        if (center == wbc):
            center = (255, 255, 255)
        else:
            center = (0, 0, 0)
            
    #Step 6 Determine the new colors for each superpixel group
    centers = cv2.convertScaleAbs("uint8")
    colors_per_clump = centers[bestLabels.flatten()]
    
    #Step 7 Recolor the superpixels with their new group colors (which are now just white and black)
    cell_mask = colors_per_clump[segmented_image]
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_RGB2GRAY)
    
    #Step 8 Use cv2.connectedComponents to get disjoint blobs from cell_mask
    retval, labels = cv2.connectedComponents(image=cell_mask, connectivity=8, ltype=cv2.CV_16U)
    
    #Step 9 For each blob group (except 0, which is the background)
    for i in range(len(labels)):
        coords = np.where(labels == i, labels, -1)
        if (coords != -1):
            ymin, xmin, ymax, xmax = coords[0], coords[1], coords[2], coords[3]
            bounding_boxes = [ymin, xmin, ymax, xmax]
    
    return bounding_boxes
    
