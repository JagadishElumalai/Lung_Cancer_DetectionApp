import os
import shutil
import cv2
import numpy as np
import SimpleITK as stk
import tensorflow as tf
import pandas as pd
from skimage import measure
from sklearn.cluster import KMeans
from tensorflow.keras import backend as K
import copy
import matplotlib.pyplot as plt



def load_mhd(file):
    mhdimage = stk.ReadImage(file)
    ct_scan = stk.GetArrayFromImage(mhdimage)
    origin = np.array(list(mhdimage.GetOrigin()))
    space = np.array(list(mhdimage.GetSpacing()))
    return ct_scan, origin, space

def process_ct_scan(filepath):
    # Load your models here
    model = tf.keras.models.load_model("AppDetector\\models\\LC2_cGAN.h5", custom_objects={'dice_coef':dice_coef, 'dice_coef_loss':dice_coef_loss})
    fpr_model = tf.keras.models.load_model("AppDetector\\models\\FPR_classifier_model.h5")

    ct, origin, space = load_mhd(filepath)
    ct_norm = cv2.normalize(ct, None, 0, 255, cv2.NORM_MINMAX)  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  
    ct_norm_improved = []

    for layer in ct_norm:
        ct_norm_improved.append(clahe.apply(layer.astype(np.uint8)))  
    plt.imshow(ct_norm_improved[ct.shape[0]//2],cmap="grey")
    plt.savefig("AppDetector/static/assets/normalized_image.png",bbox_inches='tight')
    plt.close()

    centeral_area = ct_norm_improved[len(ct_norm_improved)//2][100:400, 100:400]
    kmeans = KMeans(n_clusters=2).fit(np.reshape(centeral_area, [np.prod(centeral_area.shape), 1]))
    centroids = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centroids)

    lung_masks= genarate_lung_masks(ct_norm_improved,threshold)

    extracted_lungs = []
    for lung, mask in zip(ct_norm_improved,lung_masks):
        extracted_lungs.append(cv2.bitwise_and(lung, lung, mask=mask))

    plt.imshow(extracted_lungs[ct.shape[0]//2],cmap="grey")
    print("Length of Extracted lungs:",len(extracted_lungs))

    X = np.array(extracted_lungs)
    X = (X-127.0)/127.0
    X = X.astype(np.float32)

    X = np.reshape(X, (len(X), 512, 512, 1))
    print(X.shape)

    predictions = model.predict(X)
    print("predict shape",predictions.shape)

    predictions[predictions>=0.5] = 255
    predictions[predictions<0.5] = 0
    predictions = predictions.astype(np.uint8)
    pred = list(predictions)
    pred = [np.squeeze(i) for i in pred]
    print("Length of Preds",len(pred))
    bboxes = []
    centroids = []
    diams = []

    for mask in pred:
        mask = cv2.dilate(mask, kernel=np.ones((5,5)))
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        bb = []
        cc = []
        dd = []
        for prop in regions:
            B = prop.bbox
            C = prop.centroid
            D = prop.equivalent_diameter_area
            bb.append((( max(0, B[1]-8), max(0, B[0]-8) ),( min(B[3]+8, 512), min(B[2]+8, 512) )))   
            cc.append(C)    # (y,x)
            dd.append(D)
        bboxes.append(bb)
        centroids.append(cc)
        diams.append(dd)
    
    video_name = 'AppDetector/static/assets/extracted_lungs_video.webm'
    frame_height, frame_width = extracted_lungs[0].shape[:2]  
    fps = 10  
    fourcc = cv2.VideoWriter_fourcc(*'vp80') 
    out = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))
    mimgs = copy.deepcopy(extracted_lungs)
    for i, (img, boxes) in enumerate(zip(mimgs, bboxes)):
        for rect in boxes:
            img = cv2.rectangle(img, rect[0], rect[1], (255), 2)
        lung_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        out.write(lung_image)

    out.release()
    print(f"Video saved as {video_name}")


    originals = copy.deepcopy(ct_norm_improved)
    final_boxes = []
    for i,(img,bbox) in enumerate(zip(originals, bboxes)):
        img_boxes = []
        for box in bbox:
            x1 = box[0][0]
            y1 = box[0][1]
            x2 = box[1][0]
            y2 = box[1][1]
            if abs(x1-x2) <=50 or abs(y1-y2)<=50:
                x = (x1+x2)//2
                y = (y1+y2)//2
                x1 = max(x-25, 0)
                x2 = min(x+25, 512)
                y1 = max(y-25, 0)
                y2 = min(y+25, 512)
                imgbox = img[y1:y2,x1:x2]
                img_boxes.append(imgbox)
            else:
                imgbox = img[y1:y2,x1:x2]
                img_boxes.append(imgbox)
        final_boxes.append(img_boxes)
    
    fpr_preds = []
    for i in final_boxes:
        each_p = []
        for img in i:
            if img.shape != (50,50):
                img = np.resize(img, (50,50))
            img = img/255.
            img = np.reshape(img, (1,50,50,1))
            pred = fpr_model.predict(img)
            pred = int(pred>=0.5)
            each_p.append(pred)
        fpr_preds.append(each_p)
    
    for i in range(len(diams)):
        if len(diams[i]):
            for j in range(len(diams[i])):
                diams[i][j] = diams[i][j]*space[0] 

    final_img_bbox = []
    cancer = []
    df = pd.DataFrame(columns = ['Layer', 'Position (x,y)', 'Diameter (mm)', 'BBox [(x1,y1),(x2,y2)]'])
    e_lungs = copy.deepcopy(ct_norm_improved)
    for i,(img,bbox,preds,cents,dms) in enumerate(zip(e_lungs, bboxes, fpr_preds, centroids, diams)):
        token = False
        for box,pred,cent,dm in zip(bbox,preds,cents,dms):
            if pred:
                x1 = box[0][0]
                y1 = box[0][1]
                x2 = box[1][0]
                y2 = box[1][1]
                img = cv2.rectangle(img, (x1,y1), (x2,y2), (255), 2)
                dct = pd.DataFrame({'Layer':i, 'Position (x,y)':[f"{cent[::-1]}"], 'Diameter (mm)':dm, 'BBox [(x1,y1),(x2,y2)]':[f"{list(box)}"]})
                df = pd.concat([df,dct], ignore_index = True)
                token = True
        final_img_bbox.append(img)
        cancer.append(token)
    
    video_name = 'AppDetector/static/assets/final_img_bbox_video.webm'
    frame_height, frame_width = extracted_lungs[0].shape[:2] 
    fps = 5  
    fourcc = cv2.VideoWriter_fourcc(*'vp80') 
    out = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))
    for lung_image in final_img_bbox:
        if len(lung_image.shape) == 2:  
            lung_image = cv2.cvtColor(lung_image, cv2.COLOR_GRAY2BGR)
        out.write(lung_image)

    out.release()
    print(f"Video saved as {video_name}")


    cancer_pred=[f"True%:{(cancer.count(True)/len(cancer))*100}",f"False%:{(cancer.count(False)/len(cancer))*100}"]

    print(cancer_pred)
    results = {
        'normalized_img': 'normalized_image.png',  
        'mask_video': 'extracted_lungs_video.mp4',
        'final_img_bbox_video': 'final_img_bbox_video.mp4',  
        'cancer_probabilities': cancer_pred  
    }
    
    return results

# Evaluation metrics
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def genarate_lung_masks(ct_norm_improved,threshold):
    lung_masks = []
    for layer in ct_norm_improved:
        ret, lung_roi = cv2.threshold(layer, threshold, 255, cv2.THRESH_BINARY_INV)
        lung_roi = cv2.erode(lung_roi, kernel=np.ones([4,4]))
        lung_roi = cv2.dilate(lung_roi, kernel=np.ones([13,13]))
        lung_roi = cv2.erode(lung_roi, kernel=np.ones([8,8]))

        labels = measure.label(lung_roi)        
        regions = measure.regionprops(labels)   
        good_labels = []
        for prop in regions:        
            B = prop.bbox          
            if B[2]-B[0] < 475 and B[3]-B[1] < 475 and B[0] > 40 and B[2] < 472:
                good_labels.append(prop.label)
        lung_roi_mask = np.zeros_like(labels)
        for N in good_labels:
            lung_roi_mask = lung_roi_mask + np.where(labels == N, 1, 0)

        contours, hirearchy = cv2.findContours(lung_roi_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        external_contours = np.zeros(lung_roi_mask.shape)
        for i in range(len(contours)):
            if hirearchy[0][i][3] == -1: 
                area = cv2.contourArea(contours[i])
                if area>518.0:
                    cv2.drawContours(external_contours,contours,i,(1,1,1),-1)
        external_contours = cv2.dilate(external_contours, kernel=np.ones([4,4]))

        external_contours = cv2.bitwise_not(external_contours.astype(np.uint8))
        external_contours = cv2.erode(external_contours, kernel=np.ones((7,7)))
        external_contours = cv2.bitwise_not(external_contours)
        external_contours = cv2.dilate(external_contours, kernel=np.ones((12,12)))
        external_contours = cv2.erode(external_contours, kernel=np.ones((12,12)))

        external_contours = external_contours.astype(np.uint8)     
        lung_masks.append(external_contours)
    print(len(lung_masks))
    return lung_masks