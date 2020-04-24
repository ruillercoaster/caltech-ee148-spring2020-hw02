import os
import json
import numpy as np
import matplotlib.pyplot as plt
def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    '''
    BEGIN YOUR CODE
    '''
    # print(box_1)
    tl_row_1,tl_col_1,br_row_1,br_col_1 = box_1
    tl_row_2,tl_col_2,br_row_2,br_col_2 = box_2

    assert tl_row_1 < br_row_1
    assert tl_col_1 < br_col_1
    assert tl_row_2 < br_row_2
    assert tl_col_2 < br_col_2


    x_left = np.max([tl_col_1,tl_col_2])
    x_right = np.min([br_col_1,br_col_2])
    y_top = np.max([tl_row_1,tl_row_2])
    y_bottom = np.min([br_row_1, br_row_2])

    if (x_left > x_right) or (y_top>y_bottom):
        iou = 0.0
    else:
        intersection_area = (x_right-x_left)*(y_bottom-y_top)

        box_area1 = (br_row_1 - tl_row_1) * (br_col_1 - tl_col_1)
        box_area2 = (br_row_2 - tl_row_2) * (br_col_2 - tl_col_2)
        
        iou = intersection_area/(box_area1 + box_area2 - intersection_area) 
        print(iou)
    '''
    END YOUR CODE
    '''
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''

    
    for pred_file, pred in preds.items():
        n = 0
        conf_list = np.ones(len(pred)); conf_list[:] = 0.0
        
        for j in range(len(pred)):
            conf_list[j] = pred[j][4]
        M_ind = np.where(conf_list>=conf_thr)[0]; M = len(M_ind)
    
        gt = gts[pred_file]; N = len(gt)
        
        if (N>0) and (M>0):
            for i in range(len(gt)):
                iou_list = np.ones(M); iou_list[:] = 0.
                for j in range(len(M_ind)):

                    iou = compute_iou(pred[M_ind[j]][:4], gt[i])
                    if iou>IOU_THR:
                        iou_list[j] = iou
                iou_max = np.max(iou_list)
                if iou_max > iou_thr:
                    n = n+1
        TP = TP + n
        FP = FP + (M - n)
        FN = FN + (N - n)

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = './hw02_preds'
gts_path = './hw02_annotations'

# load splits:
split_path = './hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 
confidence_thrs = []
for fname in preds_train:
    for bs in preds_train[fname]:
        confidence_thrs.append(bs[4])
confidence_thrs = np.unique(np.array(confidence_thrs))# using (ascending) list of confidence scores as thresholds

plt.figure()
color = {0.25:'green',0.5:'red',0.75:'blue'}
for IOU_THR in [0.25, 0.5, 0.75]:
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    Precision = np.zeros(len(confidence_thrs))
    Recall  = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=IOU_THR, conf_thr=conf_thr)
    
        if tp_train[i]+fp_train[i] == 0:
            Precision[i] = 1
        else:
            Precision[i] = tp_train[i]/(tp_train[i]+fp_train[i])

        if tp_train[i]+fn_train[i] ==0:
            Recall[i] = 1
        else:
            Recall[i] = tp_train[i]/(tp_train[i]+fn_train[i])

    plt.plot(Recall,Precision,ls = '-', label='train iou_thrs = %.2f'%IOU_THR, color= color[IOU_THR])
plt.xlabel('Recall')
plt.ylabel('Precision')
# plt.legend()
# plt.title('PR curve for training')
# plt.savefig('./PR_curve_train.jpg')



# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')
    confidence_thrs = []
    for fname in preds_test:
        for bs in preds_test[fname]:
            confidence_thrs.append(bs[4])
    confidence_thrs = np.unique(np.array(confidence_thrs))# using (ascending) list of confidence scores as thresholdsf

    # plt.figure()
    for IOU_THR in [0.25, 0.5, 0.75]:
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        Precision = np.zeros(len(confidence_thrs))
        Recall  = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=IOU_THR, conf_thr=conf_thr)
    
            if tp_test[i]+fp_test[i] == 0:
                Precision[i] = 1
            else:
                Precision[i] = tp_test[i]/(tp_test[i]+fp_test[i])

            if tp_test[i]+fn_test[i] ==0:
                Recall[i] = 1
            else:
                Recall[i] = tp_test[i]/(tp_test[i]+fn_test[i])
        
        plt.plot(Recall,Precision,ls ='dotted', label='test iou_thrs = %.2f'%IOU_THR, color= color[IOU_THR])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # plt.title('PR curve for testing')
    # plt.savefig('./PR_curve_test.jpg')
    plt.title('PR curve')
    plt.savefig(preds_path+'/PR_curve.jpg')

    plt.figure()
