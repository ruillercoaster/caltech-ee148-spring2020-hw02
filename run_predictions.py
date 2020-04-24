import os
import numpy as np
import json
from PIL import Image

def compute_convolution(I, T, top_half):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    ############## my code starts here ##################
    
    # get the shape of T
    T_rows, T_cols,T_channels = np.shape(T)
    # make T a vector
    Tvector = T.flatten()
    assert n_channels == T_channels
    
    heatmap = np.empty((n_rows, n_cols))
    if top_half == True:
        end_row = int(n_rows/2)-T_rows
    else:
        end_row = n_rows-T_rows
    
    for irow in range(end_row):
        for icol in range(n_cols-T_cols):
    
            # get the overlapping area of I and T, and make it a vector
    
            Ivector = I[irow:irow+T_rows, icol:icol+T_cols, :].flatten()

            heatmap[irow+int(T_rows/2),icol+int(T_cols/2)] = np.dot(Ivector,Tvector)
            #  heatmap[irow,icol] = np.corrcoef(Ivector,Tvector)[0,1]
    
    ############## my code ends here ##################


    '''
    END YOUR CODE
    '''

    return heatmap


'''
    BEGIN MY CODE
'''

def generate_templates(number_of_templates):

# load annotations
    gts_path = './hw02_annotations'       
    with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
            gts = json.load(f)

    widths = []
    heights = []
    annotations = []
    # for i in range(1):
    for i in range(len(file_names_train)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_train[i]))
        I = np.asarray(I,dtype=np.int64)
        # normalize I 

        ############# split the image into color channels
        redarr = I[:,:,0]
        greenarr = I[:,:,1]
        bluearr = I[:,:,2] 

        ##### normalization
        sumarr = (redarr+greenarr+bluearr)/3
        mask = np.where(sumarr==0)
        redN = redarr/sumarr
        greenN = greenarr/sumarr
        blueN = greenarr/sumarr
        redN[mask] = 0
        greenN[mask]=0
        blueN[mask] =0

        ##### standardization
        redmax =np.max(redN)
        redmin = np.min(redN)
        redS = (redN-redmin)/(redmax-redmin)
        redS = redS*255
        
        greenmax = np.max(greenN)
        greenmin = np.min(greenN)
        greenS = (greenN-greenmin)/(greenmax-greenmin)
        greenS = greenS*255

        bluemax = np.max(blueN)
        bluemin = np.min(blueN)
        blueS = (blueN-bluemin)/(bluemax-bluemin)
        blueS = blueS*255

        I = np.stack([redS, greenS, blueS],axis=-1)

        bounding_boxes = gts[file_names_train[i]]

        for j in range(len(bounding_boxes)):
            tl_col = int(np.trunc(bounding_boxes[j][1]))
            br_col = int(np.trunc(bounding_boxes[j][3]))
            br_row = int(np.trunc(bounding_boxes[j][2]))
            tl_row = int(np.trunc(bounding_boxes[j][0]))
            annotations.append(np.array(I[ tl_row:br_row,tl_col:br_col,:]))
            shape = I[ tl_row:br_row,tl_col:br_col,:].shape
           
            widths.append(br_col-tl_col)
            heights.append(br_row-tl_row)
    widths = np.array(widths)
    heights = np.array(heights)
    annotations = np.array(annotations)
    # find annotations with the same size
    combo = [[widths[k],heights[k]] for k in range(len(widths))]
    # find the unique sizes of annotations
    unique_combo,combo_counts = np.unique(combo,axis=0,return_counts=True)
    # select <number_of_templates> most frequent sizes of annotations
    nthres = np.min([number_of_templates, len(unique_combo)])
    thres = np.sort(combo_counts)[-nthres]
    selected_ind = np.where(combo_counts>=thres)[0]

    # calculate the template as the average of annotations with the same size.
    Ts = []
    for isize in selected_ind:
        annt_ind = np.where((np.array(combo) == unique_combo[isize]).all(axis=1))[0]
        Tselected = np.mean(annotations[annt_ind],axis=0)
        Ts.append(Tselected)
    Ts = np.array(Ts)

    # 





    return Ts
'''
    END MY CODE
'''

def predict_boxes(score_map,score_threshold,wrange,hrange):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    

    ind = np.where(score_map>=score_threshold)
    newind_y = ind[0]
    newind_x = ind[1]
    
    # Tw = [ti.shape[0] for ti in Ts]
    # Th = [ti.shape[1] for ti in Ts]
    # wrange = np.max(Tw)
    # hrange = np.min(Th)

    while(len(newind_x)>0):
        random_i = np.random.choice(len(newind_y),1)
        x0 = newind_x[random_i]
        y0 = newind_y[random_i]

        captured = np.where((newind_x>=x0-wrange) &(newind_x<=x0+wrange) &(newind_y>=y0-hrange) &(newind_y<=y0+hrange))[0]
#     captured = np.where((newind_x>=x0-wrange*2) &(newind_x<=x0+wrange*2) &(newind_y>=y0-hrange*2) &(newind_y<=y0+hrange*2))[0]
    
        newind_x_captured = newind_x[captured]
        newind_y_captured = newind_y[captured]

        tl_row = int(np.percentile(newind_y_captured,10))
        br_row = int(np.percentile(newind_y_captured,90))
        if tl_row == br_row:
            br_row = br_row + 1

        tl_col = int(np.percentile(newind_x_captured,10))
        br_col = int(np.percentile(newind_x_captured,90))
        if tl_col == br_col:
            br_col = br_col + 1

        # tl_row = int(np.min(newind_y_captured) - int(hrange))
        # br_row = int(np.max(newind_y_captured) + int(hrange))
        # tl_col = int(np.min(newind_x_captured) - int(wrange))
        # br_col = int(np.max(newind_x_captured) + int(wrange))

        # tl_row = int((np.max(newind_y_captured) + np.min(newind_y_captured))/2 - int(hrange/2))
        # br_row = int((np.max(newind_y_captured) + np.min(newind_y_captured))/2 + int(hrange/2))
        # tl_col = int((np.max(newind_x_captured) + np.min(newind_x_captured))/2 - int(wrange/2))
        # br_col = int((np.max(newind_x_captured) + np.min(newind_x_captured))/2 + int(wrange/2))
        # target_x = int(np.nanmean(newind_x_captured))
        # target_y = int(np.nanmean(newind_y_captured))
        # confidence=float(score_map[y0,x0])
        confidence = float(np.nanmax(score_map[newind_y_captured,newind_x_captured]))

        newind_x = np.delete(newind_x,captured)
        newind_y = np.delete(newind_y,captured)

        overlapped = np.where((newind_y>=tl_row-50) & (newind_y<=br_row+50) & (newind_x>=tl_col-50) & (newind_x<=br_col+50))[0]
        if len(overlapped)>0:
            newind_x = np.delete(newind_x,overlapped)
            newind_y = np.delete(newind_y,overlapped)

        output.append([tl_row,tl_col,br_row,br_col, confidence])
        # output.append([tl_row,tl_col,br_row,br_col, confidence])

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I, Ts, top_half,score_threshold,wrange,hrange):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    # normalize I 

    ############# split the image into color channels
    redarr = I[:,:,0]
    greenarr = I[:,:,1]
    bluearr = I[:,:,2] 

    ##### normalization
    # sumarr = (0.1*redarr+0.1*greenarr+0.8*bluearr)
    sumarr = (redarr+greenarr+bluearr)/3
    mask = np.where(sumarr==0)
    redN = redarr/sumarr
    greenN = greenarr/sumarr
    blueN = greenarr/sumarr
    redN[mask] = 0
    greenN[mask]=0
    blueN[mask] =0

    ##### standardization
    redmax =np.max(redN)
    redmin = np.min(redN)
    redS = (redN-redmin)/(redmax-redmin)
    redS = redS*255
    
    greenmax = np.max(greenN)
    greenmin = np.min(greenN)
    greenS = (greenN-greenmin)/(greenmax-greenmin)
    greenS = greenS*255

    bluemax = np.max(blueN)
    bluemin = np.min(blueN)
    blueS = (blueN-bluemin)/(bluemax-bluemin)
    blueS = blueS*255

    I = np.stack([redS, greenS, blueS],axis=-1)


    # You may use multiple stages and combine the results
    
    scores = np.ones((I.shape[0],I.shape[1])) # score matrix saves the scores of each pixel being a target
    scores[:] = 0


    for T in Ts:
        # heatmap = compute_convolution(I, T,top_half)
        heatmap = np.ones((I.shape[0],I.shape[1]))
        heatmap[:] = 0
        # calculate heatmap with each template
        heatmap_ = compute_convolution(I,T,top_half)
        # normalize heatmaps
        heatmap_ = (heatmap_ - np.min(heatmap_))/(np.max(heatmap_)-np.min(heatmap_)) 
        #giving each pixel a score determing whether it is a target
        if top_half==True:
            thres = np.percentile(heatmap_[:int(heatmap_.shape[0]/2),:],99.95)
        else:
            thres = np.percentile(heatmap_,99.95)
        ind = np.where(heatmap_>=thres)    
        scores[ind] = scores[ind]+1

        #predict the bounding boxes
        output = predict_boxes(scores,score_threshold,wrange,hrange)


    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        output[i][4] = output[i][4]/len(Ts)
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output





# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../RedLights2011_Medium'

# load splits: 
split_path = './hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))




# set a path for saving predictions:
preds_path = './hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

NUMBER_OF_TEMPLATES = 30
TOP_HALF=False
SCORE_THRESHOLD = 3
WRANGE=26
HRANGE=27


# load annotations and generate templates
Ts = generate_templates(NUMBER_OF_TEMPLATES)


# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
########################### for training######################
preds_train = {}

for i in range(len(file_names_train)):
    print('%d/%d'%(i,len(file_names_train)))

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I,dtype=np.int64)


    preds_train[file_names_train[i]] = detect_red_light_mf(I,Ts,TOP_HALF,SCORE_THRESHOLD,WRANGE,HRANGE)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)
##################################################################

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''

    preds_test = {}
    for i in range(len(file_names_test)):

        print('%d/%d'%(i,len(file_names_test)))
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I,dtype=np.int64)

        
        preds_test[file_names_test[i]] = detect_red_light_mf(I,Ts,TOP_HALF,SCORE_THRESHOLD,WRANGE,HRANGE)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
