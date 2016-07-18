#Script by CIERA Intern Group, 7/12/16
#Updated by Scott Coughlin July 12, 2016

# ---- Import standard modules to the python path.
import numpy as np
from scipy.io import loadmat
import random
from pdb import set_trace
import pandas as pd
# Import the random data generation function get_data.py also in this folder.
import gen_data

###############################################################################
##########################                     ################################
##########################     decider         ################################
##########################                     ################################
###############################################################################

#decider function to determine where an image is placed
def decider(pp_matrix, ML_dec, t, R_lim, num_annotators):

    pp_matrix2 = np.hstack((pp_matrix, ML_dec.reshape((15,1)))) #concatenate transpose of ML_decision to pp_matrix
    v = np.sum(pp_matrix2, axis=1)/np.sum(np.sum(pp_matrix2)) #create vector of normalized sums of pp_matrix2
    maximum = np.amax(v) #initialize maximum, max value of v
    maxIdx = np.argmax(v) #initialize maxIdx, index of max value of v

    if maximum >= t[maxIdx]: #if maximum is above threshold for given class, retire image

        decision = 1
        print('Image is retired')

    elif num_annotators >= R_lim: #if more than R_lim annotators have looked at image and no decision reached, pass to more experience users

        decision = 2
        print('Image is given to the upper class')

    else: #if fewer than R_lim annotators have looked at image, keep image

        decision = 3
        print('More labels are needed for the image')

    image_class = maxIdx

    return decision, image_class

###############################################################################
##########################                         ############################
##########################  main_trainingandtest   ############################
##########################                         ############################
###############################################################################

#main function to evaluate batch of images
def main_trainingandtest(images,conf_matrices,PP_matrices,retired_images):

    R_lim = 23 #initialize R, max # of citizens who can look at an image before it is passed to a higher level if consensus is not reached
    N = images['type'].size #initialize N, # of images in batch

    #initialize C, # of classes
    for i in range(N):
        if images['type'][i] == 'T':
            C = images['ML_posterior'][i].size
            break

    # Flat priors. We have no sense of what category a given image should be
    # ahead of time.
    priors = np.ones((1,C))/C
    t = .4*np.ones((C,1)) #initialize t, threshold vector of .4 for each class

    dec_matrix = np.zeros((1,N)) #define dec_matrix, matrix of each image's decision
    class_matrix = np.zeros((1,N)) #define class_matrix, matrix of each decision's class

    pp_matrices_rack = [] #create list of pp_matrices for all images #np.zeros((15,23,N)) create 3D matrix of all posterior matrices

    #main for loop to iterate over images
    for i in range(N):

        if images['type'][i] == 'G': #check if golden set image
            labels  = images['labels'][i] #take citizen labels of image
            userIDs = images['userIDs'][i] #take IDs of citizens who label image
            tlabel  = images['truelabel'][i] #take true label of image

            for ii in range(userIDs.size): #iterate over user IDs of image

                indicator = 0

                for cc in range(len(conf_matrices)): #iterate over confusion matrices

                    if userIDs[ii] == conf_matrices['userID'][cc]: #if user is registered

                        #take confusion matrix of citizen and index by true label, label given by user and update the confusion matrix at that entry.
                        conf_matrices['conf_matrix'][cc][tlabel,labels[ii]] += 1
                        indicator = 1

                if indicator == 0: #if user not registered

                    dummy_matrix = np.zeros((C,C)) #create dummy matrix
                    dummy_matrix[tlabel,labels[ii]] += 1 #update dummy matrix
                    tmp = pd.DataFrame({ 'userID' : [userIDs[ii]],'conf_matrix' : [dummy_matrix]},index = [len(conf_matrices)])
                    conf_matrices = conf_matrices.append(tmp)

            dec_matrix[0,i] = 0 #since it is a training image, no decision is made
            class_matrix[0,i] = tlabel #class of image is its true label
            pp_matrices_rack.append([0])
            print('The image is from the training set')

        else: #if image not in golden set, i.e. has ML label but no true label

            indicator1 = 0

            for kk in range(len(retired_images)): #loop over retired images

                if images['imageID'][i] == retired_images['imageID'][kk]: #if image is retired
                    indicator1 = 1
                    dec_matrix[0,i] = -1 #give invalid decision
                    pp_matrices_rack.append([0])
                    break

            if indicator1 == 0: #if image is not retired

                labels           = images['labels'][i] #take citizen labels of image
                userIDs          = images['userIDs'][i] #take IDs of citizens who label image
                num_annotators   = labels.size #define number of citizens who annotate image
                ML_dec           = images['ML_posterior'][i] #take ML posteriors of image
                imageID          = images['imageID'][i] #take ID of image
                image_prior      = priors #set priors for image to original priors

                for y in range(len(PP_matrices)): #iterate over posterior matrices

                    if imageID == PP_matrices['imageID'][y]: #find posterior matrix for the image
                        image_prior = np.sum(PP_matrices['pp_matrix'][y],axis=1)/np.sum(PP_matrices['pp_matrix'][y]) #if image labeled but not retired, PP_matrix information is used in the place of priors
                        break

                pp_matrix = np.zeros((C,num_annotators)) #create posterior matrix
                #print(labels)
                for k in range(num_annotators): #iterate over citizens that labeled image
                    for iN in range(len(conf_matrices)): #iterate over confusion matrices

                        if userIDs[k] == conf_matrices['userID'][iN]: #find confusion matrix corresponding to citizen

                            conf = conf_matrices['conf_matrix'][iN] #take confusion matrix of citizen
                            break

                    conf_divided,x,z,s = np.linalg.lstsq(np.diag(np.sum(conf,axis=1)),conf) #calculate p(l|j) value

                    for j in range(C): #iterate over classes

                        pp_matrix[j,k] = (conf_divided[j,labels[k]]*priors[0][j])/sum(conf_divided[:,labels[k]]*priors[0]) #calculate posteriors
                pp_matrices_rack.append(pp_matrix) #assign values to pp_matrices_rack


                dec_matrix[0,i], class_matrix[0,i] = decider(pp_matrix, ML_dec, t, R_lim, num_annotators) #make decisions for each image in batch

"""At this point, the decisions for each image in the batch are given. For
golden images in the set, the decision is 0. For the ML labelled images, the
decisions are one of 1,2, or 3.

The posterior probability matrices are kept for all the ML labelled images. If
the decision is 2 or 3, the probabilities in this matrix will be used in a
further step. Not currently implemented.

Also, the confusion matrices are updated based on the golden images.

Next step is updating the confusion matrices for the test images and
citizen evaluation/promotion."""
   
    for i in range(N): #iterate over images to update confusion matrices
        
        if dec_matrix[0,i] == 1: #if image is retired
            
            labels = images['labels'][i] #the citizen label of the image is taken
            userIDs = images['userIDs'][i] #the IDs of the citizens that labeld that image are taken
            
            for ii in range(userIDs.size): #iterate over user IDs of image

                indicator2 = 0

                for cc in range(len(conf_matrices)): #iterate over confusion matrices

                    if userIDs[ii] == conf_matrices['userID'][cc]: #if user is registered

                        #take confusion matrix of citizen and index by true label, label given by user and update the confusion matrix at that entry.
                        conf_matrices['conf_matrix'][cc][tlabel,labels[ii]] += 1
                        indicator2 = 1

                if indicator2 == 0: #if user not registered

                    dummy_matrix = np.zeros((C,C)) #create dummy matrix
                    dummy_matrix[tlabel,labels[ii]] += 1 #update dummy matrix
                    tmp = pd.DataFrame({ 'userID' : [userIDs[ii]],'conf_matrix' : [dummy_matrix]},index = [len(conf_matrices)])
                    conf_matrices = conf_matrices.append(tmp)

    """    for jj = 1:len(conf_matrices)  # for all the citizens

        conf_update = conf_matrices['conf_matrix'][jj]; # their conf. matrices are taken one by one

        conf_update_divided,x,z,s = np.linalg.lstsq(np.diag(sum(conf_update,2)),conf_update) #calculate p(l|j) value

        alpha[:,jj] = np.diag(conf_update_divided);    # alpha parameters are recalculated
    """
    #Thresholding alpha vectors and citizen evaluation (needs work)


    # Ordering the images and sending/saving them
    counter1 = len(retired_images)
    counter2 = len(PP_matrices)

    for i in range(N):

        if dec_matrix[0,i] == 1: # if it is decided to be retired
            tmp = pd.DataFrame({ 'imageID' : [images['imageID'][i]],'class' : [class_matrix[0,i]]},index = [counter1])
            retired_images = retired_images.append(tmp)

            counter1 = counter1 + 1

        elif dec_matrix[0,i] == 2 or dec_matrix[0,i] == 3:  #if the decision is forwarding to the upper class or wait for more labels

            dummy_decider = 1

            for y in range(len(PP_matrices)):        #in case the image was waiting for more labels beforehand

                if images['imageID'][i] == PP_matrices['imageID'][y]:
                    PP_matrices['pp_matrix'][y] = pp_matrices_rack[i]      #the PP matrix is overwritten.
                    dummy_decider = 0
                    break

            if dummy_decider:
                tmp = pd.DataFrame({ 'imageID' : [images['imageID'][i]],'pp_matrix' : [pp_matrices_rack[i]]},index = [counter2])
                PP_matrices = PP_matrices.append(tmp)

                counter2 = counter2 + 1
    return conf_matrices, PP_matrices, retired_images


###############################################################################
##########################                     ################################
##########################      MAIN CODE      ################################
##########################                     ################################
###############################################################################

if __name__ == '__main__':
    #import data that does not change between batches
    conf_matrices = loadmat('conf_matrices.mat')

    tmpCM  = []
    tmpCM1 = []
    for iN in range(conf_matrices['conf_matrices'].size):
        tmpCM.append(conf_matrices['conf_matrices'][iN]['userID'][0][0][0])
        tmpCM1.append(conf_matrices['conf_matrices'][iN]['conf_matrix'][0])

    conf_matrices  = pd.DataFrame({ 'userID' : tmpCM,'conf_matrix' : tmpCM1})
    retired_images = pd.DataFrame({ 'imageID' : [], 'class' : []})
    PP_matrices    = pd.DataFrame({ 'imageID' : [],'pp_matrix' : []}) 

    hold,conf_matrices = gen_data.gen_data()

    #for loop to iterate over each batch
    for i in range(1,11):
        batch_name = 'batch' + str(i) + '.mat' #batch1.mat, batch2.mat, etc
        batch = loadmat(batch_name) #read batch file
        tmpType         = []
        tmpLabels       = []
        tmpuserIDs      = []
        tmpTruelabel    = []
        tmpImageID      = []
        tmpML_posterior = []
        # Subtracting 1 off the index from the mat file for the "labels" so that the indexing works in python.
        for iN in range(batch['images'].size):
            tmpType.append(batch['images'][iN]['type'][0][0])
            tmpLabels.append(batch['images'][iN]['labels'][0][0]-1)
            tmpuserIDs.append(batch['images'][iN]['IDs'][0][0])
            tmpTruelabel.append(batch['images'][iN]['truelabel'][0][0][0]-1)
            tmpML_posterior.append(batch['images'][iN]['ML_posterior'][0][0])
            tmpImageID.append(batch['images'][iN]['imageID'][0][0][0])

        images = pd.DataFrame({'type' : tmpType,'labels' : tmpLabels,'userIDs' : tmpuserIDs, 'ML_posterior' : tmpML_posterior, 'truelabel' : tmpTruelabel, 'imageID' : tmpImageID})

        images,hold = gen_data.gen_data()

        conf_matrices, PP_matrices, retired_images = main_trainingandtest(images,conf_matrices,PP_matrices,retired_images) #call main_trainingandtest function to evaluate batch
        print('Batch done')

