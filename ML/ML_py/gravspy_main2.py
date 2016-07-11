#gravspy_main2 script by Luke Calian, 6/29/16
#before running, run run_main in matlab and save each batch as a .mat file
#save conf_matrices, PP_matrices, true_labels, and retired_images as .mat files

#import modules
import numpy as np
from scipy.io import loadmat
from pdb import set_trace
import pandas as pd

#import data that does not change between batches
retired_images = loadmat('retired_images.mat')
conf_matrices = loadmat('conf_matrices.mat')
PP_matrices = loadmat('PP_matrices.mat')

tmpPP  = []
tmpPP1 = []
for iN in range(PP_matrices['PP_matrices'][0].size):
    tmpPP.append(PP_matrices['PP_matrices'][0][iN]['imageID'][0][0])
    tmpPP1.append( PP_matrices['PP_matrices'][0][iN]['matrix'])


tmpCM  = []
tmpCM1 = []
for iN in range(conf_matrices['conf_matrices'].size):
    tmpCM.append(conf_matrices['conf_matrices'][iN]['userID'][0][0][0])
    tmpCM1.append(conf_matrices['conf_matrices'][iN]['conf_matrix'][0])


tmpRI  = []
for iN in range(retired_images['retired_images'].size):
    tmpRI.append(retired_images['retired_images'][0][iN]['imageID'][0][0])


conf_matrices  = pd.DataFrame({ 'userID' : tmpCM,'conf_matrix' : tmpCM1})
retired_images = pd.DataFrame({ 'imageID' : tmpRI})
PP_matrices    = pd.DataFrame({ 'imageID' : tmpPP,'pp_matrix' : tmpPP1}) 

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

#main function to evaluate batch of images
def main_trainingandtest(images,conf_matrices,PP_matrices):

    R_lim = 23 #initialize R, max # of citizens who can look at an image before it is passed to a higher level if consensus is not reached
    N = images['type'].size #initialize N, # of images in batch

    #initialize C, # of classes
    for i in range(N):
        if images['type'][i] == 'T':
            C = images['ML_posterior'][i].size
            break

    priors = np.ones((1,C))
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

                        conf_matrix = conf_matrices['conf_matrix'][cc] #take confusion matrix of citizen
                        conf_matrix[tlabel,labels[ii]] += 1 #update confusion matrix
                        conf_matrices['conf_matrix'][cc] = conf_matrix #confusion matrix put back in stack
                        indicator = 1

                if indicator == 0: #if user not registered

                    dummy_matrix = np.zeros((C,C)) #create dummy matrix
                    dummy_matrix[tlabel,labels[ii]] += 1 #update dummy matrix
                    tmp = pd.DataFrame({ 'userID' : [userIDs[ii]],'conf_matrix' : [dummy_matrix]},index = [len(conf_matrices)])
                    conf_matrices = conf_matrices.append(tmp)

            dec_matrix[0,i] = 0 #since it is a training image, no decision is made
            class_matrix[0,i] = tlabel #class of image is its true label
            print('The image is from the training set')

        else: #if image not in golden set, i.e. has ML label but no true label

            indicator1 = 0

            for kk in range(retired_images.size): #loop over retired images

                if images['imageID'][i] == retired_images['imageID'][kk]: #if image is retired
                    indicator1 = 1
                    dec_matrix[0,i] = -1 #give invalid decision
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

                for k in range(num_annotators): #iterate over citizens that labeled image
                    for iN in range(len(conf_matrices)): #iterate over confusion matrices

                        if userIDs[k] == conf_matrices['userID'][iN]: #find confusion matrix corresponding to citizen

                            conf = conf_matrices['conf_matrix'][iN] #take confusion matrix of citizen
                            break

                    conf_divided,x,z,s = np.linalg.lstsq(np.diag(sum(conf,2)),conf) #calculate p(l|j) value

                    for j in range(C): #iterate over classes

                        pp_matrix = np.zeros((C,num_annotators)) #create posterior matrix
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
citizen evaluation/promotion.


Updating the Confusion Matrices for Test Data and Promotion
    """
    for i in range(N):
        if dec_matrix[0,i] == 1: #if image is retired
            labels = images['labels'][i] #the citizen label of the image is taken
            userIDs = images['userIDs'][i] #the IDs of the citizens that labeld that image are taken
            for ii in range(userIDs.size): #iterate over user IDs of image

                indicator2 = 0

                for cc in range(len(conf_matrices)): #iterate over confusion matrices

                    if userIDs[ii] == conf_matrices['userID'][cc]: #if user is registered

                        conf_matrix = conf_matrices['conf_matrix'][cc] #take confusion matrix of citizen
                        conf_matrix[tlabel,labels[ii]] += 1 #update confusion matrix
                        conf_matrices['conf_matrix'][cc] = conf_matrix #confusion matrix put back in stack
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

"""
counter1 = length(retired_images) + 1;
counter2 = length(PP_matrices) + 1;

for i = 1:N %for each image
    
    if decision(i) == 1     %if it is decided to be retired
        
        retired_images(counter1).imageID = images(i).imageID;         %it is put into the retired images array with the ID and the class it is classified into.
        retired_images(counter1).class = class(i);
        
        for y = 1:length(PP_matrices)  %in case the retired image was waiting for more labels beforehand
            
            if images(i).imageID == PP_matrices(y).imageID        
                
                PP_matrices(y) = [];       %the PP matrix is taken out of the saved matrices.
                break
            end
        end
        
        counter1 = counter1 + 1;
    
    elseif decision(i) == 2 || decision(i) == 3      %if the decision is forwarding to the upper class or wait for more labels
        
        dummy_decider = 1;
        
        for y = 1:length(PP_matrices)        %in case the image was waiting for more labels beforehand
            
            if images(i).imageID == PP_matrices(y).imageID
                PP_matrices(y).imageID = images(i).imageID;      %the PP matrix is overwritten.
                dummy_decider = 0;
                break
            end
        end
        
        if dummy_decider
        
            PP_matrices(counter2).imageID = images(i).imageID;           %The PP matrix of the image is saved with the corresponding ID to be used in the place of the prior in the next batch
            PP_matrices(counter2).matrix = pp_matrices_rack{i};
            counter2 = counter2 + 1;
        end
        
        
    end
end"""

#for loop to iterate over each batch
for i in range(1,2):
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

    main_trainingandtest(images,conf_matrices,PP_matrices) #call main_trainingandtest function to evaluate batch
    print('Batch done')
