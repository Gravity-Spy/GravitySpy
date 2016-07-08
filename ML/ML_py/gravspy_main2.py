#gravspy_main2 script by Luke Calian, 6/29/16
#before running, run run_main in matlab and save each batch as a .mat file
#save conf_matrices, PP_matrices, true_labels, and retired_images as .mat files

#import modules
import numpy as np
from scipy.io import loadmat
from pdb import set_trace

#import data that does not change between batches
true_labels = loadmat('true_labels.mat')
retired_images = loadmat('retired_images.mat')
conf_matrices = loadmat('conf_matrices.mat')
PP_matrices = loadmat('PP_matrices.mat')

#decider function to determine where an image is placed
def decider(pp_matrix, ML_dec, t, R_lim, no_annotators):

<<<<<<< HEAD
  pp_matrix2 = np.append(pp_matrix, ML_dec.reshape((15,1))) #concatenate transpose of ML_decision to pp_matrix
  v = np.sum(pp_matrix2, axis=1)/np.sum(pp_matrix) #create vector of normalized sums of pp_matrix2
  maximum = np.amax(v) #initialize maximum, max value of v
  maxIdx = np.argmax(v) #initialize maxIdx, index of max value of v
=======
    pp_matrix2 = np.append(pp_matrix, ML_dec.reshape((15,1))) #concatenate transpose of ML_decision to pp_matrix
    set_trace()
    v = np.sum(pp_matrix2, axis=1)/np.sum(pp_matrix) #create vector of normalized sums of pp_matrix2
    maximum = np.amax(v) #initialize maximum, max value of v
    maxIdx = np.argmax(v) #initialize maxIdx, index of max value of v
>>>>>>> 0c0907958eb46cc2614a66cc16ce3d64bda62679

    if maximum >= t[maxIdx]: #if maximum is above threshold for given class, retire image

      decision = 1
      print('Image is retired')

    elif no_annotators >= R_lim: #if more than R_lim annotators have looked at image and no decision reached, pass to more experience users

      decision = 2
      print('Image is given to the upper class')

    else: #if fewer than R_lim annotators have looked at image, keep image

      decision = 3
      print('More labels are needed for the image')

    image_class = maxIdx

    return decision, image_class

#main function to evaluate batch of images
def main_trainingandtest(batch_name):

    batch = loadmat(batch_name) #read batch file

    #calculate prior probability of each image
    no_labels = np.histogram((true_labels['true_labels'][0]),np.unique(true_labels['true_labels'][0]))

    R_lim = 23 #initialize R, max # of citizens who can look at an image before it is passed to a higher level if consensus is not reached
    N = len(batch['images']) #initialize N, # of images in batch

    #initialize C, # of classes
    for i in range(N):
        if batch['images'][i]['type'][0][0] == 'T':
            C = len(batch['images'][i]['ML_posterior'][0][0])

    priors = np.ones((1,C))
    t = .4*np.ones((C,1)) #initialize t, threshold vector of .4 for each class

    dec_matrix = np.zeros((1,N)) #define dec_matrix, matrix of each image's decision
    class_matrix = np.zeros((1,N)) #define class_matrix, matrix of each decision's class

    pp_matrices_rack = [] #create list of pp_matrices for all images #np.zeros((15,23,N)) create 3D matrix of all posterior matrices

    #main for loop to iterate over images
    for i in range(N):

        if batch['images'][i]['type'][0][0] == 'G': #check if golden set image
            labels = batch['images'][i]['labels'][0][0] #take citizen labels of image
            userIDs = batch['images'][i]['IDs'][0][0] #take IDs of citizens who label image
            tlabel = batch['images'][i]['truelabel'][0][0][0] #take true label of image

            for ii in range(len(userIDs)): #iterate over user IDs of image

                indicator = 0

                for cc in range(len(conf_matrices['conf_matrices'][0])): #iterate over confusion matrices

                    if userIDs[ii] == conf_matrices['conf_matrices'][cc]['userID'][0][0][0]: #if user is registered

                        conf_matrix = conf_matrices['conf_matrices'][cc]['conf_matrix'][0] #take confusion matrix of citizen
                        conf_matrix[tlabel-1,labels[ii]-1] += 1 #update confusion matrix
                        conf_matrices['conf_matrices'][cc]['conf_matrix'][0] = conf_matrix #confusion matrix put back in stack
                        indicator = 1

                if indicator == 0: #if user not registered

                    dummy_matrix = np.zeros((C,C)) #create dummy matrix
                    dummy_matrix[tlabel-1,labels[ii]-1] += 1 #update dummy matrix
                    #conf_matrices['conf_matrices'] = np.append(conf_matrices['conf_matrices'][0], dummy_matrix) #append to confusion matrices
                    #conf_matrices(end + 1).userID = IDs(ii)

            dec_matrix[0,i] = 0 #since it is a training image, no decision is made
            class_matrix[0,i] = tlabel #class of image is its true label
            print('The image is from the training set')

        else: #if image not in golden set, i.e. has ML label but no true label

            indicator1 = 0

            for kk in range(len(retired_images['retired_images'][0])): #loop over retired images

                if batch['images'][i]['imageID'][0][0][0] == retired_images['retired_images'][0][kk]['imageID'][0][0]: #if image is retired

                    indicator1 = 1
                    dec_matrix[0,i] = -1 #give invalid decision
                    break

            if indicator1 == 0: #if image is not retired

                labels = batch['images'][i]['labels'][0][0] #take citizen labels of image
                userIDs = batch['images'][i]['IDs'][0][0] #take IDs of citizens who label image
                no_annotators = len(labels) #define number of citizens who annotate image
                ML_dec = batch['images'][i]['ML_posterior'][0][0] #take ML posteriors of image
                imageID = batch['images'][i]['imageID'][0][0][0] #take ID of image
                image_prior = priors #set priors for image to original priors

                for y in range(len(PP_matrices['PP_matrices'][0])): #iterate over posterior matrices

                    if imageID == PP_matrices['PP_matrices'][0][y]['imageID'][0][0]: #find posterior matrix for the image

                        image_prior = np.sum(PP_matrices['PP_matrices'][0][y]['matrix'],axis=1)/np.sum(PP_matrices['PP_matrices'][0][y]['matrix']) #if image labeled but not retired, PP_matrix information is used in the place of priors
                        break

                for j in range(C): #iterate over classes

                    for k in range(no_annotators): #iterate over citizens that labeled image
                        for iN in range(len(conf_matrices['conf_matrices'])): #iterate over confusion matrices

                            if userIDs[k] == conf_matrices['conf_matrices'][iN]['userID'][0][0][0]: #find confusion matrix corresponding to citizen

                                conf = conf_matrices['conf_matrices'][iN]['conf_matrix'][0] #take confusion matrix of citizen
                                break
                        conf_divided,x,z,s = np.linalg.lstsq(np.diag(sum(conf,2)),conf) #calculate p(l|j) value
                        pp_matrix = np.zeros((C,no_annotators)) #create posterior matrix
                        pp_matrix[j,k] = (conf_divided[j,labels[k]-1]*priors[0][j])/sum(conf_divided[:,labels[k]-1]*priors[0]) #calculate posteriors
                pp_matrices_rack.append(pp_matrix) #assign values to pp_matrices_rack


                dec_matrix[0,i], class_matrix[0,i] = decider(pp_matrix, ML_dec, t, R_lim, no_annotators) #make decisions for each image in batch

"""for jj in range(0, len(data['conf_matrices']): #for all citizens
	conf_update = data['conf_matrices'][IDs[jj-1]-1][0] #confusion matrices taken one by one
	conf_update_divided = np.linalg.sovle(np.diag(np.sum(conf_update, axis=0))), conf_update)
	alpha[:,jj] = np.diag(conf_update_divided)
	###TEST CODE###"""

"""#update the confusion matrices for test batch and promotion
  for i in range(0,N):
    if dec_matrix[0,i] == 1: #if image is retired
      labels = batch['images'][i]['labels'][0][0] #the citizen label of the image is taken
      IDs = batch['images'][i]['IDs'][0][0] #the IDs of the citizens that labeld that image are taken
      for ii in (0, len(IDs)): #for each citizen
        conf_matrix = batch['conf_matrices'][IDs[ii-1]-1][0] #take confusion matrix of each citizen
        conf_matrix[class_[i]-1,labels[ii]-1] = conf_matrix[class_[i]-1,labels[ii]-1]+1 #update confusion matrix
        batch['conf_matrices'][IDs[ii]-1][0] = conf_matrix"""

"""for jj in range(0, len(batch['conf_matrices']): #for all citizens
	conf_update = batch['conf_matrices'][IDs[jj-1]-1][0] #confusion matrices taken one by one
	#### FINISH ####"""

"""counter1 = len(data['images'][i]['PP_matrices'][0][0] + 1
counter2 = len() + 1
for i in range(0,N):
	if de"""

#for loop to iterate over each batch
for i in range(1,2):
  batch_name = 'batch' + str(i) + '.mat' #batch1.mat, batch2.mat, etc
  main_trainingandtest(batch_name) #call main_trainingandtest function to evaluate batch
  print('Batch done')

	#import pdb
  #pdb.set_trace()
