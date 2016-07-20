%% Initializations and Training Data

% The Zooniverse server (specifically Nero https://github.com/zooniverse/nero)
% will continuously send data containing the following classification
% information:
%


%   The ID of the user, the ID of the image they classified, and 
%   the classification made by that user for that image. Potentially,
%   information concerning whether this image was a golden set image, that
%   is an image where the label was known a head of time, versus a ML
%   classified images, that is an image where the label is only the
%   confidence level of the ML classifier, will also be sent in the message
%   from Nero.
%
% This information will then be parsed and saved in the following format.
% A structure array with rows relating to a given image and columns 
% containing the following information about that image:
%
%       The Type - A label (string) either 'T' or 'G' to determine if it is a ML
%       classified label or a pre-labelled "golden" image.
%
%       The Labels - An array (double) of a 1XN row vector where N is the number
%       of labels this image has been given at a certain time. Each column
%       is a different answer that is associated with a different user.
%       This takes us to the next column...
%
%       The User IDs - An array (double) of a  1XN row vector where N is the number
%       of labels this image has been given. Each column
%       is the userID associated with the answer given in The Labels
%       column.
%
%       ML Posterior - An array (double) of a  1XC row vector where C is
%       the number of pre-determined morphologies that the classifier has
%       been trained on. Each column is the ML confidence that the image
%       belongs in one of the C classes.
%
%       True Label - (int) For images labelled 'T' this values is set to -1
%       but for images labelled 'G' This value indicates the "true" class
%       that this image belongs in for the purposes of comparing a citizens
%       classification with this true label.

% Once data for X amount of images has been parsed and stored in the above
% format we will run the following script to update the retirability of an
% image as well as the skill level of the citizens.

% In addition to the above information, we will also store information on
% the "confusion matrix" of a user. This information (stored in a .dat
% file) will be a NX1 array where N is the number of users. in each row we
% have a cell array that contains the CXC "confusion matrix" for that user.
% A perfectly skilled user would only have values on the diagonal of this
% matrix and all off diagonal values indicate wrong answers were given to
% one category or another when presented with a 'G' true labelled image.

% Initialize varaible t. t is a CX1 column vector where C is
% the number of pre-determined morphologies and where each row is the
% predetermined certainty threshold that an image most surpass to be
% considered part of class C. Here all classes have the same threshold but
% in realty different categories will have more difficult or more relax
% thresholds for determination of class and therefore retirability.



% Initialize T_alpha, T_alpha is a CX1 column vector where C is
% the number of pre-determined morphologies and where each row is the
% predetermined "skill" level threshold for each class of glitch that a user
% must surprass in order to move on to the next user level (Levels are
% B1-B4, Apprentice, and Master).  Here all classes have the same threshold but
% in realty different categories will have more difficult or more relax
% thresholds based on how challenging a given class is.

% Initialize R, the citizen limit: The citizen limit refers to
% the max amount of citizens who can look at an image in a given level
% before it is based on to the higher levels for more analysis. The idea is
% that if an image's retirability cannot be determined from 30 labels then
% this image needs to be looked at by more skilled users or LIGO experts.

R_lim = 23;

%one batch of data is read
load(batch_name);

N = size(images, 1);       %N is the # of images in the batch

for i = 1:N
    if images(i).type == 'T'
        C = length(images(i).ML_posterior);      %C is the no of classes
        break
    end
end

t = 0.4*ones(C,1);

alpha_threshold = 0.4*ones(C,1);

g_c = 0.7*ones(C,1);

load('true_labels.mat')

load('retired_images.mat')

%Read confusion matrices: Each confusion matrix belongs to a userID
load('conf_matrices.mat')

%Read PP matrices: Each PP matrix belongs to an imageID that is not retired
%yet.
load('PP_matrices.mat')

%Calculate the Priors
priors = calc_priors(true_labels)'; %The prior probability of each image is calculated.


%% The main loop that goes through the batch of images one by one

for i = 1:N  %for each image
  
    if images(i).type == 'G'       %if it is a golden set image (has true label)
        
        labels = images(i).labels;     %the citizen labels of that image are taken
        
        IDs = images(i).IDs;          %The IDs of the citizens that labeled that image are taken
        
        tlabel = images(i).truelabel;   %the true label is taken
        
        for ii = 1:length(IDs)  %for each citizen who labeled the image
        
            indicator = 0;
            
            for cc = 1:length(conf_matrices)
                
                if IDs(ii) == conf_matrices(cc).userID   %if they have registered before
                    
                    conf_matrix = conf_matrices(cc).conf_matrix; %their conf matrix is taken
                    conf_matrix(tlabel, labels(ii)) = conf_matrix(tlabel, labels(ii)) + 1;  %Conf matrix updated
                    conf_matrices(cc).conf_matrix = conf_matrix;  %Conf matrix put back into the stack
                    
                    indicator = 1;
                end
            end
  
            if indicator == 0   %if they haven't registered before
                
                dummy_matrix = zeros(C,C);            
                dummy_matrix(tlabel, labels(ii)) = dummy_matrix(tlabel, labels(ii)) + 1;
                conf_matrices(end + 1).conf_matrix = dummy_matrix;
                conf_matrices(end + 1).userID = IDs(ii);        %A new confusion matrix is created at the end of the array
            end
        end
        
        decision(i) = 0;  %Since it is a training image, there is no decision that needs to be made what class the image belongs to.
        class(i) = tlabel;   %The class the image belongs to is its true label. This won't be used anywhere.
        
        disp('image is from the training set')
        
    else  %if the image does not have a true label but only a ML label
        
        indicator1 = 0;
        
        for kk = 1:length(retired_images)         %In case the image is already retired before
            
            if images(i).imageID == retired_images(kk).imageID
                indicator1 = 1;
                decision(i) = -1;                 %Do nothing and give an invalid decision
                break
            end
        end
        
        if indicator1 == 0    %If the image has not been retired before
        
            labels = images(i).labels;     %the labels of that image are taken
        
            no_annotators = length(labels);   %number of citizens that labeled that image is calculated
        
            IDs = images(i).IDs;               %The IDs of the citizens that labeled that image are taken
        
            ML_dec = images(i).ML_posterior;     %The ML posteriors for that image are taken.
        
            imageID = images(i).imageID;          %The imageID is taken
        
            image_prior = priors;                  %Priors for that image are set to the original priors, in case the test image is a new test image. (Intra-batch algorithm)
        
            for y = 1:length(PP_matrices)
            
                if imageID == PP_matrices(y).imageID
                
                    image_prior = sum(PP_matrices(y).matrix,2)/sum(sum(PP_matrices(y).matrix));   %If the image has labeled before but not retired, the PP_matrix information is used in the place of priors (Inter-batch algorithm)
                    break
                end
            end
        
            for j = 1:C       %for each class
                for k = 1:no_annotators   % loop over number of annotators for this image
                    for iN = 1:length(conf_matrices) % loop over confusion matrix structure array to find the specific users matrix
                        if IDs(k) == conf_matrices(iN).userID
                            break
                        end
                    end
            
                    conf = conf_matrices(iN).conf_matrix;      %the conference matrix of the citizen is taken
            
                    conf_divided = diag(sum(conf,2))\conf;     %The p(l|j) value is calculated
            
                    pp_matrix(j,k) = (conf_divided(j,labels(k))*image_prior(j))/sum(conf_divided(:,labels(k)).*image_prior);   %Posterior is calculated
            
                end
            end
    
            pp_matrices_rack{i} = pp_matrix;
    
            [decision(i), class(i)] = decider(pp_matrix, ML_dec, t, R_lim, no_annotators);     %A decision for the image is given. 1 is retire, 2 is upper class, 3 is next batch
        end
    end
     
end    
       
%At this point, the decisions for each image in the batch are given. For
%golden images in the set, the decision is 0. For the ML labelled images, the
%decisions are one of 1,2, or 3.

%The posterior probability matrices are kept for all the ML labelled images. If
%the decision is 2 or 3, the probabilities in this matrix will be used in a
%further step. Not currently implemented.

%Also, the confusion matrices are updated based on the golden images.

%Next step is updating the confusion matrices for the test images and
%citizen evaluation/promotion.


%% Updating the Confusion Matrices for Test Data and Promotion

for i = 1:N %for each image
    
    if decision(i) == 1 %if it is retired
        
        labels = images(i).labels;  %The citizen labels of the image are taken
        
        IDs = images(i).IDs;            %The IDs of the citizens that labeled that image are taken
        
        for ii = 1:length(IDs)  %for each citizen
        
            indicator2 = 0;
            
            for cc = 1:length(conf_matrices)
                
                if IDs(ii) == conf_matrices(cc).userID      %if they have registered before
                    
                    conf_matrix = conf_matrices(cc).conf_matrix; %their conf matrix is taken
                    conf_matrix(tlabel, labels(ii)) = conf_matrix(class(i), labels(ii)) + 1;  %Conf matrix updated
                    conf_matrices(cc).conf_matrix = conf_matrix;  %Conf matrix put back into the stack
                    
                    indicator2 = 1;
                end
            end
  
            if indicator2 == 0        %if they haven't registered before
                
                dummy_matrix = zeros(C,C);
                dummy_matrix(tlabel, labels(ii)) = dummy_matrix(class(i), labels(ii)) + 1;
                conf_matrices(end + 1).conf_matrix = dummy_matrix;
                conf_matrices(end + 1).userID = IDs(ii); %A new confusion matrix is created at the end of the array
            end
        end
    end
end


for jj = 1:length(conf_matrices)  %for all the citizens
    
    conf_update = conf_matrices(jj).conf_matrix;   %their conf. matrices are taken one by one
    
    conf_update_divided = diag(sum(conf_update,2))\conf_update;  
    
    alpha(:,jj) = diag(conf_update_divided);    %alpha parameters are recalculated
    
end


%Thresholding alpha vectors and citizen evaluation (needs work)

for jj = 1:length(conf_matrices)
    
    if alpha(:,jj) > alpha_threshold
        
        citizen_decision(jj).decision = 'P';
        citizen_decision(jj).userID = conf_matrices(jj).userID;
    else
        
        citizen_decision(jj).decision = 'R';
        citizen_decision(jj).userID = conf_matrices(jj).userID;
    end
end
    
%% Ordering the images and sending/saving them

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
        
        if max(images(i).ML_posterior) > g_c(class(i))
            retired_images(counter1).ret_cat = 'G';
        else
            retired_images(counter1).ret_cat = '?';
        end
        
        
        counter1 = counter1 + 1;
    
    elseif decision(i) == 2 || decision(i) == 3      %if the decision is forwarding to the upper class or wait for more labels
        
        dummy_decider = 1;
        
        for y = 1:length(PP_matrices)        %in case the image was waiting for more labels beforehand
            
            if images(i).imageID == PP_matrices(y).imageID
                PP_matrices(y).matrix = pp_matrices_rack{i};      %the PP matrix is overwritten.
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
end

%% Saving the Confusion Matrices and the PP Matrices

save('conf_matrices', 'conf_matrices')

save('PP_matrices', 'PP_matrices')
        
save('true_labels', 'true_labels')

save('retired_images', 'retired_images')    

save('citizen_decision', 'citizen_decision')
