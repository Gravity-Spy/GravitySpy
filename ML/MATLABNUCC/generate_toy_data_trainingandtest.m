% This Program simulates one batch of data

%% Initialization

%100 images
N = 100;

%30 citizens
R = 30;

%15 Classes
C = 15;

%% Simulating Training Labels and True Labels for Test Data (We won't have the latter in real data!!!)

true_labels =  randi(C,1,N);

training_labels =  randi(C,1,N/5);

%% Simulating ML decisions

for i = 1:N
        big_prob = 0.7 + 0.2*rand(1);
        rest_prob = 1 - big_prob;
    for j = 1:C
        if j == true_labels(i)
            ML_decision(j,i) = big_prob;   
        else 
            ML_decision(j,i) = rest_prob/(C-1);
        end
    end
end

ML_decision = ML_decision';

%% Simulating Data

%Simulating Confusion Matrices

for k = 1:R
    matrix = zeros(C);
    for ii = 1:C
        for jj = 1:C
           if ii == jj
               matrix(ii,jj) = 180+randi(40,1,1);
            else
               matrix(ii,jj) = randi(5,1,1);
            end
        end
    end
   conf_matrices_cell{k} = matrix;
end


for k = 1:R
    
    conf_matrices_dummy(k).conf_matrix = conf_matrices_cell{k};
    conf_matrices_dummy(k).userID = k;
end

conf_matrices = conf_matrices_dummy';

%Simulating Citizen Labels and Corresponding IDs (we assume all images have
%the same amount of labels

for i = 1:N
    labels = [];
    total = round(20 + 10*rand(1));
    correct = round(6 + 14*rand(1));
    rest = round(1 + 14*rand(1,total-correct));
    labels(1:correct) = true_labels(i);
    labels = [labels rest];
    citizen_labels{i} = labels;
    IDs = randperm(30,total);
    userIDs{i} = IDs; 
end

for i = 1:N/5
    
    citizen_training_labels = [];
    citizen_training_correct = round(6 + 14*rand(1));
    citizen_training_rest = round(1 + 14*rand(1,R-citizen_training_correct));
    citizen_training_labels(1:citizen_training_correct) = training_labels(i);
    citizen_training_labels = [citizen_training_labels citizen_training_rest];
    all_citizen_training_labels{i} = citizen_training_labels;
    citizen_training_IDs = randperm(30);
    citizen_training_userIDs{i} = citizen_training_IDs;
    
end

citizen_labels = citizen_labels';
userIDs = userIDs';
all_citizen_training_labels = all_citizen_training_labels';
citizen_training_userIDs = citizen_training_userIDs';
%% Putting the labels and IDs in corresponding image and creating the training images

for i = 1:N    %Putting the labels to the images
        images(i).type = 'T';
        images(i).labels = citizen_labels{i};
        images(i).IDs = userIDs{i};
        images(i).ML_posterior = ML_decision(i,:);
        images(i).truelabel = -1;
end

for i = N+1:N + N/5   %Creating the training images
        images(i).type = 'G';
        images(i).truelabel = training_labels(i-N);
        images(i).labels = all_citizen_training_labels{i-N};
        images(i).IDs = citizen_training_userIDs{i-N};
end

images = images(randperm(length(images)));  %shuffling test and training images

images = images';

dummy = randperm(250, N + N/5);       %generating image IDs

for i = 1:N+N/5
    images(i).imageID = dummy(i);     %putting the IDs to the images
end

