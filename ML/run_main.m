clear
close all
clc


%% This part creates ten batches of image data.

for maincounter1 = 1:10
    
    name = ['batch' num2str(maincounter1)]
    
    run generate_toy_data_trainingandtest
    
    save(name, 'images')
    
    clearvars -EXCEPT conf_matrices true_labels
end

PP_matrices = [];
retired_images = [];

save('conf_matrices', 'conf_matrices')

save('PP_matrices', 'PP_matrices')

save('true_labels', 'true_labels')

save('retired_images', 'retired_images')      %Initially, PP_matrices and retired_images are empty. They will be filled as the batches are processed.

%%

clear
clc

%%

for maincounter2 = 1:10

    batch_name = ['batch' num2str(maincounter2)]    %For each batch, the main algorithm is run.
    
    run main_trainingandtest
    
    clear
    
    disp('batch done')
    
end
