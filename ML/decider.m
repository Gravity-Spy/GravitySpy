function [decision, class] = decider(pp_matrix, ML_decision, t, R, no_annotators)

pp_matrix2 = [pp_matrix ML_decision'];

v = sum(pp_matrix2, 2)/sum(sum(pp_matrix2));

[maximum loc] = max(v);

if maximum >= t(loc);
    
    decision = 1;
    disp('image is retired')
    
elseif no_annotators >= R
    
    decision = 2;
    disp('image is given to the upper class')
    
else
    
    decision = 3;
    
    disp('more labels are needed for the image')
    
end

class = loc;

    