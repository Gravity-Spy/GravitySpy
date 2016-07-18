function [priors] = calc_priors(tr_labels)

    
    no_labels = hist(tr_labels,unique(tr_labels));

    
    priors = no_labels / length(tr_labels);
