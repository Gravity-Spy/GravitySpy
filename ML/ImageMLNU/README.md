This README explain how to run the ML Image classifer for GravitySpy

required packages include

http://scikit-image.org/docs/dev/install.html

http://deeplearning.net/software/theano/install.html#install

http://keras.io/#installation

Scenario one

Having the trained model and want to apply it on the unlabeled glitches

Step1: Making the pickles files corresponding to unlabeled data (test samples):


python make_pickle.py --dataset-path ./data/unlabeled_glitches/ –-save-address ./pickeles --test-flag 1 


--dataset-path: defines the input directory of unlabeled data. Each test samples needs 4 durations. 

–-save-address : target directory where the pickle files should be there 

--test-flag: a flag that says these are unlabeled data (if it is zero, it shows them as golden set) 

Step2: Loading the trained model and apply it on the test samples saved in the pickles files made in previous step: 

python labelling_test_glitches.py --pickle-address ./pickeles/ --model-address ./trained_model/ --save-address ./labels/
