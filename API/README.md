Readme detailing the functions utilzied in the production mode of the Gravity Spy website. 

callAPI.py - Takes classification data from Zooniverse API and packages it in two data sets which are saved as mysql tables.
	
	* Classifications Data Set
	* 'choiceINT','choice', 'userID','workflow','classificationID','zooID','classification_number','session','timeToClass'

        * Images Data Set
        * 'choice', choiceINT', 'classificationID', 'classification_number', 'userID','workflow', 'imageID', 'subject_set', 'ML_posterior', 'ML_label', 'ML_confidence', 'type', 'true_label', 'pp_matrix', 'decision' 

CC.py - Models image retirement and user promotion 

CCPostProc.py - takes output from CC.py and acts on decision (i.e. retiring an image into a new subject set, moving an image that has been labeled but not retired, and promoting a user to the next workflow.)

DataProducts.py - Produces visualization of these different data sets such as user confusion matrices, the pp_matrix for every image, and will also display things like how many users are in every workflow, average agreement between ML and people, and how long people take to classify in every workflow.
