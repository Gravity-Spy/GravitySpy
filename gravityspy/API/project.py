#!/usr/bin/env python

# ---- Import standard modules to the python path.

from panoptes_client import *
from scipy.sparse import coo_matrix
from gwpy.table import EventTable

import re, operator, pickle, copy, os
import pandas as pd
import numpy as np

__all__ = ['ZooProject', 'flatten', 'GravitySpyProject']

# This function generically flatten a dict
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        try:
            items.extend(flatten(v, new_key, sep=sep).items())
        except:
            items.append((new_key, v))
    return dict(items)


class ZooProject(object):
    '''
    `ZooProject` inherits from the Project class
    from the panoptes client and adds some
    wrapper functions to extracts helpful
    project information.
    '''
    def __init__(self, zoo_project_id, cache_zoo_project=False,
                 load_zoo_project_from_cache=False):

        if load_zoo_project_from_cache:
            print('Trying to load project from cache...')
            inputFile = open('{0}.pkl'.format(zoo_project_id), 'rb')
            # Pickle dictionary using protocol 0.
            project = pickle.load(inputFile)
            print('successful')
            self.project_info =  project.project_info
            self.workflow_info = project.workflow_info
            return

        self.zoo_project_id = zoo_project_id
        tmp = Project.find(zoo_project_id)
        self.project_info = flatten(tmp.raw)

        # Determine workflow order
        self.workflow_info = {}
        order = self.project_info['configuration_workflow_order']
        workflows = [int(str(iWorkflow)) for iWorkflow in order]

        # Save workflow information
        for iWorkflow in workflows:
            tmp1 = Workflow.find(iWorkflow)
            self.workflow_info[str(iWorkflow)] = flatten(tmp1.raw)

        # For faster creation of the ZooProject object
        # it is recommended to save Object to cache
        # and to use the load_zoo_project_from_cache option
        # for future use.
        if cache_zoo_project:
            print('Saving project to cache...')
            output = open('{0}.pkl'.format(zoo_project_id), 'wb')
            # Pickle dictionary using protocol 0.
            pickle.dump(self, output)
            

    def get_golden_subject_sets(self):
        """Parameters
        ----------

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        workflowGoldenSetDict = {}

        for iWorkflow in self.workflow_info.keys():
            try:
                workflowGoldenSetDict[iWorkflow] = \
                    self.workflow_info[iWorkflow]['configuration_gold_standard_sets']
            except:
                # Workflow has no assosciated golden set
                workflowGoldenSetDict[iWorkflow] = []

        return workflowGoldenSetDict


    def get_golden_images(self):
        """Parameters
        ----------

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        workflowGoldenSetImagesDict = {}
        workflowGoldenSetDict = self.get_golden_subject_sets()

        for iWorkflow in workflowGoldenSetDict.keys():
            goldenImages = {}

            for iGoldenSubjectSet in workflowGoldenSetDict[iWorkflow]:
                tmp = SubjectSet.find(iGoldenSubjectSet)


                while True:
                    try:
                        nextSubject = tmpSubjects.next()
                        goldenImages[str(nextSubject.id)] = \
                            [str(nextSubject.raw['metadata']['subject_id']),\
                             str(nextSubject.raw['metadata']['#Label']),
                             int(test.raw['id'])]
                    except:
                        break

            workflowGoldenSetImagesDict[iWorkflow] = goldenImages

        return workflowGoldenSetImagesDict


    def get_answers(self, workflow=None):
        """Parameters
        ----------
        workflow : `int`, optional, default None

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        # now determine infrastructure of workflows so we know what workflow
        # this image belongs in
        workflowDictAnswers = {}

        if workflow:
            workflows = [str(workflow)]
        else:
            workflows = self.workflow_info.keys()

        # Determine possible answers to the workflows
        for iWorkflow in workflows:
            answerDict = {}

            for iAnswer in self.workflow_info[iWorkflow]\
                                             ['tasks_T1_choicesOrder']:
                answerDict[iAnswer] = []
            workflowDictAnswers[iWorkflow] = answerDict

        return workflowDictAnswers


    def get_subject_sets_per_workflow(self, workflow=None):
        """Parameters
        ----------
        workflow : `int`, optional, default None

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        workflowDictSubjectSets = {}
        workflowGoldenSetDict = self.get_golden_subject_sets()

        if workflow:
            workflows = [str(workflow)]
        else:
            workflows = self.workflow_info.keys()

        for iWorkflow in workflows:
            # check if golden set exists for workflow
            goldenset = workflowGoldenSetDict[iWorkflow]
            # Determine subject sets associated with this workflow
            subject_sets_for_workflow = self.workflow_info[iWorkflow]\
                                        ['links_subject_sets']
            subjectset_id = [int(str(iSubject)) \
                            for iSubject in subject_sets_for_workflow]
            subjectset_id = [iSubject for iSubject in subjectset_id\
                            if iSubject not in goldenset]
            workflowDictSubjectSets[iWorkflow] = subjectset_id

        return workflowDictSubjectSets


class GravitySpyProject(ZooProject):
    '''
    `GravitySpyProject` inherits from the `ZooProject` class
    and adds some GravitySpy specific wrapper functions
    '''
    def get_level_structure(self, workflow=None, IDfilter=''):
        """Parameters
        ----------
        workflow : `int`, optional, default None
        IDfilter : `str`, optional, default ''

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        level_structure = {}
        workflowDictSubjectSets = self.get_subject_sets_per_workflow(workflow=workflow)

        for iworkflow in workflowDictSubjectSets.keys():
            # If it is final workflow level 4 subject sets are also linked
            # so need to filter for level 5 subject sets
            if int(iworkflow) == 2117:
                IDfilter = IDfilter + ' (M)'

            subjectset_id = workflowDictSubjectSets[iworkflow]
            # Determine Display names of subject set
            subjectset_displayname_id = {}
            for iSubjectSet in subjectset_id:
                tmp2 = SubjectSet.find(iSubjectSet)
                displayname = str(tmp2.raw['display_name'])
                if IDfilter in displayname:
                    str_tmp = displayname.split(" '")[0].replace(' ','_')
                    subjectset_displayname_id[str_tmp] = \
                        (iworkflow, iSubjectSet,
                            [float(iThres)
                                for iThres 
                                    in re.findall("\d+\.\d+", displayname)
                            ]
                        )
            level_structure[iworkflow] = subjectset_displayname_id

        return level_structure


    def calculate_confusion_matrices(self):
        """Parameters
        ----------

        Returns
        -------
        A dict with keys of workflow IDs and values list
        of golden sets associated with that workflow
        """
        
        # Load classifications, and golden images from DB
        classifications = EventTable.fetch('gravityspy', 'classificationsdev',
                                           columns=['links_user',
                                                    'links_subjects',
                                                    'links_workflow',
                                                    'annotations_value_choiceINT'])

        classifications = classifications.to_pandas() 
        golden_images = EventTable.fetch('gravityspy', 'goldenimages')
        golden_images_df = golden_images.to_pandas()

        # Make sure choice is a valid index
        classifications = classifications.loc[
                              classifications.annotations_value_choiceINT != -1
                                             ]

        # Make sure to evaluate only logged in users
        classifications = classifications.loc[classifications.links_user != 0]

        # Ignore NONEOFTHEABOVE classificatios when constructing confusion
        # matrix
        classifications = classifications.loc[
                              classifications.annotations_value_choiceINT != 12
                                             ]

        # From answers Dict determine number of classes
        numClasses = len(self.get_answers(workflow=2360).values()[0])

        # merge the golden image DF with th classification (this merge is on
        # links_subject (i.e. the zooID of the image classified)
        image_and_classification = classifications.merge(golden_images_df,
                                                         on=['links_subjects'])

        # This is where the power of pandas comes in...on the fly in very quick
        # order we can fill all users confusion matrices
        # by smartly chosen groupby
        test = image_and_classification.groupby(['links_user',
                                                 'annotations_value_choiceINT',
                                                 'GoldLabel'])
        test = test.count().links_subjects.to_frame().reset_index()

        # Create "Sparse Matrices" and perform a normalization task on them.
        # Afterwards determine if the users diagonal is above the threshold set above
        confusion_matrices = pd.DataFrame()
        for iUser in test.groupby('links_user'):
            columns = iUser[1].annotations_value_choiceINT
            rows = iUser[1]['GoldLabel']
            entry = iUser[1]['links_subjects']
            tmp = coo_matrix((entry, (rows,columns)), shape=(numClasses,
                                                             numClasses))
            conf_divided, a1, a2, a3 = \
                np.linalg.lstsq(np.diagflat(tmp.sum(axis=1)),
                                            tmp.todense(rcond=None))
            confusion_matrices = \
                confusion_matrices.append(pd.DataFrame(
                                                      {'userID' : iUser[0],
                                                       'conf_matrix' : [conf_divided],
                                                       'alpha' : [np.diag(conf_divided)]},
                                                      index=[iUser[0]]))

        return confusion_matrices


    def determine_level(self, alpha=None):

        answers = self.get_answers(workflow=2360)
        if not alpha:
            numClasses = len(answers.values()[0])
            alpha = .7*np.ones(numClasses)
            alpha[4] = 0.65

        

        # Retrieve Answers
        answersDictRev =  dict(enumerate(sorted(answers['2360'].keys())))
        answersDict = dict((str(v),k) for k,v in answersDictRev.iteritems())
