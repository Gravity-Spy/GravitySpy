#!/usr/bin/env python

# ---- Import standard modules to the python path.

from panoptes_client import *
import re, operator
import pickle

#This function generically flatten a dict
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        try:
            items.extend(flatten(v, new_key, sep=sep).items())
        except:
            items.append((new_key, v))
    return dict(items)

def getAnswers(ProjectID):
    try:
        print('Trying to load dict from cache...')
        inputFile = open('workflowDictAnswers.pkl', 'rb')
        # Pickle dictionary using protocol 0.
        workflowDictAnswers = pickle.load(inputFile)
        print('successful')
        return workflowDictAnswers

    except:
        print('failure...')
        print('generating dict on the fly')
        # now determine infrastructure of workflows so we know what workflow this image belongs in
        workflowDictAnswers = {}
        tmp = Project.find(ProjectID)
        project_flat = flatten(tmp.raw)
        order = project_flat['configuration_workflow_order']

        # Determine workflow order
        workflows = [int(str(iWorkflow)) for iWorkflow in order]

        # Determine possible answers to the workflows
        for iWorkflow in workflows:
            workflow = Workflow.find(iWorkflow)
            if workflow.raw['tasks']['T1']['questionsMap']:
                workflowDictAnswers[iWorkflow] = workflow.raw['tasks']['T1']['questionsMap']

                # Find Answers with follow ups
                followupAns = [k for k, v in workflow.raw['tasks']['T1']['questionsMap'].iteritems() if v != []]

                # Loop over follow answers which have follow ups to them.
                for iFollow in followupAns:
                    questionsAndAnswersDict = {}

                    # Loop over questions
                    for iQuestion in workflowDictAnswers[iWorkflow][iFollow]:
                        if iQuestion in workflow.raw['tasks']['T1']['questions'].keys():
                            questionsAndAnswersDict[iQuestion] = workflow.raw['tasks']['T1']['questions'][iQuestion]['answersOrder']

                workflowDictAnswers[iWorkflow][iFollow] = questionsAndAnswersDict

            else:

                answerDict = {}

                for iAnswer in workflow.raw['tasks']['T1']['choicesOrder']:
                    answerDict[iAnswer] = []
                workflowDictAnswers[iWorkflow] = answerDict

        print('Saving dict to cache...')
        output = open('workflowDictAnswers.pkl', 'wb')
        # Pickle dictionary using protocol 0.
        pickle.dump(workflowDictAnswers, output)

        return workflowDictAnswers
