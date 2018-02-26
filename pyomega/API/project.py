#!/usr/bin/env python

# ---- Import standard modules to the python path.

from panoptes_client import *
import re, operator

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

def getGoldenSubjectSets(ProjectID):
    # now determine infrastructure of workflows so we know what workflow this image belongs in
    workflowGoldenSetDict = {}
    tmp = Project.find(ProjectID)
    project_flat = flatten(tmp.raw)
    order = project_flat['configuration_workflow_order']
    # Determine workflow order
    workflows = [int(str(iWorkflow)) for iWorkflow in order]
    # Determine subject sets and answers
    for iWorkflow in workflows:
        tmp1 = Workflow.find(iWorkflow)
        tmp1 = flatten(tmp1.raw)
        try:
            workflowGoldenSetDict[iWorkflow] = tmp1['configuration_gold_standard_sets']
        except:
            workflowGoldenSetDict[iWorkflow] = []

    return workflowGoldenSetDict

def getGoldenImages(workflowGoldenSetDict):
    workflowGoldenSetImagesDict = {}

    for iWorkflow in workflowGoldenSetDict.keys():
        goldenImages = {}

        for iGoldenSubjectSet in workflowGoldenSetDict[iWorkflow]:
            tmp = SubjectSet.find(iGoldenSubjectSet)
            tmpSubjects = tmp.subjects()

            while True:
                try:
                    nextSubject = tmpSubjects.next()
                    goldenImages[str(nextSubject.id)] = [str(nextSubject.raw['metadata']['subject_id']), str(nextSubject.raw['metadata']['#Label'])]
                except:
                    break

        workflowGoldenSetImagesDict[iWorkflow] = goldenImages

    return workflowGoldenSetImagesDict

def getGoldenImagesAsInts(workflowGoldenSetDict):
    from pyomega.API.getLabelDict import getAnswers
    import pandas as pd

    answers = getAnswers('1104')
    answersDictRev =  dict(enumerate(sorted(answers[2360].keys())))
    answersDict = dict((str(v),k) for k,v in answersDictRev.iteritems())

    goldenImagesList = []

    for iWorkflow in sorted(workflowGoldenSetDict.keys()):

        if not workflowGoldenSetDict[iWorkflow]:
            continue

        for iGoldenSubjectSet in workflowGoldenSetDict[iWorkflow]:

            tmp = SubjectSet.find(iGoldenSubjectSet)
            tmpSubjects = tmp.subjects()

            while True:
                try:
                    nextSubject = tmpSubjects.next()
                    goldenImagesList.append([int(nextSubject.id), answersDict[str(nextSubject.raw['metadata']['#Label']).upper().translate(None,'() ')],nextSubject.raw['metadata']['subject_id']])
                except:
                    break

    return pd.DataFrame(goldenImagesList,columns=['links_subjects', 'GoldLabel', 'uniqueID'])

