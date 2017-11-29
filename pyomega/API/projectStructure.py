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


def main(ProjectID,IDfilter=''):
    # First try to read the workflowDictSubjectSets from cache
    try:
        print 'Trying to load dict from cache...'
        inputFile = open('workflowDictSubjectSets.pkl', 'rb')
        # Pickle dictionary using protocol 0.
        workflowDictSubjectSets = pickle.load(inputFile)
        print 'successful'
        return workflowDictSubjectSets

    except:
        print 'failure...'
        print 'generating dict on the fly'
        # now determine infrastructure of workflows so we know what workflow this image belongs in
        workflowDictSubjectSets = {}
        tmp = Project.find(ProjectID)
        project_flat = flatten(tmp.raw)
        order = project_flat['configuration_workflow_order']
        # Determine workflow order
        workflows = [int(str(iWorkflow)) for iWorkflow in order]
        # Determine subject sets and answers
        for iWorkflow in workflows:
            tmp1 = Workflow.find(iWorkflow)
            tmp1 = flatten(tmp1.raw)
            # check if golden set exists for workflow
            try:
                goldenset = tmp1['configuration_gold_standard_sets']
            except:
                goldenset = []
            # Determine subject sets associated with this workflow
            subjects_workflow = tmp1['links_subject_sets']
            subjectset_id = [int(str(iSubject)) for iSubject in subjects_workflow]
            subjectset_id = [iSubject for iSubject in subjectset_id if iSubject not in goldenset]


            # If it is final workflow level 4 subject sets are also linked so need to filter
            # for level 5 subject sets
            if iWorkflow == 2117:
                IDfilter = IDfilter + ' (M)'
            # Determine Display names of subject set
            subjectset_displayname_id = {}
            for iSubjectSet in subjectset_id:
                tmp2 = SubjectSet.find(iSubjectSet)
                displayname = str(tmp2.raw['display_name'])
                if IDfilter in displayname:
                    subjectset_displayname_id[displayname.split(" '")[0].replace(' ','_')] = \
                        (iWorkflow, iSubjectSet,
                            [float(iThres) for iThres in re.findall("\d+\.\d+", displayname)])
            workflowDictSubjectSets[iWorkflow] = subjectset_displayname_id

        print 'Saving dict to cache...'
        output = open('workflowDictSubjectSets.pkl', 'wb')
        # Pickle dictionary using protocol 0.
        pickle.dump(workflowDictSubjectSets, output)

        return workflowDictSubjectSets

if __name__ == "__main__":
   main(ProjectID)
