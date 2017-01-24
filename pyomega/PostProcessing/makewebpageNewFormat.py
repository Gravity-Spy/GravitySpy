# Import Python Modules
import json
import os, subprocess, shlex
import glob
import numpy as np
import pdb
import csv
from jinja2 import Environment, FileSystemLoader
import re
import pandas as pd
import optparse
from operator import itemgetter
import difflib

from gwpy.table.lsctables import SnglBurstTable
from gwpy.segments import SegmentList, Segment

from matplotlib import use
use('agg')
from matplotlib import (pyplot as plt, cm)

# Set this options so that we can print the href link to the image fromt he glitch metadata table
pd.set_option('display.max_colwidth',1000)

###############################################################################
##########################                             ########################
##########################   Func: parse_commandline   ########################
##########################                             ########################
###############################################################################
# Definite Command line arguments here

def parse_commandline():
    """Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--outPath", help="Where you would like the GSpy out pages to live")
    parser.add_option("--dataPath", help="Path to omega scans")
    parser.add_option("--metadata", help="Text file of metadata")
    parser.add_option("--ML-metadata", help="Text file of ML metadata")
    parser.add_option("--GravitySpy", action="store_true", default=False,help="Make GravitySpy output summary page")
    parser.add_option("--ML", action="store_true", default=False,help="Make Training Set summary page")
    parser.add_option("--TrainingSet", action="store_true", default=False,help="Make ML summary page")
    parser.add_option("--O1GlitchClassification", action="store_true", default=False,help="O1 Glitch Classification Paper Summary Page")
    parser.add_option("--verbose", action="store_true", default=False,help="Run in Verbose Mode")
    opts, args = parser.parse_args()


    return opts


opts = parse_commandline()
###############################################################################
##########################                             ########################
##########################   HTML HEADERS              ########################
##########################                             ########################
###############################################################################

tableHeader = """<!DOCTYPE HTML PUBLIC '-//W3C//DTD HTML 4.01 Transitional//EN'>
<html lang="en">
<head>
<link media="all" href="main.css" type="text/css" rel="stylesheet" />
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"></script>
<script src= https://cdn.datatables.net/1.10.12/js/jquery.dataTables.min.js></script>
<script src=https://cdn.datatables.net/1.10.12/js/dataTables.bootstrap.min.js></script> 
<!-- Latest compiled and minified CSS -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous">

<!-- Optional theme -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap-theme.min.css" integrity="sha384-fLW2N01lMqjakBkx3l/M9EahuwpSfeNvV63J5ezn3uZzapT0u7EYsXMjQV+0En5r" crossorigin="anonymous">

<link rel="stylesheet" href="https://cdn.datatables.net/1.10.12/css/dataTables.bootstrap.min.css">

<!-- Latest compiled and minified JavaScript -->
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js" integrity="sha384-0mSbJDEHialfmuBBQP6A4Qrprq5OVfW37PRR3j5ELqxss1yVqOtnepnHVP9aJ7xS" crossorigin="anonymous"></script>
<title>{{ page_title }}</title>
</head>
<div class="container">
    <div class="jumbotron" align="center">
    <h2 align="center"> Big Summary Table </h2>
    </div>
    <div class="panel panel-success">
        <div class="panel-body">
"""

tableFooter = """
        </div>
    </div>
</div>
<script>
$(document).ready(function() {
    $('#bigtable').DataTable();
} );
</script>"""

ExtraHTML = ''
ExtraHTML2 = ''
if opts.O1GlitchClassification:
    ExtraHTML = """  d3.select("[name=Pipeline]").on("change", function(){
    pipeline = this.value;
    d3.selectAll(".dot")
        // .transition()
        // .duration(500)
        .style("opacity", 0.0)
        // filter out the ones we want to show and apply properties
        .filter(function(d) {
            return d.Pipeline == pipeline;
        })
            .style("opacity", function(d) { return d.score; }); // need this line to unhide dots
    });"""

    ExtraHTML2 ="""<div id="label"><b>Pipeline:</b></div>
  <select name="Pipeline" id="PipelineID">\n"""
    for iX in metadata.Pipeline.unique():
        ExtraHTML2 = ExtraHTML2 + '<option value ="{0}">{0}</option>\n'.format(iX)
    ExtraHTML2 = ExtraHTML2 + "  </select>\n"
    ExtraHTML2 = ExtraHTML2 + "  </br></br>\n"

###############################################################################
##########################                     ################################
##########################      MAIN CODE      ################################
##########################                     ################################
###############################################################################

outPath  = opts.outPath
dataPath = opts.dataPath

outPath          += '/'

# Determine type of summary page being made
if opts.GravitySpy:
    summaryType = "tmp/"
elif opts.ML:
    summaryType = "O2/"
elif opts.TrainingSet:
    summaryType = "TrainingSet/"
elif opts.O1GlitchClassification:
    summaryType = "O1GlitchClassification/"
else:
    ValueError("Please supply type of summary page you want")

outPath = outPath + summaryType
triggerGrams = outPath + 'triggerGrams/'
indPages         = outPath + 'indPages/'

dataPath += '/'

# report status
if not os.path.isdir(outPath):
    if opts.verbose:
        print('creating main directory')
    os.makedirs(outPath)

# report status
if not os.path.isdir(triggerGrams):
    if opts.verbose:
        print('creating triggerGrams directory')
    os.makedirs(triggerGrams)

# report status
if not os.path.isdir(indPages):
    if opts.verbose:
        print('creating individual Classes directory')
    os.makedirs(indPages)
if opts.verbose:
    print('outputDirectory:  {0}'.format(outPath))

# Determine type of summary page being made
if opts.GravitySpy:
    header1 = "GravitySpy Output"
elif opts.ML:
    header1 = "ML Output"
elif opts.TrainingSet:
    header1 = "Training Set"
elif opts.O1GlitchClassification:
    header1 = "O1 Glitch Classification"
else:
    ValueError("Please supply type of summary page you want")

metadata = pd.read_hdf('{0}'.format(opts.metadata))

if opts.ML:
    files = glob.glob('{0}/*.h5'.format(opts.ML_metadata))
    tmp = pd.read_hdf('{0}'.format(files[0]))
    for iFile in files[1::]:
        tmp = pd.concat([tmp,pd.read_hdf('{0}'.format(iFile))])
    tmp.to_hdf('ML_GSpy.h5','gspy_ML_classification')
    metadata = metadata.merge(tmp)

# Get types from label columns
types = sorted(metadata.Label.unique().tolist())
summaryPage = open('{0}/index.html'.format(outPath),"w")
env = Environment(loader=FileSystemLoader('./'))
template = env.get_template('home.html')
print >>summaryPage, template.render(types=types,header=header1,gpsStart=np.floor(metadata.peakGPS.min()),gpsEnd=np.ceil(metadata.peakGPS.max()),ExtraHTML=ExtraHTML,ExtraHTML2=ExtraHTML2)
summaryPage.close()

ID              = []
imagePathAllInd = []
imagePathAllBig = []
scoreInd   = []
Pipeline   = []

for Type in types:
    imagePaths = []
    tmp1 = metadata[metadata.Label == Type]
    tmp1 = tmp1.sort_values('{0}'.format(Type),ascending=False)
    for IDtmp in tmp1.uniqueID:
        ID.append(IDtmp)
        image = glob.glob('{0}/{1}.png'.format(dataPath,IDtmp))
        # We have identified the path to the image. Now we need to do a comparison between outPath and dataPath to find the right relative path of this image to the individual page
        imageTmp = filter(None, image[0].split('/'))
        pathTmp  = filter(None, indPages.split('/'))
        tmp = [item for item in imageTmp if item in pathTmp]
        imagePaths.append(''.join(['../'] * len(filter(None,indPages.split(tmp[-1])[1].split('/')))) + image[0].split(tmp[-1])[1])
        imagePathAllInd.append(''.join(['../'] * len(filter(None,indPages.split(tmp[-1])[1].split('/')))) + image[0].split(tmp[-1])[1])
        imagePathAllBig.append(''.join(['../'] * (len(filter(None,indPages.split(tmp[-1])[1].split('/')))-1)) + image[0].split(tmp[-1])[1])
        scoreInd.append(tmp1.loc[tmp1.uniqueID == IDtmp,Type].iloc[0])

    # Open a new html page which is named after the type
    title = dict(zip(imagePaths,scoreInd))
    pipeline = dict(zip(imagePaths,Pipeline))
    indSummary= open('{0}/{1}.html'.format(indPages,Type),"w")
    template = env.get_template('individual.html')
    print >>indSummary, template.render(types=types,imagePath=imagePaths,glitchType=Type,title=title,gpsStart=np.floor(metadata.peakGPS.min()),gpsEnd=np.ceil(metadata.peakGPS.max()),ExtraHTML=ExtraHTML,ExtraHTML2=ExtraHTML2,Pipeline=pipeline)
    indSummary.close()

# Create pandas table with ID and type labels. This is to join all the information together for a giant data table.
tmpTable = pd.DataFrame({'uniqueID' : ID, 'linkInd' : imagePathAllInd, 'linkBig' : imagePathAllBig})
bigTable = pd.merge(tmpTable,metadata)
###############################################################################
##########################                           ##########################
##########################   DATA TABLES AND FILES   ##########################
##########################                           ##########################
###############################################################################

# For d3js fun!
# Initialize list
json_metadata = []
for iCol in bigTable.keys()[bigTable.dtypes == np.float32]:
    bigTable[iCol] = bigTable[iCol].astype(np.float64)

json_metadata = []
for iCol in bigTable.keys()[bigTable.dtypes == np.int32]:
    bigTable[iCol] = bigTable[iCol].astype(np.int64)

for i in xrange(len(bigTable)):
    temp_dict={}
    col_vals = bigTable.columns.values  # Save the values of the columns
    temp_df = bigTable.iloc[i]
    temp_dict["day"] = int(str(temp_df['peakGPS'])[0:5])  # Add the day for scrolling
    # Determine type of summary page being made
    if opts.GravitySpy:
        temp_dict['score'] = temp_df[temp_df.Label]  # Add maximum score
    elif opts.ML:
        temp_dict['score'] = temp_df[temp_df.Label]   # Add maximum score
    elif opts.TrainingSet:
        temp_dict['score'] = 1   # Add maximum score
    elif opts.O1GlitchClassification:
        temp_dict['score'] = 1
    else:
        ValueError("Please supply type of summary page you want")
    for field in list(col_vals):
        temp_dict[field] = temp_df[field]  # Add all the metadata

    json_metadata.append(temp_dict)  # Now, we are building up a dict with each ID as a key

with open('{0}/metadata.json'.format(outPath), 'w') as f:
    json.dump(json_metadata, f)

# Make json file for indiviudal page d3js fun
for Type in types:
    # Initialize list
    json_metadata = []
    df = bigTable[bigTable.Label == Type]
    for i in xrange(len(df)):
        temp_dict={}
        col_vals = df.columns.values  # Save the values of the columns
        temp_df = df.iloc[i]
        temp_dict["day"] = int(str(temp_df['peakGPS'])[0:5])  # Add the day for scrolling
        # Determine type of summary page being made
        if opts.GravitySpy:
            temp_dict['score'] = max(temp_df[types])   # Add maximum score
        elif opts.ML:
            temp_dict['score'] = max(temp_df[types])   # Add maximum score
        elif opts.TrainingSet:
            temp_dict['score'] = 1   # Add maximum score
        elif opts.O1GlitchClassification:
            temp_dict['score'] = 1
        else:
            ValueError("Please supply type of summary page you want")
        for field in list(col_vals):
            temp_dict[field] = temp_df[field]  # Add all the metadata

        json_metadata.append(temp_dict)  # Now, we are building up a dict with each ID as a key

    with open('{0}/{1}_metadata.json'.format(indPages,Type), 'w') as f:
        json.dump(json_metadata, f)

# Save to CSV now before creating links for the purpose of the HTML table
bigTable.drop(['linkInd','linkBig'],1).to_csv(open('{0}/metadata.csv'.format(outPath),'w'))
for Type in types:
    bigTable[bigTable.Label == Type].drop(['linkInd','linkBig'],1).to_csv(open('{0}/{1}_metadata.csv'.format(indPages,Type),'w'))

# This allows us to add a column of links that will link to the omega scan image for this row of trigger data.
bigTable.insert(0,'links',bigTable.linkBig.apply(lambda x: '<a href="' + '{0}"'.format(x) + '>image</a>'))
bigTable.drop('linkBig',1,inplace=True)

# Make HTML table for uber table
mHTML = open('{0}/metadata.html'.format(outPath),'w')
mHTML.write(tableHeader + '\n')
bigTable.drop('linkInd', axis=1).to_html(mHTML,classes= 'table table-striped table-bordered" id = "bigtable" width="100%" cellspacing="0',float_format=lambda x: '%10.7f' % x,escape=False)
mHTML.write(tableFooter + '\n')
mHTML.close()

# Remove link for big table
bigTable.drop('links',1,inplace=True)

# Make HTML table for individual class pages

# This allows us to add a column of links that will link to the omega scan image for this row of trigger data.
bigTable.insert(0,'links',bigTable.linkInd.apply(lambda x: '<a href="' + '{0}"'.format(x) + '>image</a>'))
bigTable.drop('linkInd',1,inplace=True)

for Type in types:
    mHTML = open('{0}/{1}_metadata.html'.format(indPages,Type),'w')
    mHTML.write(tableHeader + '\n')
    bigTable[bigTable.Label == Type].to_html(mHTML,classes= 'table table-striped table-bordered" id = "bigtable" width="100%" cellspacing="0',float_format=lambda x: '%10.7f' % x,escape=False)
    mHTML.write(tableFooter + '\n')
    mHTML.close()

# Writing XML table


###############################################################################
##########################                           ##########################
##########################   DATA TABLES AND FILES   ##########################
##########################                           ##########################
###############################################################################
