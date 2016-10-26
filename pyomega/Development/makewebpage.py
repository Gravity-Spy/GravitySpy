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
    parser.add_option("--omicrontriggers", help="XML of omicron triggers")
    parser.add_option("--GravitySpy", action="store_true", default=False,help="Make GravitySpy output summary page")
    parser.add_option("--ML", action="store_true", default=False,help="Make Training Set summary page")
    parser.add_option("--TrainingSet", action="store_true", default=False,help="Make ML summary page")
    parser.add_option("--verbose", action="store_true", default=False,help="Run in Verbose Mode")
    opts, args = parser.parse_args()


    return opts

###############################################################################
##########################                     ################################
##########################      MAIN CODE      ################################
##########################                     ################################
###############################################################################

opts = parse_commandline()

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

# Based on the directory you point to determine the classes of glitches.
types = [ name for name in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, name)) ]
types = sorted(types)
#types[10] = 'No_Glitch'
#types[9] = 'None_of_the_Above'

# Open and create a home page
# Determine type of summary page being made
if opts.GravitySpy:
    header1 = "GravitySpy Output"
elif opts.ML:
    header1 = "ML Output"
elif opts.TrainingSet:
    header1 = "Training Set"
else:
    ValueError("Please supply type of summary page you want")

# iN is represent which Glitch we are making html pages and trigger grams for
iN = 1

# Initialize a blank ID list and a blank IDType
IDType          = []
ID              = []
imagePathAllInd = []
imagePathAllBig = []

# Load omicron trigger metadata
metadata = csv.reader(open('{0}'.format(opts.metadata)), delimiter=" ")
# Sort by SNR
sortedlist = sorted(metadata, key=lambda row: row[0], reverse=True)
# Put the metadata into a pandas DF

if opts.GravitySpy:
    metadata = pd.DataFrame(sortedlist,columns=['snr','amplitude','peak_frequency','central_freq','duration','bandwidth','chisq','chisq_dof','GPStime','ID','channel'])
elif opts.ML:
    metadata = pd.DataFrame(sortedlist,columns=['snr','amplitude','peak_frequency','central_freq','duration','bandwidth','chisq','chisq_dof','GPStime','ID','channel'])
elif opts.TrainingSet:
    metadata = pd.DataFrame(sortedlist,columns=['snr','amplitude','peak_frequency','central_freq','duration','bandwidth','chisq','chisq_dof','GPStime','ID','channel','Label'])
else:
    ValueError("Please supply type of summary page you want")


metadata.GPStime = metadata.GPStime.apply(float)
summaryPage = open('{0}/index.html'.format(outPath),"w")
env = Environment(loader=FileSystemLoader('./'))
template = env.get_template('home.html')
print >>summaryPage, template.render(types=types,header=header1,gpsStart=np.floor(metadata.GPStime.min()),gpsEnd=np.ceil(metadata.GPStime.max()))
summaryPage.close()

for Type in types:

    try:
        imagePaths = []
        scoreInd   = []
        if opts.ML:
            # Open the scores file for all the images put into that glitch category
            reader = csv.reader(open('{0}/{1}/scores.csv'.format(dataPath,Type)), delimiter=",")
            # sort it
            list1 = sorted(reader)
            list2 = [x for x in list1 if len(x) == (21)]
            sortedlist = sorted(list2, key=itemgetter(iN),reverse=True)
            for score in sortedlist:
                # The score.csv file is laid out so that the first entry is the unqiueID of the image. The subsequent inputs are the 1 by 20 scores which correspond to the types in the order of the "types" variable defined above.
                IDType.append(Type)
                ID.append(score[0])
                image = glob.glob('{0}/{1}/*/{2}.png'.format(dataPath,Type,score[0]))
                # We have identified the path to the image. Now we need to do a comparison between outPath and dataPath to find the right relative path of this image to the individual page
                imageTmp = filter(None, image[0].split('/'))
                pathTmp  = filter(None, indPages.split('/'))
                tmp = [item for item in imageTmp if item in pathTmp]
                imagePaths.append(''.join(['../'] * len(filter(None,indPages.split(tmp[-1])[1].split('/')))) + image[0].split(tmp[-1])[1])
                imagePathAllInd.append(''.join(['../'] * len(filter(None,indPages.split(tmp[-1])[1].split('/')))) + image[0].split(tmp[-1])[1])
                imagePathAllBig.append(''.join(['../'] * (len(filter(None,indPages.split(tmp[-1])[1].split('/')))-1)) + image[0].split(tmp[-1])[1])
                scoreInd.append(score[iN])

        elif opts.TrainingSet:
            tmp = metadata[metadata.Label == Type]
            for IDtmp in tmp.ID:
                IDType.append(Type)
                ID.append(IDtmp)
                image = glob.glob('{0}/{1}/{2}.png'.format(dataPath,Type,IDtmp))
                # We have identified the path to the image. Now we need to do a comparison between outPath and dataPath to find the right relative path of this image to the individual page
                imageTmp = filter(None, image[0].split('/'))
                pathTmp  = filter(None, indPages.split('/'))
                tmp = [item for item in imageTmp if item in pathTmp]
                imagePaths.append(''.join(['../'] * len(filter(None,indPages.split(tmp[-1])[1].split('/')))) + image[0].split(tmp[-1])[1])
                imagePathAllInd.append(''.join(['../'] * len(filter(None,indPages.split(tmp[-1])[1].split('/')))) + image[0].split(tmp[-1])[1])
                imagePathAllBig.append(''.join(['../'] * (len(filter(None,indPages.split(tmp[-1])[1].split('/')))-1)) + image[0].split(tmp[-1])[1])


        # Open a new html page which is named after the type
        title = dict(zip(imagePaths,scoreInd))
        indSummary= open('{0}/{1}.html'.format(indPages,Type),"w")
        template = env.get_template('individual.html')
        print >>indSummary, template.render(imagePath=imagePaths,glitchType=Type,title=title)
        indSummary.close()
        iN = iN + 1
    except:
        iN = iN + 1
        print('Warning: {0} failed'.format(Type))


# Create pandas table with ID and type labels. This is to join all the information together for a giant data table.
tmpTable = pd.DataFrame({'ID' : ID, 'Label' : IDType, 'linkInd' : imagePathAllInd, 'linkBig' : imagePathAllBig})
bigTable = pd.merge(tmpTable,metadata)

if opts.ML:
    # Load aggregate score information
    reader = csv.reader(open('{0}/allscores.csv'.format(dataPath)), delimiter=",")
    list1 = sorted(reader)
    list2 = [x for x in list1 if len(x) == (21)]
    scores = pd.DataFrame(list2,columns=["ID","Air_Compressor","Blip","Chirp","Extremely_Loud","Helix","Koi_Fish","Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","None_of_the_Above","No_Glitch","Paired_Doves","Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Wandering_Line","Whistle"])
    bigTable = pd.merge(bigTable,scores)

###############################################################################
##########################                           ##########################
##########################   DATA TABLES AND FILES   ##########################
##########################                           ##########################
###############################################################################

# For d3js fun!
# Initialize list
json_metadata = []

# Make Frequency, SNR, and GPStime floats
bigTable.GPStime        = bigTable.GPStime.apply(float)
bigTable.snr            = bigTable.snr.apply(float)
bigTable.peak_frequency = bigTable.peak_frequency.apply(float)

bigTable.Air_Compressor = bigTable.Air_Compressor.apply(float)
bigTable.Blip = bigTable.Blip.apply(float)
bigTable.Chirp = bigTable.Chirp.apply(float)
bigTable.Extremely_Loud = bigTable.Extremely_Loud.apply(float)
bigTable.Helix = bigTable.Helix.apply(float)
bigTable.Koi_Fish = bigTable.Koi_Fish.apply(float)
bigTable.Light_Modulation = bigTable.Light_Modulation.apply(float)
bigTable.Low_Frequency_Burst = bigTable.Low_Frequency_Burst.apply(float)
bigTable.Low_Frequency_Lines = bigTable.Low_Frequency_Lines.apply(float)
bigTable.None_of_the_Above = bigTable.None_of_the_Above.apply(float)
bigTable.No_Glitch = bigTable.No_Glitch.apply(float)
bigTable.Paired_Doves = bigTable.Paired_Doves.apply(float)
bigTable.Power_Line = bigTable.Power_Line.apply(float)
bigTable.Repeating_Blips = bigTable.Repeating_Blips.apply(float)
bigTable.Scattered_Light = bigTable.Scattered_Light.apply(float)
bigTable.Scratchy = bigTable.Scratchy.apply(float)
bigTable.Tomte = bigTable.Tomte.apply(float)
bigTable.Violin_Mode = bigTable.Violin_Mode.apply(float)
bigTable.Wandering_Line = bigTable.Wandering_Line.apply(float)
bigTable.Whistle = bigTable.Whistle.apply(float)

for i in xrange(len(bigTable)):
    temp_dict={}
    col_vals = bigTable.columns.values  # Save the values of the columns
    temp_df = bigTable.iloc[i]
    temp_dict["day"] = int(str(temp_df['GPStime'])[0:5])  # Add the day for scrolling
    # Determine type of summary page being made
    if opts.GravitySpy:
        temp_dict['score'] = max(temp_df[types])   # Add maximum score
    elif opts.ML:
        temp_dict['score'] = max(temp_df[types])   # Add maximum score
    elif opts.TrainingSet:
        temp_dict['score'] = 0   # Add maximum score
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
        temp_dict["day"] = int(str(temp_df['GPStime'])[0:5])  # Add the day for scrolling
        # Determine type of summary page being made
        if opts.GravitySpy:
            temp_dict['score'] = max(temp_df[types])   # Add maximum score
        elif opts.ML:
            temp_dict['score'] = max(temp_df[types])   # Add maximum score
        elif opts.TrainingSet:
            temp_dict['score'] = 0   # Add maximum score
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

###############################################################################
##########################                           ##########################
##########################   DATA TABLES AND FILES   ##########################
##########################                           ##########################
###############################################################################
