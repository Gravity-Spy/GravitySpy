# Import Python Modules
import os, subprocess, shlex
import glob
import numpy as np
import pdb
import csv
import re
import pandas as pd
import optparse
from operator import itemgetter

from gwpy.table.lsctables import SnglBurstTable
from gwpy.segments import SegmentList, Segment

import matplotlib.pyplot as plt

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
    parser.add_option("--omicrontriggers", help="XML of omicron triggers")
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

# iN is represent which Glitch we are making html pages and trigger grams for
iN = 1

colors = ["r","darksalmon","sienna","sandybrown","gold","olivedrab","chartreuse","darkkhaki","darkslategray","darkblue","c","dodgerblue","teal","pink","cadetblue","springgreen","purple","darkgoldenrod","chocolate","magenta"]
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd','o', 'v', '^', '<', '>', '8', 's', 'p']

# Initialize a list of the Glitch types
types = ["Air_Compressor","Blip","Chirp","Extremely_Loud","Helix","Koi_Fish","Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","None_of_the_Above","No_Glitch","Paired_Doves","Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Wandering_Line","Whistle"]

# Initialize empty lists for all the Glitch types
Whistle, Low_Frequency_Burst, Chirp, Repeating_Blips, Scattered_Light, Light_Modulation, Extremely_Loud, Low_Frequency_Lines, Air_Compressor, Blip, Power_Line, Paired_Doves, Tomte, Wandering_Line, Helix, Scratchy, None_of_the_Above, Violin_Mode, Koi_Fish, No_Glitch = ([],)*len(types)

IDType = []
ID     = []
# First loop will take the the tiled omega scans (that is all four durations plotted at once) and tile them on the page from the highest ML score to the lowest for that category.

for Type in types: 

    # Open a new html page which is named after the type
    page= open('{0}/{1}.html'.format(indPages,Type),"w")

    # Create the same header for each type. This follows the masonry jQuery lay out and therefore you will need the masonry jQuery plug in to work. (http://masonry.desandro.com)
    header = """<head>
    <link rel="stylesheet" type="text/css" href="../../css/classes.css">
    <script src="../../node_modules/masonry.pkgd.min.js"></script>
</head>
<body>
    <h1>{0}</h1>

    <IMG SRC="{0}_Freq.png" ALT="Freq" WIDTH=600 HEIGHT=400">
    <IMG SRC="{0}_SNR.png" ALT="SNR" WIDTH=600 HEIGHT=400">
    <p><a href="{0}_metadata.html" > {0} metadata </a></p>
    <p><a href="{0}_metadata.csv" > {0} metadata exports </a></p>
    <p><a href="{0}_Freq.png" > {0} trigger Gram Freq</a></p>
    <p><a href="{0}_SNR.png" > {0} trigger Gram SNR</a></p>

    <div class="grid">
    <div class="grid-sizer"></div>""".format(Type)
    page.write(header + '\n')

    # We do try except here because there is no guarentee every category has an image in it from the ML output or the people output for that matter.
    try:
        # Open the scores file for all the images put into that glitch category
        reader = csv.reader(open('{0}/{1}/scores.csv'.format(dataPath,Type)), delimiter=",")
        # sort it
        list1 = sorted(reader)
        list2 = [x for x in list1 if len(x) == (len(types)+1)]
        sortedlist = sorted(list2, key=itemgetter(iN),reverse=True)
        # Begin loop over sorted list
        for score in sortedlist:
            # The score.csv file is laid out so that the first entry is the unqiueID of the image. The subsequent inputs are the 1 by 20 scores which correspond to the types in the order of the "types" variable defined above.
            locals()[Type].append(score[0])
            IDType.append(Type)
            ID.append(score[0])
            image = glob.glob('{0}/{1}/*/{2}.png'.format(dataPath,Type,score[0]))

            page.write('       <div class="grid-item">')
            #page.write('       <a href="../../PyOmegaProd/ML/{0}/Beginner/>')
            page.write(' <img src="../../{0}" title="{1} {2} = {3} {4} = {5} {6} = {7} {8} = {9} {10} = {11} {12} = {13} {14} = {15} {16} = {17} {18} = {19} {20} = {21} {22} = {23} {24} = {25} {26} = {27} {28} = {29} {30} = {31} {32} = {33} {34} = {35} {36} = {37} {38} = {39} {40} = {41}"'.format('/'.join(image[0].split('/')[4::]),score[0],types[0],score[1],types[1],score[2],types[2],score[3],types[3],score[4],types[4],score[5],types[5],score[6],types[6],score[7],types[7],score[8],types[8],score[9],types[9],score[10],types[10],score[11],types[11],score[12],types[12],score[13],types[13],score[14],types[14],score[15],types[15],score[16],types[16],score[17],types[17],score[18],types[18],score[19],types[19],score[20]))
            page.write(' /></div>\n')

        # Finish writing this html page
        page.write('    </div>\n')
        page.write('</body>')
        page.close()
    except:
        print("{0}:Failed".format(Type))
        page.close()
    iN = iN +1

colorsmapped  = dict(zip(types, colors))
markersmapped = dict(zip(types, markers))

# Create a uber pandas
allLabels = pd.DataFrame({"ID" : ID, "IDType" :IDType})

dataForPie = allLabels.IDType.value_counts()
data = np.asarray(dataForPie.to_dict().values())
percent= 100.*data/data.sum()

colorslab = [colorsmapped[i] for i in dataForPie.to_dict().keys()]
patches, texts = plt.pie(data, colors=colorslab, startangle=90, radius=1.2)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(dataForPie.to_dict().keys(), percent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, data),
                                          key=lambda x: x[2],
                                          reverse=True))

plt.legend(patches, labels, loc='upper left', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)

plt.savefig('{0}/test.png'.format(outPath))
# Load omicron trigger metadata
metadata = csv.reader(open('{0}'.format(opts.metadata)), delimiter=" ")

# Sort by SNR
sortedlist = sorted(metadata, key=lambda row: row[0], reverse=True)

# Put the metadata into a pandas DF
metadata = pd.DataFrame(sortedlist,columns=['snr','amplitude','peak_frequency','central_freq','duration','bandwidth','chisq','chisq_dof','GPStime','ID','channel'])

# Read in the scores again
iN = 1

events    = SnglBurstTable.read('{0}'.format(opts.omicrontriggers))
superPlotFreq = events.plot('time', 'peak_frequency',edgecolor='none',facecolor='w')
superPlotFreq.set_ylabel('Frequency [Hz]')
superPlotFreq.set_yscale('log')
superPlotFreq.set_ylim(10, 2048)
superPlotFreq.set_title('LIGO Livingston')
#superPlot.add_colorbar(clim=[1, 100], label='Signal-to-noise ratio', cmap='viridis')
axFreq = superPlotFreq.gca()

superPlotSNR = events.plot('time', 'snr',edgecolor='none',facecolor='w')
superPlotSNR.set_ylabel('SNR')
superPlotSNR.set_yscale('log')
superPlotSNR.set_ylim(7.5, 2500)
superPlotSNR.set_title('LIGO Livingston')
#superPlot.add_colorbar(clim=[1, 100], label='Signal-to-noise ratio', cmap='viridis')
axSNR = superPlotSNR.gca()

for Type in types: 
    try:
        # Read and sort the scores for this cateogry again
        reader = csv.reader(open('{0}/{1}/scores.csv'.format(dataPath,Type)), delimiter=",")
        # sort it
        list1 = sorted(reader)
        list2 = [x for x in list1 if len(x) == (len(types)+1)]
        scores = sorted(list2, key=itemgetter(iN))
        # Open html where the metadata plus ML scores will go
        mHTML = open('{0}/{1}_metadata.html'.format(indPages,Type),'w')

        # Take scores and make it a pandas. This will help us merge it with the
        # omicron triggers metadata pandas table
        scores = pd.DataFrame(scores,columns=["ID","Light_Modulation","Air_Compressor","Blip","Chirp","Extremely_Loud","Helix","Koi_Fish","Low_Frequency_Burst","Low_Frequency_Lines","No_Glitch","None_of_the_Above","Paired_Doves","Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Wandering_Line","Whistle"])
        # Find only the scores that correspond with the list of unique IDs for this category
        scores = scores[scores.ID.isin(locals()[Type])]
        locals()[Type] = metadata[metadata.ID.isin(locals()[Type])]

        # Set header for metadata table. This allows the table to be sortable
        header = """<head>
    <link rel="stylesheet" type="text/css" href="../css/classes.css">
    <script src="../node_modules/sorttable.js"></script>
               </head>"""
        mHTML.write(header + '\n')

        # Merge scores and metadata
        metadataAll = pd.merge(locals()[Type],scores)
        locals()[Type] = locals()[Type][locals()[Type].ID.isin(scores.ID)]
        # Output this merged table to a csv
        metadataAll.to_csv(open('{0}/{1}_metadata.csv'.format(indPages,Type),'w'))

        # This allows us to add a column of links that will link to the omega scan image for this row of trigger data.
        startOfLink = '<a href="../../GravitySpyO1ImageFinal/ML/{0}/Beginner/'.format(Type)
        metadataAll.insert(0,'links',metadataAll.ID.apply(lambda x: startOfLink + '{0}.png"'.format(x) + '>image</a>'))


        # Export the table as html
        metadataAll.to_html(mHTML,classes=['sortable'],float_format=lambda x: '%10.7f' % x,escape=False)
        mHTML.close()


        #tmp = events.vetoed([Segment(i) for i in zip(locals()[Type].GPStime,locals()[Type].GPStime)])
    except:
        print("{0}:Failed".format(Type))
    iN = iN +1

legargs = {
            'loc': 'upper left',
            'borderaxespad': 0,
            'numpoints': 1,
            'scatterpoints': 1,
            'handlelength': 1,
            'handletextpad': .5,
            'fontsize': 7
}

# Overlap superPlot with each category
for i, Types in enumerate(types):
    try:
        axFreq.scatter(locals()[Types].GPStime.apply(float), locals()[Types].peak_frequency.apply(float), color=colorsmapped[Types], marker=markersmapped[Types], label=Types.replace("_"," "), s=20)
        axSNR.scatter(locals()[Types].GPStime.apply(float), locals()[Types].snr.apply(float), color=colorsmapped[Types], marker=markersmapped[Types], label=Types.replace("_"," "), s=20)
        tmp = events.plot('time', 'peak_frequency',edgecolor='none',facecolor='w')
        tmp.set_ylabel('Frequency [Hz]')
        tmp.set_yscale('log')
        tmp.set_ylim(10, 2048)
        tmp.set_title('LIGO Livingston')
        tmpAx = tmp.gca()
        tmpAx.scatter(locals()[Types].GPStime.apply(float), locals()[Types].peak_frequency.apply(float), color=colorsmapped[Types], marker=markersmapped[Types], label=Types.replace("_"," "), s=20)
        handles, labels = tmpAx.get_legend_handles_labels()
        tmpAx.legend(handles, labels,**legargs)
        tmpAx.legend(handles, labels,**legargs)
        tmpfig = tmpAx.get_figure()
        tmpfig.savefig('{0}/{1}_Freq.png'.format(indPages,Types))
        tmpAx.clear()


        tmpSNR = events.plot('time', 'snr',edgecolor='none',facecolor='w')
        tmpSNR.set_ylabel('SNR')
        tmpSNR.set_yscale('log')
        tmpSNR.set_ylim(7.5, 2500)
        tmpSNR.set_title('LIGO Livingston')
        tmpAxSNR = tmp.gca()
        tmpAxSNR.scatter(locals()[Types].GPStime.apply(float), locals()[Types].snr.apply(float), color=colorsmapped[Types], marker=markersmapped[Types], label=Types.replace("_"," "), s=20)
        handles, labels = tmpAxSNR.get_legend_handles_labels()
        tmpAxSNR.legend(handles, labels,**legargs)
        tmpAxSNR.legend(handles, labels,**legargs)
        tmpfigSNR = tmpAxSNR.get_figure()
        tmpfigSNR.savefig('{0}/{1}_SNR.png'.format(indPages,Types))
        tmpAxSNR.clear()
    except:
        print("{0}:Failed".format(Types))

legargs = {
            'loc': 'upper left',
            'borderaxespad': 0,
            'numpoints': 1,
            'scatterpoints': 1,
            'handlelength': 1,
            'handletextpad': .5,
            'fontsize': 7
}

handles, labels = axFreq.get_legend_handles_labels()
superPlotFreq.legend(handles, labels,**legargs)
superPlotSNR.legend(handles, labels,**legargs)

startTime = events.get_peak().min()._LIGOTimeGPS__seconds - np.mod(events.get_peak().min()._LIGOTimeGPS__seconds,86400)
endTime = events.get_peak().max()._LIGOTimeGPS__seconds + (86400 - np.mod(events.get_peak().max()._LIGOTimeGPS__seconds,86400))

superPlotFreq.save('{0}/allTriggers_Freq.png'.format(outPath))
superPlotSNR.save('{0}/allTriggers_SNR.png'.format(outPath))
for iDay in range(startTime,endTime,86400):
    superPlotFreq.set_xlim(iDay,iDay+86400)
    superPlotFreq.set_epoch(iDay)
    superPlotFreq.save('{0}/{1}_Freq.png'.format(triggerGrams,iDay))
    superPlotSNR.set_xlim(iDay,iDay+86400)
    superPlotSNR.set_epoch(iDay)
    superPlotSNR.save('{0}/{1}_SNR.png'.format(triggerGrams,iDay))


# Finally we make the trigger grams into a tiled html page for easy viewing

page= open('{0}/ML_summary.html'.format(outPath),"w")

blurb = """All LIGO Livingston omicron triggers SNR>7.5 and 10<f<2048Hz and which have survived the C02 burst Data Quality Vetoes, categorized by  machine learning algorithms.
Start time = 1126400000 September 15, 2015
End time = 1137250000 January 18, 2016
Number of triggers = 29652
Fig 1: Time-frequency plot of all triggers, with colors and shapes assigned based on categories assigned to these in the training set.
Fig 2: Time-SNR plot of the same.
Fig 3: Breakdown of the number of triggers in each category.
Please click the link to daily cumulative trigger plots to see Fig 1 and 2 for individual days.
"""

header = """<head>
    <link rel="stylesheet" type="text/css" href="../css/classes.css">
    <script src="../node_modules/masonry.pkgd.min.js"></script>
</head>
<body>
    <h1>Gravity Spy ML Classified: Livingston</h1>
    <textarea rows="10" cols="100">
    {0}
    </textarea>
    <IMG SRC="allTriggers_Freq.png" ALT="Freq" WIDTH=600 HEIGHT=400">
    <IMG SRC="allTriggers_SNR.png" ALT="SNR" WIDTH=600 HEIGHT=400">
    <IMG SRC="test.png" ALT="TEST" WIDTH=600 HEIGHT=400">
    <p><a href="allTriggers_Freq.png" > Cumulative Triggers Freq </a></p>
    <p><a href="allTriggers_SNR.png" > Cumulative Triggers SNR </a></p>
    <p><a href="triggerGrams/daily_summary.html" > Daily Cumulative Trigger Plots </a></p>

    <h1>Individual Glitch Class Home Pages</h1>
""".format(blurb)
page.write(header + '\n')
for Type in types:
    tmp = """<p><a href="indPages/{0}.html" > {0} </a></p>""".format(Type)
    page.write(tmp + '\n')

header2 = """    <div class="grid">
<div class="grid-sizer"></div>"""

page.write(header2 + '\n')

page= open('{0}/triggerGrams/daily_summary.html'.format(outPath),"w")
for iDay in range(startTime,endTime,86400):
    page.write('       <div class="grid-item">')
    page.write(' <img src="{0}_Freq.png"'.format(iDay))
    page.write(' /></div>\n')
    page.write('       <div class="grid-item">')
    page.write(' <img src="{0}_SNR.png"'.format(iDay))
    page.write(' /></div>\n')

page.write('    </div>\n')
page.write('</body>')
page.close()
