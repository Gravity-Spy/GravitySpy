import pandas as pd
import glob
import numpy as np
from sqlalchemy.engine import create_engine
import ast
import os
import optparse

from matplotlib import use
use('agg')
from matplotlib import (pyplot as plt, cm)

def parse_commandline():
    """Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--outPath", help="Where you would like the GSpy out pages to live")
    parser.add_option("--pp-matrix", help="pp_matrices path")
    parser.add_option("--confusion-matrix", help="confusion matrix path")
    parser.add_option("--verbose", action="store_true", default=False,help="Run in Verbose Mode")
    opts, args = parser.parse_args()


    return opts

opts = parse_commandline()

pp_matrix_path   = opts.pp_matrix + '/'
outPath          = opts.outPath + '/'
confusion_matrix_path = opts.confusion_matrix + '/'
indPages         = outPath + 'indPages/'

# report status
if not os.path.isdir(indPages):
    os.makedirs(indPages)

# Initialize a list of the Glitch types
types = ["Air_Compressor","Blip","Chirp","Extremely_Loud","Helix","Koi_Fish","Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","None_of_the_Above","No_Glitch","Paired_Doves","Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Wandering_Line","Whistle"]

# Load my sql tables of confusion matrices and pp_matrices
SQL_USER = os.environ['SQL_USER']
SQL_PASS = os.environ['SQL_PASS']
engine = create_engine('mysql://{0}:{1}@localhost/GravitySpy'.format(SQL_USER,SQL_PASS))

# Open classification table and extract most recent classificationID
images             = pd.read_sql('SELECT * FROM images_for_pp',engine)
confusion_matrices = pd.read_sql('SELECT * FROM confusion_matrices',engine)

# convert strings to array and reshape to C by C
images.ML_posterior = images.ML_posterior.apply(ast.literal_eval)
images.choice       = images.choice.apply(ast.literal_eval)

C = len(images[images['type']=='T'].ML_posterior.iloc[0])

def reshape_array(x):
    try:
        x = np.asarray(x)
        x = x.reshape(C,-1)
    except:
        x = False
    return x

confusion_matrices.conf_matrix = confusion_matrices.conf_matrix.apply(ast.literal_eval)
confusion_matrices.conf_matrix = confusion_matrices.conf_matrix.apply(reshape_array)
images.pp_matrix               = images.pp_matrix.apply(ast.literal_eval)
images.pp_matrix               = images.pp_matrix.apply(reshape_array)

# These choices are strings and we need to change them to integers in order to run the crowd sourcing classifer (CC for short). This is the thing that evaluates users and images.

int_to_label = ["AIRCOMP","BLIP","CHIRP","EXTREMELOUD","HELIX","KFISH","LM","LFB","LFL","NOA","NOGTCH","PRDDVS","PLINE","REBLIPS","STTDLGHT","STCHY","TOMTE","VM","WL","WSTLE"]

def plt_conf_matrices(x):
    reducedConf = x.conf_matrix[~np.all(x.conf_matrix == 0,axis=1)]
    reducedConf = reducedConf[:,~np.all(reducedConf == 0,axis=0)]
    conf_divided,a1,a2,a3 = np.linalg.lstsq(np.diag(np.sum(reducedConf,axis=1)),reducedConf)

    fig, ax = plt.subplots()
    cax = ax.matshow(conf_divided, cmap='viridis_r', vmin=0, vmax=1)
    colorbarticks = np.linspace(0,1,6)
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    fig.colorbar(cax,ticks=colorbarticks,cax=cbaxes)

    ax.set_xlabel('predicted class')
    ax.set_ylabel('actual class')
    fig.suptitle('confusion matrix: {0}'.format(x.userID))

    confEntries = np.unique(np.where(x.conf_matrix != 0 ))
    ax.set_xticks(range(confEntries.size))
    ax.set_xticklabels(np.array(int_to_label)[confEntries].tolist())
    ax.set_yticks(range(confEntries.size))
    ax.set_yticklabels(np.array(int_to_label)[confEntries].tolist())
    ax.xaxis.set_ticks_position('bottom')

    fig.savefig('{0}/{1}_confusion.png'.format(confusion_matrix_path,x.userID))
    plt.close(fig)

def plt_pp_matrix(x):
    fig,ax = plt.subplots()

    cax = ax.matshow(x.pp_matrix, cmap='viridis_r',vmin=0, vmax=1)
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    colorbarticks = np.linspace(0,1,6)
    fig.colorbar(cax,ticks=colorbarticks,cax=cbaxes)

    fig.suptitle('pp_matrix of {0}'.format(x.imageID))
    ax.set_xlabel('p(i|j)')
    ax.set_ylabel('classes')
    ax.set_yticks(range(len(x.pp_matrix)))
    ax.set_yticklabels(int_to_label)
    fig.savefig('{0}/{1}_pp_matrix.png'.format(pp_matrix_path,x.imageID))
    plt.close(fig)

images[images['type']=='T'][['imageID','pp_matrix']].apply(plt_pp_matrix,axis=1)
confusion_matrices[['userID','conf_matrix']].apply(plt_conf_matrices,axis=1)

# Make summary page based on combined human and ML classifications
GravitySpySummary = open('{0}/summary.html'.format(outPath),'w')

header = """<head>
    <link rel="stylesheet" type="text/css" href="../../Production/Production/css/classes.css">
</head>
<body>
    <h1>Gravity Spy Summary</h1>
    <h1>Individual Glitch Class Home Pages</h1>
"""
GravitySpySummary.write(header + '\n')
for Type in types:
    tmp = """<p><a href="indPages/{0}.html" > {0} </a></p>""".format(Type)
    GravitySpySummary.write(tmp + '\n')

GravitySpySummary.close()

iN = 0
def tile_images(x):
    # First find path to the image from ML summary page
    pathToImage = glob.glob('/var/www/html/images/Production/Production/GravitySpyO1ImageFinal/**/**/**/{0}.png'.format(x))
    page.write('       <div class="grid-item">')
    page.write(' <img src="../../../{0}"'.format(pathToImage[0].split('images')[1]))
    page.write(' /></div>\n')
    page.write('       <div class="grid-item">')
    page.write(' <img src="../../pp_matrices/{0}_pp_matrix.png"'.format(x))
    page.write(' /></div>\n')


for Type in types:

    # Open a new html page which is named after the type
    page= open('{0}/{1}.html'.format(indPages,Type),"w")

    # Create the same header for each type. This follows the masonry jQuery lay out and therefore you will need the masonry jQuery plug in to work. (http://masonry.desandro.com)

    header = """<head>
    <link rel="stylesheet" type="text/css" href="../../../Production/Production/css/classes.css">
    <script src="../../../Production/Production/node_modules//masonry.pkgd.min.js"></script>
</head>
<body>
    <h1>{0}</h1>
    <div class="grid">
    <div class="grid-sizer"></div>""".format(Type)
    page.write(header + '\n')
    tmp1 = images[images['type']=='T']
    tmp = tmp1[tmp1.true_label == iN]
    tmp.imageID.apply(tile_images)
    page.write('    </div>\n')
    page.write('</body>')
    page.close()
    iN = iN + 1
