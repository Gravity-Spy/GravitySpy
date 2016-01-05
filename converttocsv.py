#!/usr/bin/env python

# Take images and TAR them as well as changing the image metadata.txt to a CSV file that the API needs to combine images on the Zooniverse side.

# -------------------------------------------------------------------------
#      Setup.
# -------------------------------------------------------------------------

# ---- Import standard modules to the python path.
import csv, os, optparse

def parse_commandline():
    """
    Parse the options given on the command-line.
    Convert image metadata.txt to image metadata CSV. TAR the png to make uploaded easier.
    """
    parser = optparse.OptionParser()
    parser.add_option("-j", "--imagepath", help="path to images")
    parser.add_option("-i", "--iUploadMax", help="Number of subfolders")
    opts, args = parser.parse_args()

    return opts

# =============================================================================
#
#                                    MAIN
#
# =============================================================================

opts = parse_commandline()

# take image metadata and convert to csv

for iUpload in xrange(1,int(opts.iUploadMax)):
        txt_file = opts.imagepath + str(iUpload) + '/metadata.txt'
        csv_file = opts.imagepath + str(iUpload) + '/metadata.csv'
        in_txt = csv.reader(open(txt_file, "rb"), delimiter = ' ')
        out_csv = csv.writer(open(csv_file, 'wb'))
        out_csv.writerows(in_txt)
        system_call = 'tar -czvf ' + opts.imagepath + str(iUpload) + '/L_O1_plots_' + str(iUpload) + '.tar ' + opts.imagepath + str(iUpload) + '/*.png'
        os.system(system_call)
