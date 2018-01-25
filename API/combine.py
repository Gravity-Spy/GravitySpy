import pandas as pd
import os, sys
import argparse
import pdb

argp = argparse.ArgumentParser()
argp.add_argument("-d", "--directory", type=str, help="Specify the directory where the multiprocessing file output are")
args = argp.parse_args()

# combined retired subjects and image_db pickles

files = os.listdir(args.directory)
ret_subjects_list = []
image_db_list = []
for names in files:
    if names.startswith("ret_subjects"):
        ret_subjects_list.append(names)
    elif names.startswith("image_db"):
        image_db_list.append(names)
ret_subjects_list.sort()
image_db_list.sort()

ret_subjects = pd.DataFrame()
for f in ret_subjects_list:
    temp = pd.read_pickle(args.directory+f)
    ret_subjects = ret_subjects.append(temp)

image_db = pd.DataFrame()
for f in image_db_list:
    temp = pd.read_pickle(args.directory+f)
    image_db = image_db.append(temp)

ret_subjects.to_pickle(args.directory+'ret_subjects.pkl')
image_db.to_pickle(args.directory+'image_db.pkl')

