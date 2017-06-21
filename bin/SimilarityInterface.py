#notes: many functions are incomplete or disabled because sql isn't implemented yet.

import numpy as np #this does math
import pandas as pd #this helps with the sorting and the math associated
import argparse #this helps parse the arguments to the call
import sys #this is needed to deal with inputs

#sql database to look at
SQLCommand = ''

#main function
def main(args):
    args = check()
    simularites = sqlCall(args)  # the table of simularity mesurements
    sortedSims = sort(simularites)  # the sorted table of simularity mesurements (top = highest)
    output(sortedSims, args)

# inputs: arguments
# output: arguments in a better format
# checks to see if the arguement inputs are correct, if not, program ends
#temporarly changed because it crashes
def null_check():
    #you cannot split the parser into required and unrequired groups because there is no "required input"... if you input --howmany and --ZooID it is valid and if you input --UniqueID and --thresh it is valid
    parser = argparse.ArgumentParser()
    parser.add_argument("--howmany", help="How many closest simularites to display.", type=float)
    parser.add_argument("--thresh" , help="threshold for what simularites to display.", type=float)
    parser.add_argument("--threshArray", help="Array in which to display simularites from. (unclear) (DONT USE THIS UNLESS YOU KNOW WHAT YOU ARE DOING)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ZooID", help="ZooID of the image you want to compare.")
    group.add_argument("--UniqueID", help="UniqueID of the image you want to compare.")
    args = parser.parse_args()
    #make sure that there arn't too many inputs
    #this return statment is not completely fleshed out yet
    return 0;
#temporary for testing other parts of projet
def check():
    return {'count': 7, 'c-type': 'num', 'ID': "memes", 'ID-type': "ZooID"}

# inputs: arguments
# output: a pandas list of the simularity mesurements of inputed ID
def sqlCall(args):
    #Temporaryly commented out for non-sql testing
    #return pd.read_sql(SQLCommand)
    s = {'one' : pd.Series([3, 2, 1, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e', 'f']), 'two' : pd.Series([1, 2, 3, 4, 5, 6 ,7], index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])}
    return pd.DataFrame(s)

# inputs: a simularity mesurements array
# outputs: a sorted simularity mesurements array
def sort(simularities):
    return simularities.sort_values(by="two")

# inputs: a sorted simularity mesurements array and the arguments list
# outputs: nil
# prints out all the simular images information, calls the image downloading if active
def output(simularities, args):
    #case where user choses how many to print off of top
    count = 0;
    for index, row in simularities.iterrows():
        if row[1] >= args['thresh'] and row[1] <= args['threshHigh'] and count < args['count']:
            print row[0], row[1]
            count += 1

#calls main with the inputs given to the function
main(sys.argv[1:])
