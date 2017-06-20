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
    parser = argparse.ArgumentParser(description="Script to look for other images that are very simular in the database")
    parser.add_argument("--howmany", help="How many closest simularites to display.", type=float, action="store_true", default=5)
    parser.add_argument("--thresh" , help="threshold for what simularites to display.", type=float, action="store_true", default=.9)
    parser.add_argument("--threshUpper", help="Upper bound on similarity (there is no reason to use this)", type=float, action="store_true", default=99)
    parser.add_argument("--ZooID", help="ZooID of the image you want to compare.", action="store_true")
    parser.add_argument("--UniqueID", help="UniqueID of the image you want to compare.", action="store_true")
    args = parser.parse_args();
    #make sure that there arn't too many inputs
    if (args.ZooID and args.UniqueId):
        input_error(2)
    if (~(args.ZooID or args.UniqueId)):
        input_error(3)
    #this return statment is not completely fleshed out yet
    out = {'count': args.howmany, 'thresh': args.thresh, 'tresh-high': args.threshHigh}
    if args.ZooID:
        out['ID'] = args.ZooID
        out['ID-type'] = 'Zoo'
    else:
        out['ID'] = args.UniqueID
        out['ID-type'] = 'Uni'
#temporary for testing other parts of projet
def check():
    return {'count': 7, 'c-type': 'num', 'ID': "memes", 'ID-type': "ZooID"}

#helper of check that displays the error and stops the program
#error 1: too many treshhold inputs
#error 2: too many ID inputs
#error 3: no ID input
def input_error(i):
    #currently i != 1
    #if (i == 1):
    #    print "You can only have one restricting input, you put in two or more"
    if (i == 2):
        print "You can only have one ID input, if you know both, just put one."
    if (i == 3):
        print "You did not give an ID input."
    sys.exit()

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
    for index, row in simularities.nlargest(args['count'], 'two').iterrows():
        if row[1] >= args['tresh'] and row[1] <= args['treshHigh']:
            print row[0], row[1]

#calls main with the inputs given to the function
main(sys.argv[1:])
