import os,csv,ast
import optparse
from panoptes_client import *
#Hold
import pdb


def parse_commandline():
    """Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--detector", help="detector you are rewriting the csv file for")
    parser.add_option("--dataPath", help="path to directory that holds the directories of categories")
    opts, args = parser.parse_args()

    return opts

opts = parse_commandline()

dataPath = opts.dataPath

Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')
# Based on the directory you point to determine the classes of glitches.
types = [ name for name in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, name)) ]
types = sorted(types)

B1 = 1610
B2 = 1934
B3 = 1935
A  = 2360
M  = 2117

if opts.detector == 'H1':
    workflow_subject_set_dict_app = {
"Air_Compressor":((A,6714,[1,0.6]),(M,6715,0.6)),
"Blip":((B1,6717,.998),(A,6718,[.998,.85]),(M,6719,.85)),
"Chirp":((B3,6721,1),(A,6722,[1,.5]),(M,6723,.50)),
"Extremely_Loud":((A,6725,[1,.815]),(M,6726,.815)),
"Helix":((A,6728,[1,.50]),(M,6729,.50)),
"Koi_Fish":((B2,6731,.98),(A,6732,[.98,.621]),(M,6733,.621)),
"Light_Modulation":((A,6752,[1,0.9]),(M,6753,0.9)),
"Low_Frequency_Burst":((B3,6755,.99995),(A,6756,[.99995,.93]),(M,6757,.93)),
"Low_Frequency_Lines":((A,6759,[1,.65]),(M,6760,.65)),
"No_Glitch":((B3,6762,.9901),(A,6763,[.9901,.85]),(M,6764,.85)),
"None_of_the_Above":((A,6766,[1,.50]),(M,6767,.50)),
"Paired_Doves":((A,6769,[1,.5]),(M,6770,.50)),
"Power_Line":((B2,6772,.998),(A,6773,[.998,.86]),(M,6774,.86)),
"Repeating_Blips":((A,6776,[1,.686]),(M,6777,.686)),
"Scattered_Light":((B3,6779,.99965),(A,6780,[.99965,.96]),(M,6781,.96)),
"Scratchy":((A,6783,[1,.913]),(M,6784,.913)),
"Tomte":((A,6786,[1,.7]),(M,6787,.7)),
"Violin_Mode":((B2,6902,1),(A,6789,[1,.5]),(M,6790,.50)),
"Wandering_Line":((A,6792,[1,.97]),(M,6793,.97)),
"Whistle":((B1,6795,1),(A,6796,[1,.6]),(M,6797,.6)),
    }
elif opts.detector == 'L1':
    workflow_subject_set_dict_app = {
"Air_Compressor":((A,6714,[1,0.6]),(M,6715,0.6)),
"Blip":((B1,6717,.998),(A,6718,[.998,.85]),(M,6719,.85)),
"Chirp":((B3,6721,1),(A,6722,[1,.5]),(M,6723,.50)),
"Extremely_Loud":((A,6725,[1,.815]),(M,6726,.815)),
"Helix":((A,6728,[1,.50]),(M,6729,.50)),
"Koi_Fish":((B2,6731,.98),(A,6732,[.98,.621]),(M,6733,.621)),
"Light_Modulation":((A,6752,[1,0.9]),(M,6753,0.9)),
"Low_Frequency_Burst":((B3,6755,.99995),(A,6756,[.99995,.93]),(M,6757,.93)),
"Low_Frequency_Lines":((A,6759,[1,.65]),(M,6760,.65)),
"No_Glitch":((B3,6762,.9901),(A,6763,[.9901,.85]),(M,6764,.85)),
"None_of_the_Above":((A,6766,[1,.50]),(M,6767,.50)),
"Paired_Doves":((A,6769,[1,.5]),(M,6770,.50)),
"Power_Line":((B2,6772,.998),(A,6773,[.998,.86]),(M,6774,.86)),
"Repeating_Blips":((A,6776,[1,.686]),(M,6777,.686)),
"Scattered_Light":((B3,6779,.99965),(A,6780,[.99965,.96]),(M,6781,.96)),
"Scratchy":((A,6783,[1,.913]),(M,6784,.913)),
"Tomte":((A,6786,[1,.7]),(M,6787,.7)),
"Violin_Mode":((B2,6902,1),(A,6789,[1,.5]),(M,6790,.50)),
"Wandering_Line":((A,6792,[1,.97]),(M,6793,.97)),
"Whistle":((B1,6795,1),(A,6796,[1,.6]),(M,6797,.6)),
    }
else:
    ValueError("Please select detector you are rewriting the csv file for")

header = ['date','subject_id','Filename1','Filename2','Filename3','Filename4','#ML_Posterior']

indexDict = {"Air_Compressor":0,"Blip":1,"Chirp":2,"Extremely_Loud":3,"Helix":4,"Koi_Fish":5,"Light_Modulation":6,"Low_Frequency_Burst":7,"Low_Frequency_Lines":8,"None_of_the_Above":9,"No_Glitch":10,"Paired_Doves":11,"Power_Line":12,"Repeating_Blips":13,"Scattered_Light":14,"Scratchy":15,"Tomte":16,"Violin_Mode":17,"Wandering_Line":18,"Whistle":19}

tmptmptmp =[]

for Type in types:
    Begin  = []
    Appre  = []
    Master = []
    try:
        work_subject_dict = workflow_subject_set_dict_app[Type]
        iN                = indexDict[Type]
        reader = csv.reader(open('{0}/{1}/Beginner/imagemeta.csv'.format(dataPath,Type)), delimiter=",")
        hold = sorted(reader)
        for iX in hold:
            if not len(ast.literal_eval(iX[6])) == 20:
                print(ast.literal_eval(iX[6]))
                ValueError('Problem with CSV file for type: {0}'.format(Type))
    except:
        print("Fail: {0}".format(Type))


for Type in types:
    Begin  = []
    Appre  = []
    Master = []
    try:
        work_subject_dict = workflow_subject_set_dict_app[Type]
        iN                = indexDict[Type]
        reader = csv.reader(open('{0}/{1}/Beginner/imagemeta.csv'.format(dataPath,Type)), delimiter=",")
        hold = sorted(reader)

        for iX in hold:

            for combos in work_subject_dict:
                workflow   = combos[0]
                thres      = combos[2]

                if (workflow in [B1,B2,B3]) and (float(ast.literal_eval(iX[6])[iN]) > thres):
                    Begin.append(iX)

                if (workflow in [A]) and (thres[0] >= float(ast.literal_eval(iX[6])[iN])) and (float(ast.literal_eval(iX[6])[iN]) >= thres[1]):
                    Appre.append(iX)

                if (workflow in [M]) and (thres > float(ast.literal_eval(iX[6])[iN])):
                    Master.append(iX)

        for combos in work_subject_dict:
            workflow   = combos[0]
            subjectset = SubjectSet.find(combos[1])
            tmp        = []
            if (workflow in [B1,B2,B3]):
                for iXX in Begin:
                    subject = Subject()
                    subject.links.project = project
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[2]))
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[3]))
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[4]))
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[5]))
                    subject.metadata['date']          = iXX[0]
                    subject.metadata['subject_id']    = iXX[1]
                    subject.metadata['Filename1']     = iXX[2]
                    subject.metadata['Filename2']     = iXX[3]
                    subject.metadata['Filename3']     = iXX[4]
                    subject.metadata['Filename4']     = iXX[5]
                    subject.metadata['#ML_Posterior'] = iXX[6]
                    subject.save()
                    tmp.append(subject)
                subjectset.add(tmp)

            if (workflow in [A]):
                for iXX in Appre:
                    subject = Subject()
                    subject.links.project = project
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[2]))
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[3]))
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[4]))
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[5]))
                    subject.metadata['date']          = iXX[0]
                    subject.metadata['subject_id']    = iXX[1]
                    subject.metadata['Filename1']     = iXX[2]
                    subject.metadata['Filename2']     = iXX[3]
                    subject.metadata['Filename3']     = iXX[4]
                    subject.metadata['Filename4']     = iXX[5]
                    subject.metadata['#ML_Posterior'] = iXX[6]
                    subject.save()
                    tmp.append(subject)
                subjectset.add(tmp)

            if (workflow in [M]):
                for iXX in Master:
                    subject = Subject()
                    subject.links.project = project
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[2]))
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[3]))
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[4]))
                    subject.add_location('{0}/{1}/Beginner/{2}'.format(dataPath,Type,iXX[5]))
                    subject.metadata['date']          = iXX[0]
                    subject.metadata['subject_id']    = iXX[1]
                    subject.metadata['Filename1']     = iXX[2]
                    subject.metadata['Filename2']     = iXX[3]
                    subject.metadata['Filename3']     = iXX[4]
                    subject.metadata['Filename4']     = iXX[5]
                    subject.metadata['#ML_Posterior'] = iXX[6]
                    subject.save()
                    tmp.append(subject)
                subjectset.add(tmp)
    except:
        print("Fail: {0}".format(Type))
