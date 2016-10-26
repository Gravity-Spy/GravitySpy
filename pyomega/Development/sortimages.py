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
"Light_Modulation":((A,5653,[1,0.9]),(M,5654,0.9)),
"Air_Compressor":((A,5651,[1,0.6]),(M,5652,0.6)),
"Blip":((B1,5655,.998),(A,5656,[.998,.85]),(M,5657,.85)),
"Chirp":((B3,5658,1),(A,5659,[1,.5]),(M,5660,.50)),
"Extremely_Loud":((A,5661,[1,.815]),(M,5662,.815)),
"Helix":((A,5663,[1,.50]),(M,5664,.50)),
"Koi_Fish":((B2,5665,.98),(A,5666,[.98,.621]),(M,5667,.621)),
"Low_Frequency_Burst":((B3,5905,.99995),(A,5668,[.99995,.93]),(M,5669,.93)),
"Low_Frequency_Lines":((A,5670,[1,.65]),(M,5671,.65)),
"No_Glitch":((B3,5672,.9901),(A,5673,[.9901,.85]),(M,5674,.85)),
"None_of_the_Above":((A,5676,[1,.50]),(M,5677,.50)),
"Paired_Doves":((A,5678,[1,.5]),(M,5679,.50)),
"Power_Line":((B2,5682,.998),(A,5680,[.998,.86]),(M,5681,.86)),
"Repeating_Blips":((A,5683,[1,.686]),(M,5684,.686)),
"Scattered_Light":((B3,5685,.99965),(A,5686,[.99965,.96]),(M,5687,.96)),
"Scratchy":((A,5688,[1,.913]),(M,5689,.913)),
"Tomte":((A,5690,[1,.7]),(M,5691,.7)),
"Violin_Mode":((B2,5904,1),(A,5692,[1,.5]),(M,5693,.50)),
"Wandering_Line":((A,5694,[1,.97]),(M,5695,.97)),
"Whistle":((B1,5696,1),(A,5697,[1,.6]),(M,5698,.6)),
    }
elif opts.detector == 'L1':
    workflow_subject_set_dict_app = {
"Light_Modulation":((A,5653,[1,0.9]),(M,5654,0.9)),
"Air_Compressor":((A,5651,[1,0.6]),(M,5652,0.6)),
"Blip":((B1,5655,.998),(A,5656,[.998,.85]),(M,5657,.85)),
"Chirp":((B3,5658,1),(A,5659,[1,.5]),(M,5660,.50)),
"Extremely_Loud":((A,5661,[1,.815]),(M,5662,.815)),
"Helix":((A,5663,[1,.50]),(M,5664,.50)),
"Koi_Fish":((B2,5665,.98),(A,5666,[.98,.621]),(M,5667,.621)),
"Low_Frequency_Burst":((B3,5905,.99995),(A,5668,[.99995,.93]),(M,5669,.93)),
"Low_Frequency_Lines":((A,5670,[1,.65]),(M,5671,.65)),
"No_Glitch":((B3,5672,.9901),(A,5673,[.9901,.85]),(M,5674,.85)),
"None_of_the_Above":((A,5676,[1,.50]),(M,5677,.50)),
"Paired_Doves":((A,5678,[1,.5]),(M,5679,.50)),
"Power_Line":((B2,5682,.998),(A,5680,[.998,.86]),(M,5681,.86)),
"Repeating_Blips":((A,5683,[1,.686]),(M,5684,.686)),
"Scattered_Light":((B3,5685,.99965),(A,5686,[.99965,.96]),(M,5687,.96)),
"Scratchy":((A,5688,[1,.913]),(M,5689,.913)),
"Tomte":((A,5690,[1,.7]),(M,5691,.7)),
"Violin_Mode":((B2,5904,1),(A,5692,[1,.5]),(M,5693,.50)),
"Wandering_Line":((A,5694,[1,.97]),(M,5695,.97)),
"Whistle":((B1,5696,1),(A,5697,[1,.6]),(M,5698,.6)),
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
                subjectset = SubjectSet.find(6584) #6584 #combos[1]
                tmp        = []
                if (workflow in [B1,B2,B3]):
                    for iXX in Begin:
                        subject = Subject()
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

