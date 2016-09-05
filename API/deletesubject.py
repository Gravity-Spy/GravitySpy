from panoptes_client import *
import pdb

# Connect to panoptes and query all classifications done on project 1104 (i.e. GravitySpy)
Panoptes.connect()
project = Project.find(slug='zooniverse/gravity-spy')

B1 = 1610
B2 = 1934
B3 = 1935
B4 = 1936
A  = 2360
M  = 2117

workflow_subject_set_dict_app = {
"Light_Modulation":((A,5653),(M,5654)),
"Air_Compressor":((A,5651),(M,5652)),
"Blip":((B1,5655),(A,5656),(M,5657)),
"Chirp":((B3,5658),(A,5659),(M,5660)),
"Extremely_Loud":((A,5661),(M,5662)),
"Helix":((A,5663),(M,5664)),\
"Koi_Fish":((B2,5665),(A,5666),(M,5667)),
"Low_Frequency_Burst":((A,5668),(M,5669)),
"Low_Frequency_Lines":((A,5670),(M,5671)),
"No_Glitch":((B4,5672),(A,5673),(M,5674)),
"None_of_the_Above":((B4,5675),(A,5676),(M,5677)),
"Paired_Doves":((A,5678),(M,5679)),
"Power_Line":((B2,5682),(A,5680),(M,5681)),
"Repeating_Blips":((A,5683),(M,5684)),
"Scattered_Light":((B3,5685),(A,5686),(M,5687)),
"Scratchy":((A,5688),(M,5689)),
"Tomte":((A,5690),(M,5691)),
"Violin_Mode":((A,5692),(M,5693)),
"Wandering_Line":((A,5694),(M,5695)),
"Whistle":((B1,5696),(A,5697),(M,5698)),
}
types = ['Blip','Koi_Fish','Power_Line','No_Glitch','None_of_the_Above','Scattered_Light','Whistle']
types = ['Scattered_Light']
tmp = []

for Type in types:
    work_subject_dict = workflow_subject_set_dict_app[Type]
    for combos in work_subject_dict:
        workflow = combos[0]
        subjectset = combos[1]
        if workflow in [B1,B2,B3,B4]:
            SubSet = SubjectSet.find(subjectset)
            subs = SubSet.subjects()
            while True:
                try:
                    tmp.append(subs.next())
                except:
                    break
            print(tmp)
