tmp = open('uploadImage.dag','w')
iT=0
for iLabel in ["1080Lines","1400Ripples","Air_Compressor","Blip","Chirp","Extremely_Loud","Helix","Koi_Fish","Light_Modulation","Low_Frequency_Burst","Low_Frequency_Lines","No_Glitch","None_of_the_Above","Paired_Doves","Power_Line","Repeating_Blips","Scattered_Light","Scratchy","Tomte","Violin_Mode","Wandering_Line","Whistle"]:
    iT= iT +1
    tmp.write('JOB {0} ./condor/gravityspy.sub\n'.format(iT))
    tmp.write('VARS {0} jobNumber="{0}" Label="{1}"\n'.format(iT,iLabel))
