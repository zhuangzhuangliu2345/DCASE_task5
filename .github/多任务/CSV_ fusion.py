import pandas as pd
import math


#for i in range(csv_number):
#    df1 = pd.read_csv('best{}_submission.csv'.format(i))
#    df2 = pd.read_csv('best2_submission.csv')

df1 = pd.read_csv('tijiao/best1_submission.csv')
df2 = pd.read_csv('tijiao/best2_submission.csv')
df3 = pd.read_csv('tijiao/best3_submission.csv')
df4 = pd.read_csv('tijiao/best4_submission.csv')
df5 = pd.read_csv('tijiao/best5_submission.csv')
df6 = pd.read_csv('tijiao/best6_submission.csv')
df7 = pd.read_csv('tijiao/best7_submission.csv')
df8 = pd.read_csv('tijiao/best8_submission.csv')
df9 = pd.read_csv('tijiao/best9_submission.csv')
df10 = pd.read_csv('tijiao/best10_submission.csv')
df11 = pd.read_csv('tijiao/best11_submission.csv')
audio_filename_list = df1['audio_filename'].to_list()
#submission_path = 'tijiao/end1'
#f = open(submission_path, 'w')


labels = ['1-1_small-sounding-engine', '1-2_medium-sounding-engine',
    '1-3_large-sounding-engine', '1-X_engine-of-uncertain-size',
    '2-1_rock-drill', '2-2_jackhammer', '2-3_hoe-ram', '2-4_pile-driver',
    '2-X_other-unknown-impact-machinery', '3-1_non-machinery-impact',
    '4-1_chainsaw', '4-2_small-medium-rotating-saw', '4-3_large-rotating-saw',
    '4-X_other-unknown-powered-saw', '5-1_car-horn', '5-2_car-alarm',
    '5-3_siren', '5-4_reverse-beeper', '5-X_other-unknown-alert-signal',
    '6-1_stationary-music', '6-2_mobile-music', '6-3_ice-cream-truck',
    '6-X_music-from-uncertain-source', '7-1_person-or-small-group-talking',
    '7-2_person-or-small-group-shouting', '7-3_large-crowd',
    '7-4_amplified-speech', '7-X_other-unknown-human-voice',
    '8-1_dog-barking-whining', '1_engine', '2_machinery-impact', '3_non-machinery-impact',
    '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog']

print(len(labels))
d = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
for i in range(len(labels)):
    list1 = df1[labels[i]].to_list()
    list2 = df2[labels[i]].to_list()
    list3 = df3[labels[i]].to_list()
    list4 = df4[labels[i]].to_list()
    list5 = df5[labels[i]].to_list()
    list6 = df6[labels[i]].to_list()
    list7 = df7[labels[i]].to_list()
    list8 = df8[labels[i]].to_list()
    list9 = df9[labels[i]].to_list()
    list10 = df10[labels[i]].to_list()
    list11 = df11[labels[i]].to_list()
    for j in range(len(audio_filename_list)):
        x = 0.25*math.log(list2[j]) + 0.25*math.log(list6[j]) + 0.25*math.log(list10[j]) + 0.25*math.log(list11[j])
        y = math.exp(x)
        d[i].append(y)

dataframe = pd.DataFrame({'audio_filename': audio_filename_list, labels[0]: d[0], labels[1]: d[1], labels[2]: d[2],
                          labels[3]: d[3], labels[4]: d[4], labels[5]: d[5], labels[6]: d[6], labels[7]: d[7],
                          labels[8]: d[8], labels[9]: d[9], labels[10]: d[10], labels[11]: d[11], labels[12]: d[12],
                          labels[13]: d[13], labels[14]: d[14], labels[15]: d[15], labels[16]: d[16], labels[17]: d[17],
                          labels[18]: d[18], labels[19]: d[19], labels[20]: d[20], labels[21]: d[21], labels[22]: d[22],
                          labels[23]: d[23], labels[24]: d[24], labels[25]: d[25], labels[26]: d[26], labels[27]: d[27],
                          labels[28]: d[28], labels[29]: d[29], labels[30]: d[30], labels[31]: d[31], labels[32]: d[32],
                          labels[33]: d[33], labels[34]: d[34], labels[35]: d[35], labels[36]: d[36]})


dataframe.to_csv("Liu_BUPT_task5_4.output.csv", index=False, sep=',')

