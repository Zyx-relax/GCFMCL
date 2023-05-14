import numpy as np
import pandas as pd
adj = pd.read_csv("./D2R.txt",sep='\t')
left_dict={}
right_dict={}
for _,row in adj.iterrows():
    if row[0] not in left_dict:
        left_dict[row[0]]=len(left_dict)
    if row[1] not in right_dict:
        right_dict[row[1]]=len(right_dict)

A=np.zeros((len(left_dict),len(right_dict)))
print(1)
for _,row in adj.iterrows():
    left=left_dict[row[0]]
    right=right_dict[row[1]]
    A[left,right]=1
print(1)
np.savetxt(fname='interaction.txt', X=A, newline='\n', encoding='UTF-8')