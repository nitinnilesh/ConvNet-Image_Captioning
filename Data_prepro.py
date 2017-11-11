import numpy as np
import os
import sys
import shutil
import pandas as pd
curr_dir=os.getcwd();
print(curr_dir)

#test_ids=np.genfromtxt('/home/ashish/Downloads/Files by Kshitij/test.csv',delimiter=',')
#train_ids=
os.chdir('/home/ashish/Documents/Flickr8k_Dataset/Flicker8k_Dataset/')
new_dir='/home/ashish/Documents/SMAI_TestData';
test_ids=pd.read_csv('/home/ashish/Downloads/Files by Kshitij/test.csv',header=None);
file_names=os.listdir('/home/ashish/Documents/Flickr8k_Dataset/Flicker8k_Dataset/');
for i in test_ids.iterrows():
    temp_name=i
    current_file=os.getcwd()+'/'+i[1].iloc[0]
    shutil.copy(current_file,new_dir)




    
