import numpy as np
import os
import sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
curr_dir=os.getcwd();
from PIL import Image as PImage
os.chdir('/home/ashish/Documents/SMAI_validate_data')
print(os.getcwd())
file_out=[[]]

for ind,i in enumerate(os.listdir(os.getcwd())):
    img_id=np.str.partition(i,'.');
    
    temp_id=img_id[0];
    temp_img=mpimg.imread(os.getcwd()+'/'+i);
    arr=[temp_id,temp_img]
    file_out.append(arr);
os.chdir('/home/ashish/Documents/SMAI_out_files')   
np.save('Validation_Data_array',file_out)


