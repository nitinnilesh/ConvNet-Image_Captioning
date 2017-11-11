# data_preprocessing_img_capt_Flicker8K
Preprocessing the data for effective use in image captioning algorithm
The script 'saving_numpy_train.py' saves the training images to a numpy array. First entry of the created numpy array will be an empty array and the other entries will be as: [name_of_image][[R_cha][G_cha][B_cha]]. So for a set of 1000 images, the created file will have 1001 entries with first entry being empty.
