import os

epochs = 5
time = 8
n_classes = 2
width,height,color_channels = 210,140,3
number_of_hiddenunits = 32
batch_size = 16

model_name = 'inception'
mode = 'test' #train or test

#config
base_folder = os.path.abspath(os.curdir)
data_path = os.path.join(base_folder,r'C:\Users\allis\OneDrive\Documents\skripsi-alison\all_datasets') 
dataset_path = os.path.join(base_folder,r'C:\Users\allis\OneDrive\Documents\skripsi-alison\all_datasets')

train_folder = os.path.join(dataset_path, 'train')
test_folder = os.path.join(dataset_path, 'test')
valid_folder = os.path.join(dataset_path, 'valid')
model_save_folder = os.path.join(base_folder,'files', model_name,'model_folder')
tensorboard_save_folder = os.path.join(base_folder,'files',model_name,'tensorboard_folder')
checkpoint_path = os.path.join(model_save_folder,"model_weights_{epoch:03d}.ckpt")

#optional for making training dataset
for_train_path = os.path.join(base_folder, '/home/riset/Documents/script-alison/')
train_autopilot = os.path.join(for_train_path, 'data_autopilot')
train_collision = os.path.join(for_train_path, 'data_make_collision')

#optional for making test dataset
for_test_path = os.path.join(base_folder, '/home/riset/Documents/script-alison/datasets/for_test') 
test_autopilot = os.path.join(for_test_path, 'autopilot') 
test_collision = os.path.join(for_test_path, 'collision')

#optional for making valid dataset
for_valid_path = os.path.join(base_folder, '/home/riset/Documents/script-alison/datasets/for_valid')
valid_autopilot = os.path.join(for_valid_path, 'autopilot')
valid_collision = os.path.join(for_valid_path, 'collision')

# #opt for access dataset author
# valid_set = os.path.join('/home/riset/Documents/script-alison/datasets_h5/valid_set')
# test_set = os.path.join('/home/riset/Documents/script-alison/datasets_h5/test_set')
# train_set = os.path.join('/home/riset/Documents/script-alison/datasets_h5/train_set')


# collision_data_path = os.path.join(base_folder, 'data_make_collision')
# autopilot_data_path = os.path.join(base_folder, 'data_autopilot')
