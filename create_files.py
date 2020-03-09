import os
from os import listdir
from os.path import isfile, join, isdir
import argparse

#Directory of videos
ORIGIN_DIR = 'data/origin'
#directory of preprocessed files without flownet suffix
DATA_DIR = 'data/preprocessed'
#destination to save files
SAVE_DIR = 'data/'
label_map = 'data/ucf101/label_map.txt'
  
def main(dataset,variant):
    create_train_and_test_files(dataset)
    create_rgb_and_flow_files(dataset,variant)

def create_rgb_and_flow_files(dataset,variant):
    with open(label_map, 'r') as f:
        x = f.readlines()
    label_num = {}
    count = 0
    for label in x:
        label_num[label[:-1]] = str(count)
        count = count + 1
    rgb_file = []
    flow_file = []
    for path, subdirs, files in os.walk(DATA_DIR +"-"+ variant):  
        splitted_path = path.split('/')
        flow_or_rgb = splitted_path[-1]
        dir_name = splitted_path[-2]
        splitted_dir_name = dir_name.split('_')
        num_frames = str(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])-1)
        if flow_or_rgb == 'img':
            abs_path = os.path.dirname(os.path.abspath(path+'/'+files[0]))
            rgb_file.append(dir_name+' '+abs_path+' '+num_frames+' '+label_num[splitted_dir_name[1]])
        elif flow_or_rgb == 'x':
            abs_path = os.path.dirname(os.path.abspath(path))
            flow_file.append(dir_name+' '+abs_path+'/{:s} '+num_frames+' '+label_num[splitted_dir_name[1]])
   
    with open(SAVE_DIR+dataset+'/rgb_'+variant+'.txt', 'w') as f:
        for item in rgb_file:
            f.write("%s\n" % item)
    with open(SAVE_DIR+dataset+'/flow_'+variant+'.txt', 'w') as f:
        for item in flow_file:
            f.write("%s\n" % item)    
            
def create_train_and_test_files(dataset):
    with open(label_map, 'r') as f:
        x = f.readlines()
    label_num = {}
    count = 0
    for label in x:
        label_num[label[:-1]] = str(count)
        count = count + 1
    train = []
    test = []
    for path, subdirs, files in os.walk(ORIGIN_DIR):  
        splitted_path = path.split('/')
        is_train_or_test = splitted_path[-1]
        if is_train_or_test == 'train':
           for file in files:
                splitted_filename_name = file.split('_')
                cleaned_file = '_'.join(splitted_filename_name[:3])
                train.append(splitted_filename_name[1]+'/'+cleaned_file+' '+label_num[splitted_filename_name[1]])
        elif is_train_or_test == 'test':
            for file in files:
                splitted_filename_name = file.split('_')
                cleaned_file = '_'.join(splitted_filename_name[:3])
                test.append(splitted_filename_name[1]+'/'+cleaned_file+'.MP4')
        
    with open(SAVE_DIR+dataset+'/trainlist01.txt', 'w') as f:
        for item in train:
            f.write("%s\n" % item)
    with open(SAVE_DIR+dataset+'/testlist01.txt', 'w') as f:
        for item in test:
            f.write("%s\n" % item)
  
if __name__ == '__main__':
    description = 'Creat lists for Finetuned I3D Model'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset', type=str, help="name of dataset, e.g., ucf101")
    p.add_argument('variant', type=str, help="variant of flow calculation, e.g., standard or FlowNet2-s or FlowNet2-ss or FlowNet2-css or FlowNet2-css-ft-sd or FlowNet2-CSS")
    main(**vars(p.parse_args()))
