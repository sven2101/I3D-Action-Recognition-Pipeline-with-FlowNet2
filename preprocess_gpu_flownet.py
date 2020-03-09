from __future__ import print_function

import os
import sys
import cv2
import numpy as np
import argparse
#use if flownet saves not as .npy
#import flowiz as fz
import glob
import time
from multiprocessing import Pool
import os
from os import listdir
from os.path import isfile, join, isdir

#Set relative path for source and destination in flownet directory
#it will create the same directory structure in destination as in source e.g. train and test directory
DATA_DIR = 'data/origin'
SAVE_DIR = 'data/preprocessed'

_EXT = ['.avi', '.MP4', '.mp4']

def get_video_length(video_path):
  _, ext = os.path.splitext(video_path)
  if not ext in _EXT:
    raise ValueError('Extension "%s" not supported' % ext)
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened(): 
    raise ValueError("Could not open the file.\n{}".format(video_path))
  if cv2.__version__ >= '3.0.0':
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
  else:
    CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
  length = int(cap.get(CAP_PROP_FRAME_COUNT))
  cap.release()
  return length

def compute_rgb(args):
    """Compute RGB"""
    video_path,write_to = args
    write_paths_arr = []
    write_output_flo_arr = []
    print('Do RGB for: '+ video_path +' '+ write_to, end='\r\n')
    vidcap = cv2.VideoCapture(video_path)
    success,frame = vidcap.read()
    count = 0
    while success:
        img_path = write_to+"/img/frame{:06d}.jpg".format(count)
        if count == 0:
            write_paths_arr.append(img_path)
        cv2.imwrite(img_path,frame)
        write_paths_arr.append(img_path)
        write_output_flo_arr.append(write_to+"/flow/frame{:06d}.npy".format(count))
        success,frame = vidcap.read() 
        count = count + 1
    vidcap.release()
    
    with open(write_to+'/output.txt', 'w') as f:
        for item in write_output_flo_arr:
            f.write("%s\n" % item)
    with open(write_to+'/images1.txt', 'w') as f:
        for item in write_paths_arr[:-1]:
            f.write("%s\n" % item)
    with open(write_to+'/images2.txt', 'w') as f:
        for item in write_paths_arr[1:]:
            f.write("%s\n" % item)

def compute_flow(args):
  """Compute optical flow with FlowNet2."""
  video_path, write_to, variant = args
  print('Do flow for: '+ video_path +' '+ write_to, end='\r\n')
  image_list_1 = write_to+'/images1.txt'
  image_list_2 = write_to+'/images2.txt'
  output = write_to+'/output.txt'
  os.system("./run-network.sh -n "+variant+" "+image_list_1+" "+image_list_2+" "+output+" >/dev/null")  
  #os.system("./run-network.sh -n "+variant+" "+image_list_1+" "+image_list_2+" "+output)  
  flow_path = write_to+'/flow/*.npy'
  files = glob.glob(flow_path)
  count = 0
  for file in files:
    #floArray = fz.read_flow(file)
    #uv = fz.convert_from_flow(floArray, mode='UV')
    uv = np.load(file)
    cv2.imwrite(write_to+'/x/frame{:06d}.jpg'.format(count), uv[...,0])
    cv2.imwrite(write_to+'/y/frame{:06d}.jpg'.format(count), uv[...,1])
    count = count + 1
  os.system("rm -r "+write_to+"/flow/")  
 
def main(mode='rgb',variant='FlowNet2',pool_threads=1):
    global SAVE_DIR
    SAVE_DIR = SAVE_DIR +"-"+ variant

    directories = [d for d in listdir(DATA_DIR) if isdir(join(DATA_DIR, d))]
    source = []
    destination = []
    flownet_conf = []
    for directory in directories:
        files = [f for f in listdir(DATA_DIR+'/'+directory) if isfile(join(DATA_DIR+'/'+directory, f))]
        for file in files:
            origin = DATA_DIR+'/'+directory+'/'+file
            file_split = file.split("_")
            file_name = file_split[0]+'_'+file_split[1]+'_'+file_split[2]
            if file_name.lower().endswith('.mp4') or file_name.lower().endswith('.avi'):
                file_name = file_name[:-4]
            file_dir = SAVE_DIR+'/'+directory+'/'+file_name
            if os.path.exists(file_dir+'/img')==False:
                os.makedirs(file_dir+'/img')
            if os.path.exists(file_dir+'/x')==False:
                os.makedirs(file_dir+'/x')
            if os.path.exists(file_dir+'/y')==False:
                os.makedirs(file_dir+'/y')
            if os.path.exists(file_dir+'/flow')==False:
                os.makedirs(file_dir+'/flow')
            source.append(origin)
            destination.append(file_dir)
            flownet_conf.append(variant)
    pool = Pool(pool_threads)    
    if mode == 'rgb' or mode == 'rgbflow':
        start_time = time.time()
        pool.map(compute_rgb, zip(source, destination))
        print('Compute RGB in sec: '+str(time.time() - start_time)+'...', end="\r\n")
        #print('Compute RGB in sec: '+str(time.time() - start_time), end='\r\n')
    if mode == 'flow' or mode == 'rgbflow':
        start_time = time.time()
        pool.map(compute_flow, zip(source,destination,flownet_conf))
        print('Compute flow in sec: '+str(time.time() - start_time), end='\r\n')
    
if __name__ == '__main__':
    description = 'Generate RGBs and Optical flow from videos with flownet2'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('mode', type=str, help="type of data, e.g., rgb or flow or rgbflow")
    p.add_argument('variant', type=str, help="variant of flow calculation, e.g., standard or FlowNet2-s or FlowNet2-ss or FlowNet2-css or FlowNet2-css-ft-sd or FlowNet2-CSS")
    p.add_argument('pool_threads', type=int, help="pool threads for calculating, e.g., 1")
    main(**vars(p.parse_args()))

