from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import time
import logging
import cv2
#use if flownet saves not as .npy
#import flowiz as fz
import glob
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
import tensorflow as tf
import i3d
from lib.label_trans import *

_FRAME_SIZE = 224
_GPU_FRACTION = 0.5
_MIX_WEIGHT_OF_RGB = 0.5
_MIX_WEIGHT_OF_FLOW = 0.5
_LOG_ROOT = 'output'

video_path = './data/example/_ApplyEyeMakeup_g01_c01.avi'
DATA_DIR = 'data/live'

# NOTE: Before running, change the path of checkpoints
_CHECKPOINT_PATHS = {
    'rgb': 'models/ucf101_rgb_0.946_model-44520',
    'flow': 'models/ucf101_flow_0.963_model-28620',
}

_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}

_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}

_CLASS_NUM = {
    'ucf101': 101,
    'hmdb51': 51
}

def main(dataset, mode, variant, workers, inference):
    print('Infering with mode:',mode, end="\r\n") 
    assert mode in ['rgb', 'flow', 'mixed']
    
    log_dir = os.path.join(_LOG_ROOT, 'test-%s-%s-%s' % (dataset, mode,variant))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(level=logging.INFO, filename=os.path.join(
        log_dir, 'log-%s-%s' % (mode, variant)+'.txt'), filemode='w', format='%(message)s')

    label_map = get_label_map(os.path.join(
        './data', dataset, 'label_map.txt'))
    
    fc_out,rgb_fc_out,flow_fc_out,rgb_saver,flow_saver,rgb_holder,flow_holder = build_model(mode,dataset)
    
    # GPU config 
    # Important for using flownet and i3d at the same time
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = _GPU_FRACTION
    sess = tf.Session(config=config)
    
    # start a new session and restore the fine-tuned model
    sess.run(tf.global_variables_initializer())
    
    rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
    flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])

    # Here we start the inference
    print('----Here we start!----', end="\r\n")
    print('Output writes to '+ log_dir, end="\r\n")
  
    start_time = time.time()
    os.system("rm -r "+DATA_DIR+'/*')  
    if not os.path.exists(DATA_DIR+'/img'):
        os.system("mkdir "+DATA_DIR+'/img')
    if not os.path.exists(DATA_DIR+'/flow'):
        os.system("mkdir "+DATA_DIR+'/flow')

    read_stream(variant,fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess, inference, workers)

    print('Infer Videofile in: '+str(time.time() - start_time), end="\r\n")
    sess.close()

def read_stream(variant,fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess, inference_frames, workers):
    pooling_executor = ThreadPoolExecutor(max_workers=workers)
    future_threads = []
    vidcap = cv2.VideoCapture(video_path)
    success,frame = vidcap.read()
    flow_paths = []
    img_paths = []
    img_frames = []
    count = 0
    while success: 
        img_path = DATA_DIR+"/img/frame{:06d}.jpg".format(count)
        cv2.imwrite(img_path,frame)
        img_frames.append(transform_data(frame))
        img_paths.append(img_path)
        imagelist_number = int(count/inference_frames)
        flow_path = DATA_DIR+"/flow/"+str(imagelist_number+1)+"/frame{:06d}.npy".format(count)
        flow_paths.append(flow_path)
        if count != 0 and count%inference_frames == 0:
            print('Calculate Flows for Part '+str(int(imagelist_number))+'...', end="\r\n")
            future_threads.append(pooling_executor.submit(generate_flow_and_predict,variant,DATA_DIR,flow_paths,img_paths,img_frames,str(imagelist_number), fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess))
            print('Added data of Part '+str(int(imagelist_number))+' in pipeline...', end="\r\n")
            img_frames = []
            img_paths = []
            flow_paths = []
            img_paths.append(img_path)
            flow_paths.append(flow_path)
        # use two times to reduce fps
        #success,frame = vidcap.read() 
        success,frame = vidcap.read() 
        count = count + 1
    vidcap.release()
    print('Waiting until all threads finished...', end="\r\n")
    wait(future_threads)
    print('All threads finished!', end="\r\n")
    
def generate_flow_and_predict(variant,DATA_DIR,flow_paths,img_paths,img_frames,imagelist_number,fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess):
    write_flownet_io_files(DATA_DIR,flow_paths,img_paths,imagelist_number)
    flow_frames = generate_flows(DATA_DIR,variant,imagelist_number)
    predict_action(img_frames, flow_frames,imagelist_number, fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess)
    clean_directory(DATA_DIR,imagelist_number,img_paths)
    
def predict_action(img_frames, flow_frames, imagelist_number, fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess):
    rgb_clip = np.array(img_frames)  
    rgb_clip = rgb_clip/255 
    input_rgb = rgb_clip[np.newaxis, :, :, :, :]
     
    flow_clip = np.array(flow_frames)
    flow_clip = 2*(flow_clip/255)-1
    input_flow = flow_clip[np.newaxis, :, :, :, :]
      
    predictions, curr_rgb_fc_data, curr_flow_fc_data = sess.run(
        [fc_out, rgb_fc_out, flow_fc_out],
        feed_dict={rgb_holder: input_rgb, flow_holder: input_flow,})
        
    # np.argmax(predictions[0]),predictions[0, np.argmax(predictions, axis=1)[0]]
    print('Predict Part '+imagelist_number+ ' with Class ' + trans_label(np.argmax(predictions[0]), label_map), end="\r\n")

def build_model(mode,dataset):
    if mode in ['rgb', 'mixed']:
        rgb_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['rgb']])
    if mode in ['flow', 'mixed']:
        flow_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['flow']])

    # insert the model
    if mode in ['rgb', 'mixed']:
        with tf.variable_scope(_SCOPE['rgb']):
            rgb_model = i3d.InceptionI3d(
                400, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                rgb_holder, is_training=False, dropout_keep_prob=1)
            rgb_logits_dropout = tf.nn.dropout(rgb_logits, 1)
            rgb_fc_out = tf.layers.dense(
                rgb_logits_dropout, _CLASS_NUM[dataset], use_bias=True)
                
    if mode in ['flow', 'mixed']:
        with tf.variable_scope(_SCOPE['flow']):
            flow_model = i3d.InceptionI3d(
                400, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
                flow_holder, is_training=False, dropout_keep_prob=1)
            flow_logits_dropout = tf.nn.dropout(flow_logits, 1)
            flow_fc_out = tf.layers.dense(
                flow_logits_dropout, _CLASS_NUM[dataset], use_bias=True)

    # construct two separate feature map and saver(rgb_saver,flow_saver)
    variable_map = {}
    if mode in ['rgb', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['rgb']:
                variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=variable_map)
    variable_map = {}
    if mode in ['flow', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['flow']:
                variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=variable_map, reshape=True)

    # Edited Version by AlexHu
    if mode == 'rgb':
        fc_out = rgb_fc_out
        softmax = tf.nn.softmax(fc_out)
    if mode == 'flow':
        fc_out = flow_fc_out
        softmax = tf.nn.softmax(fc_out)
    if mode == 'mixed':
        fc_out = _MIX_WEIGHT_OF_RGB * rgb_fc_out + _MIX_WEIGHT_OF_FLOW * flow_fc_out
        softmax = tf.nn.softmax(fc_out)
        
    return fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder

def transform_data(data,scale_size=256,crop_size=224,mode='rgb'):
    width = data.shape[0]
    height = data.shape[1]
    if (width==scale_size and height>=width) or (height==scale_size and width>=height):
        return data
    data = cv2.resize(data, (scale_size,scale_size)) 
    #rgb2bgr because rgb_clips are the same...
    if mode=='rgb':
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR) 
    width = data.shape[0]
    height = data.shape[1]
    x0 = int((width-crop_size)/2)
    y0 = int((height-crop_size)/2)
    x1 = x0 + crop_size
    y1 = y0 + crop_size
    return data[x0:x1, y0:y1] 

def write_flownet_io_files(DATA_DIR,flow_paths,img_paths,imagelist_number):
    with open(DATA_DIR+'/output'+imagelist_number+'.txt', 'w') as f:
        for item in flow_paths[:-1]:
            f.write("%s\n" % item)
    with open(DATA_DIR+'/images'+imagelist_number+'_1.txt', 'w') as f:
        for item in img_paths[:-1]:
            f.write("%s\n" % item)
    with open(DATA_DIR+'/images'+imagelist_number+'_2.txt', 'w') as f:
        for item in img_paths[1:]:
            f.write("%s\n" % item)    

def generate_flows(DATA_DIR,variant,imagelist_number):
    image_list_1 = DATA_DIR+'/images'+imagelist_number+'_1.txt'
    image_list_2 = DATA_DIR+'/images'+imagelist_number+'_2.txt'
    output = DATA_DIR+'/output'+imagelist_number+'.txt'
    if not os.path.exists(DATA_DIR+'/flow/'+imagelist_number):
        os.system("mkdir "+DATA_DIR+'/flow/'+imagelist_number)
    os.system("./run-network.sh -n "+variant+" "+image_list_1+" "+image_list_2+" "+output+" >/dev/null")  
    #os.system("./run-network.sh -n "+variant+" "+image_list_1+" "+image_list_2+" "+output)  
    flow_path = DATA_DIR+'/flow/'+imagelist_number+'/*.npy'
    files = glob.glob(flow_path)
    flow_frames = []
    for file in files:
        #floArray = fz.read_flow(file)
        #uv = fz.convert_from_flow(floArray, mode='UV')
        uv = np.load(file)
        flow_frames.append(transform_data(uv,mode='flow'))
    return flow_frames
    
def clean_directory(DATA_DIR,imagelist_number,img_path):
    img_str_delete = ""
    for img in img_path[:-1]:
        img_str_delete = img_str_delete + ' '+img
    os.system("rm "+img_str_delete)
    os.system("rm -r "+DATA_DIR+'/flow/'+imagelist_number)  
    os.system("rm "+DATA_DIR+'/images'+imagelist_number+'_1.txt')
    os.system("rm "+DATA_DIR+'/images'+imagelist_number+'_2.txt')
    os.system("rm "+DATA_DIR+'/output'+imagelist_number+'.txt')

if __name__ == '__main__':
    description = 'Test Finetuned I3D Model'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset', type=str, help="name of dataset, e.g., ucf101")
    p.add_argument('mode', type=str, help="type of data, e.g., rgb")
    p.add_argument('variant', type=str, help="variant of flow calculation, e.g., standard or FlowNet2-s or FlowNet2-ss or FlowNet2-css or FlowNet2-css-ft-sd or FlowNet2-CSS")
    p.add_argument('workers', type=int, help="pool threads for calculating, e.g., 1")
    p.add_argument('inference', type=int, help="inference frames, e.g., 50")
    main(**vars(p.parse_args()))
