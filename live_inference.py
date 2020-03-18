from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import time
import logging
import cv2
import copy
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

#Run Video Examples
video_path = './data/example/_Archery_g01_c01.avi'
#video_path = './data/example/_BoxingPunchingBag_g01_c01.avi' 
#video_path = './data/example/_JugglingBalls_g01_c02.avi' 

#Run camera 
#video_path = 0

DATA_DIR = 'data/live'
prediction_print=['none',0,0]

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
    
    prob_predictions, fc_out,rgb_fc_out,flow_fc_out,rgb_saver,flow_saver,rgb_holder,flow_holder = build_model(mode,dataset)
    
    # GPU config 
    # Important for using flownet and i3d at the same time
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = _GPU_FRACTION
    sess = tf.Session(config=config)
    
    # start a new session and restore the fine-tuned model
    sess.run(tf.global_variables_initializer())
    if mode in ['rgb', 'mixed']:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
    if mode in ['flow', 'mixed']:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])

    # Here we start the inference
    print('----Here we start!----', end="\r\n")
    print('Mode is set to:',mode, end="\r\n") 
    print('Dataset is set to:',dataset, end="\r\n") 
    print('FlowNet2 variant is set to:',variant, end="\r\n") 
    print('Inference frame is set to:',inference, end="\r\n") 
    print('Workers is set to:',workers, end="\r\n") 
    print('Output writes to '+ log_dir, end="\r\n")
  
    start_time = time.time()
    os.system("rm -r "+DATA_DIR+'/*')  
    if not os.path.exists(DATA_DIR+'/img'):
        os.system("mkdir "+DATA_DIR+'/img')
    if not os.path.exists(DATA_DIR+'/flow'):
        os.system("mkdir "+DATA_DIR+'/flow')

    read_stream(variant,mode,prob_predictions,fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess, inference, workers)

    print('Infer Input in: '+str(time.time() - start_time), end="\r\n")
    sess.close()

def read_stream(variant,mode, prob_predictions, fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess, inference_frames, workers):
    pooling_executor = ThreadPoolExecutor(max_workers=workers)
    future_threads = []
    cv2.namedWindow("Preview")
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    success,frame = vidcap.read()
    flow_paths = []
    img_paths = []
    img_frames = []
    count = 0
    while success: 
        frame2=copy.deepcopy(frame)
        cv2.putText(frame2, prediction_print[0] +" "+str(prediction_print[2]), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),2)
        cv2.imshow("Preview", frame2)
        # without it will break cv2.imshow - dont know why!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_path = DATA_DIR+"/img/frame{:06d}.jpg".format(count)
        cv2.imwrite(img_path,frame)
        img_frames.append(transform_data(frame))
        img_paths.append(img_path)
        imagelist_number = int(count/inference_frames)
        flow_path = DATA_DIR+"/flow/"+str(imagelist_number+1)+"/frame{:06d}.npy".format(count)
        flow_paths.append(flow_path)
        if count != 0 and count%inference_frames == 0:
            if mode in ['flow', 'mixed']:
                print('Calculate Flows for Part '+str(int(imagelist_number))+'...', end="\r\n")
            future_threads.append(pooling_executor.submit(generate_flow_and_predict,variant,mode,DATA_DIR,flow_paths,img_paths,img_frames,str(imagelist_number),prob_predictions, fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess))
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

def generate_flow_and_predict(variant,mode,DATA_DIR,flow_paths,img_paths,img_frames,imagelist_number,prob_predictions,fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess):
    flow_frames = []
    if mode in ['flow', 'mixed']:
        write_flownet_io_files(DATA_DIR,flow_paths,img_paths,imagelist_number)
        flow_frames = generate_flows(DATA_DIR,variant,imagelist_number)
    predict_action(mode,img_frames, flow_frames,imagelist_number,prob_predictions, fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess)
    clean_directory(DATA_DIR,mode,imagelist_number,img_paths)
    
def predict_action(mode,img_frames, flow_frames, imagelist_number,prob_predictions, fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder, label_map, sess):
    if mode in ['rgb', 'mixed']:
        rgb_clip = np.array(img_frames)  
        rgb_clip = rgb_clip/255 
        input_rgb = rgb_clip[np.newaxis, :, :, :, :]
     
    if mode in ['flow', 'mixed']:
        flow_clip = np.array(flow_frames)
        flow_clip = 2*(flow_clip/255)-1
        input_flow = flow_clip[np.newaxis, :, :, :, :]

    if mode in ['rgb']:  
        out_prob_predictions ,logit_predictions, curr_rgb_fc_data,  = sess.run(
            [prob_predictions,fc_out, rgb_fc_out],
            feed_dict={rgb_holder: input_rgb,})
    elif mode in ['flow']:  
        out_prob_predictions ,logit_predictions, curr_flow_fc_data,  = sess.run(
            [prob_predictions,fc_out, flow_fc_out],
            feed_dict={flow_holder: input_flow,})
    elif mode in ['mixed']:  
        out_prob_predictions ,logit_predictions, curr_rgb_fc_data, curr_flow_fc_data = sess.run(
            [prob_predictions,fc_out, rgb_fc_out, flow_fc_out],
            feed_dict={rgb_holder: input_rgb, flow_holder: input_flow,})
        
    print('Predict Part '+imagelist_number+ ' with Class ' + trans_label(np.argmax(out_prob_predictions[0]), label_map) + ' with probability ' + str(max(out_prob_predictions[0])), end="\r\n")
    update_predictions_inframe(imagelist_number,trans_label(np.argmax(out_prob_predictions[0]), label_map),max(out_prob_predictions[0]))

def update_predictions_inframe(imagelist_number,prediction,confidence):
    global prediction_print
    if imagelist_number > prediction_print[1]:
        prediction_print[0] = prediction
        prediction_print[1] = imagelist_number
        prediction_print[2] = confidence

def build_model(mode,dataset):
    rgb_holder = None
    if mode in ['rgb', 'mixed']: 
        rgb_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['rgb']])
    flow_holder = None
    if mode in ['flow', 'mixed']:
        flow_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['flow']])

    # insert the model
    rgb_fc_out = None
    if mode in ['rgb', 'mixed']:
        with tf.variable_scope(_SCOPE['rgb']):
            rgb_model = i3d.InceptionI3d(
                400, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                rgb_holder, is_training=False, dropout_keep_prob=1)
            rgb_logits_dropout = tf.nn.dropout(rgb_logits, 1)
            rgb_fc_out = tf.layers.dense(
                rgb_logits_dropout, _CLASS_NUM[dataset], use_bias=True)
    flow_fc_out = None            
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
    rgb_saver = None
    if mode in ['rgb', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['rgb']:
                variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=variable_map)
    variable_map = {}
    flow_saver = None
    if mode in ['flow', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['flow']:
                variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=variable_map, reshape=True)

    # Edited Version by AlexHu
    if mode in ['rgb']:
        fc_out = rgb_fc_out
        prob_predictions = tf.nn.softmax(fc_out)
    elif mode in ['flow']:
        fc_out = flow_fc_out
        prob_predictions = tf.nn.softmax(fc_out)
    elif mode in ['mixed']:
        fc_out = _MIX_WEIGHT_OF_RGB * rgb_fc_out + _MIX_WEIGHT_OF_FLOW * flow_fc_out
        prob_predictions = tf.nn.softmax(fc_out)
        
    return prob_predictions, fc_out, rgb_fc_out, flow_fc_out, rgb_saver, flow_saver, rgb_holder, flow_holder

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
    
def clean_directory(DATA_DIR,mode,imagelist_number,img_path):
    img_str_delete = ""
    for img in img_path[:-1]:
        img_str_delete = img_str_delete + ' '+img
    os.system("rm "+img_str_delete)
    if mode in ['flow', 'mixed']:
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
