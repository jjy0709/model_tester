#!/usr/bin/env python3
from traceback import print_exc
from common.transformations.camera import transform_img, get_M, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.lanes_image_space import transform_points
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import pbcvt
from datetime import date, datetime

import cv2 
# from tensorflow.keras.models import load_model
from common.tools.lib.parser import parser
import cv2
import sys
import os
camerafile = sys.argv[1]
input = sys.argv[1].replace(".mp4", ".json")
cmd = './gpmf-parser ' + camerafile + ' ' + input
os.system(cmd)
# supercombo = load_model('models/supercombo.keras')

import onnxruntime
import onnx
import pdb
import csv
import json

import torch
import torchvision
from onnx2pytorch import ConvertModel
from torchsummary import summary

# so = onnxruntime.SessionOptions()
# so.add_session_config_entry('session.load_model_format','ORT')

# onnxruntime.set_default_logger_severity(0)
sess_options = onnxruntime.SessionOptions()
# sess_options.add_session_config_entry('session.load_model_format','ORT')
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
    }),
    'CPUExecutionProvider',
]

onnx_session = onnxruntime.InferenceSession('models/supercombo_kl.onnx', sess_options, providers=providers)
print(onnxruntime.get_device())
print(onnx_session.get_providers())
# onnx_model = onnx.load('models/supercombo.onnx')
# torch_model = ConvertModel(onnx_model, experimental=True)
# print(torch_model)
# torch_model.eval()
# torch_model.cuda()
#MAX_DISTANCE = 140.
#LANE_OFFSET = 1.8
#MAX_REL_V = 10.

xoff = -120

###############################################################################################

FLIR_INTRINSIC = np.array([[ 761.0873603470,    0.6397502148,  xoff + 638.3952017077],
    [0.0000000000,  764.1428410652,  519.2878000527],
    [0.0000000000,    0.0000000000,    1.0000000000]])

GALAXY_INTRINSIC = np.array([[602, 0.0, xoff + 360],
                            [0.0, 602, 240],
                            [0.0, 0.0, 1.0]])

gopro_intrinsics = np.array([[850.0, 0.0, 960.0],
                              [0.0, 850.0, 640.0],
                              [0.0, 0.0, 1.0]])

galaxy_intrinsics = np.array([[1000, 0.0, 960],
                            [0.0, 1000, 540],
                            [0.0, 0.0, 1.0]])

#################################################################################################

#LEAD_X_SCALE = 10
#LEAD_Y_SCALE = 10

# 1164 874

# Input image resize target
# Make difference on result!
TARGET_WIDTH = 1164
# TARGET_HEIGHT = 720
TARGET_HEIGHT = 874
# Display size of input image
# No difference on result (Just easy to watch result)

DISPLAY_HEIGHT = 450

# import subprocess
# cmd = 'ffmpeg -i '+camerafile+' -c copy -an noaudio_'+camerafile
# subprocess.call(cmd, shell=True)
cap = cv2.VideoCapture(camerafile)

# imgs = []

# fcount = 0
# while True:
#   try:
#     ret, frame = cap.read()
#     height, width, channel = frame.shape
#     fcount += 1
#     print(fcount)
#   except:
#     break


# for i in tqdm(range(800)):
#   ret, frame = cap.read()
#   height, width, channel = frame.shape

#   #if height > TARGET_HEIGHT:
#   width = int( width * TARGET_HEIGHT / height)
#   dim = (width, TARGET_HEIGHT)
#   height = TARGET_HEIGHT
#   w = TARGET_WIDTH
#   h = TARGET_HEIGHT
#   # TARGET_WIDTH and TARGET_HEIGHT are set to 1164, 874 as sample.hevc file. You can modify these at line 31.
#   # Below code will crop image.
#   # x axis: from center (automatically)
#   # y axis: from top (you can adjust 'y' to make change for wi arae wi wi arae wi arae wiwi arae...)
#   x = int(width/2 -w/2)
#   y = 101             # modify this from 0 to (HEIGHT - TARGET_HEIGHT).
#   frame = frame[y: y + h, x: x + w]
#   width = w
#   height = h
#   frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)  

#   img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
#   # cv2.imshow('test',frame)
#   # cv2.waitKey(100)  
#   # cv2.imshow('test2',img_yuv.reshape((height*3//2, width)))
#   # cv2.waitKey(100)  
#   imgs.append(img_yuv.reshape((height*3//2, width)))
# fcount = 0
# while True:
#   try:
#     ret, frame = cap.read()
#     height, width, channel = frame.shape
#     fcount += 1
#   except:
#     break
#   # cv2.imshow('raw',frame)
#   # frame = cv2.flip(frame,0)
#   # frame = cv2.flip(frame,1)
#   #if height > TARGET_HEIGHT:
#   #width = int( width * TARGET_HEIGHT / height)
#   #dim = (width, TARGET_HEIGHT)
#   #height = TARGET_HEIGHT
#   #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#   w = TARGET_WIDTH
#   h = TARGET_HEIGHT
#   # TARGET_WIDTH and TARGET_HEIGHT are set to 1164, 874 as sample.hevc file. You can modify these at line 31.
#   # Below code will crop image.
#   # x axis: from center (automatically)
#   # y axis: from top (you can adjust 'y' to make change for wi arae wi wi arae wi arae wiwi arae...)
#   x = int(width/2 -w/2)
#   y = 0             # modify this from 0 to (HEIGHT - TARGET_HEIGHT).
#   xoff = 0
#   # frame = frame[y: y + h, x+xoff: x + w +xoff]
#   width = w
#   height = h
#   img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
#   # imgs.append(img_yuv.reshape((height*3//2, width)))
#   # if i%6 == 0:
#   img_yuv = transform_img(img_yuv, from_intr=FLIR_INTRINSIC, to_intr=medmodel_intrinsics, yuv=True,
#                                     output_size=(512,256))
#   imgs.append(img_yuv)

def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((1, 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  # cv2.imshow('0',frames[0,H+H//4:H+H//2].reshape((H//2,W//2)))
  # # print(frames[:, 0:H:2, 0::2].shape())
  # # cv2.imshow('1',in_img1[:,1])
  # # cv2.imshow('2',in_img1[:,2])
  # # cv2.imshow('3',in_img1[:,3])
  # # cv2.imshow('4',in_img1[:,4])
  # # cv2.imshow('5',in_img1[:,5])
  # cv2.waitKey(1000)
  return in_img1

# cv2.namedWindow('image')
# cv2.resizeWindow('image',1920,1080)


  # testimg = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
  #                                   output_size=(512, 256))
  # cv2.imshow('image',imgs_med_model[i])
  # cv2.waitKey(10)
  # print(img.shape)
# frame_tensors = frames_to_tensor(np.array(imgs)).astype(np.float32)


'''
ONNX PARAMS START
'''
DESIRE_PRED_SIZE = 32
OTHER_META_SIZE = 4
DESIRE_LEN = 8

MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 / 2

PLAN_MHP_N = 5
PLAN_MHP_COLUMNS = 30
PLAN_MHP_VALS = 30*33
PLAN_MHP_SELECTION = 1
PLAN_MHP_GROUP_SIZE =  (2*PLAN_MHP_VALS + PLAN_MHP_SELECTION)

LEAD_MHP_N = 5
LEAD_MHP_VALS = 4
LEAD_MHP_SELECTION = 3
LEAD_MHP_GROUP_SIZE = (2*LEAD_MHP_VALS + LEAD_MHP_SELECTION)

POSE_SIZE = 12

PLAN_IDX = 0
LL_IDX = PLAN_IDX + PLAN_MHP_N*PLAN_MHP_GROUP_SIZE
LL_PROB_IDX = LL_IDX + 4*2*2*33
RE_IDX = LL_PROB_IDX + 4
LEAD_IDX = RE_IDX + 2*2*2*33
LEAD_PROB_IDX = LEAD_IDX + LEAD_MHP_N*(LEAD_MHP_GROUP_SIZE)
DESIRE_STATE_IDX = LEAD_PROB_IDX + 3
META_IDX = DESIRE_STATE_IDX + DESIRE_LEN
POSE_IDX = META_IDX + OTHER_META_SIZE + DESIRE_PRED_SIZE
OUTPUT_SIZE =  POSE_IDX + POSE_SIZE

X_IDX = [ 0.    ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.]

def lanexyzt(data, startindex):
  column = 2
  column_offset = -1
  # curdata = data[startindex:]
  # probdata = data[LL_PROB_IDX-LL_IDX + startindex:]
  prob_idx = (startindex - LL_IDX)//66
  std_idx = startindex + 66*4
  outdict = {}
  outdict['x'] = X_IDX
  outdict['y'] = []
  outdict['std_y'] = []
  outdict['prob'] = []
  #outdict['std_y'].append(np.exp(data[std_idx]))
  #outdict['prob'].append(1/(1+np.exp(-data[LL_PROB_IDX+prob_idx])))
  for i in range(0,33):
    
    outdict['y'].append(data[startindex+i*2])
    outdict['std_y'].append(np.exp(data[std_idx+i*2]))
    outdict['prob'].append(1/(1+np.exp(-data[LL_PROB_IDX+prob_idx+i*2])))
  
  return outdict

def pathxyzt(data):
  column = PLAN_MHP_COLUMNS
  
  outdict = {}
  outdict['x'] = []
  outdict['y'] = []
  outdict['z'] = []
  for i in range(0,33):
    outdict['x'].append(data[i*column])
    outdict['y'].append(data[i*column +1])
    outdict['z'].append(data[i*column +2])

  return outdict

def velxyzt(data):
  column = PLAN_MHP_COLUMNS
  outdict = {}
  outdict['x'] = []
  outdict['y'] = []
  outdict['z'] = []
  for i in range(0,33):
    outdict['x'].append(data[i*column +3])
    outdict['y'].append(data[i*column +4])
    outdict['z'].append(data[i*column +5])

  return outdict
def get_best_data(data, size, group_size, offset):
  max_idx = 0
  for i in range(0,size):
    if data[(i+1)*group_size+offset]>data[(max_idx+1)*group_size+offset]:
      max_idx = i
  return data[(max_idx*group_size):]

def get_plan_data(plan):
  return get_best_data(plan, PLAN_MHP_N, PLAN_MHP_GROUP_SIZE, -1)

def get_lead_data(lead, t_offset):
  # pdb.set_trace()
  return get_best_data(lead, LEAD_MHP_N, LEAD_MHP_GROUP_SIZE, t_offset-LEAD_MHP_SELECTION)


def lead(lead_data, prob, t_offset):
  data = get_lead_data(lead_data, t_offset)
  lead = {}
  lead['x'] = data[0]
  lead['y'] = data[1]
  lead['v'] = data[2]
  lead['a'] = data[3]
  # leat['t'] = T
  # lead['std_x'] = np.exp(data[LEAD_MHP_VALS])
  # lead['std_y'] = np.exp(data[LEAD_MHP_VALS + 1])
  # lead['std_v'] = np.exp(data[LEAD_MHP_VALS + 2])
  # lead['std_a'] = np.exp(data[LEAD_MHP_VALS + 3])

  return lead


'''
ONNX PARAMS END
'''

state = np.zeros((1,512)).astype(np.float32)
desire = np.zeros((1,8)).astype(np.float32)
desire[0][0] = 1
traffic_convention = np.array([[0,0]]).astype(np.float32)

cap = cv2.VideoCapture(camerafile)
frame_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_list = list()
# fcount = 0
# w = TARGET_WIDTH
# h = TARGET_HEIGHT
# x = int(width/2 -w/2)
# y = 0             # modify this from 0 to (HEIGHT - TARGET_HEIGHT).
# xoff = 0
# width = w
# height = h
# for i in tqdm(range(fcount - 1)):
for i in tqdm(range(0,int(frame_len))):
  try:
    time = cap.get(cv2.CAP_PROP_POS_MSEC)
    ret, frame = cap.read()
    height, width, channel = frame.shape
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    # print(datetime.now())
    # img_yuv = pbcvt.mycvtColor(frame)

    # img_yuv = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
    #                                 output_size=(512,256))
    size = img_yuv.shape[:2]
    cy = galaxy_intrinsics[1,2]
    M = get_M(galaxy_intrinsics, cy, size, np.array([0,0,0]), np.array([0,0,0]), medmodel_intrinsics)
    # print(datetime.now())
    
    img_yuv = pbcvt.transform_img(img_yuv, M)
    # print(datetime.now())
    
    # img_yuv = transform_img(img_yuv, M, from_intr=galaxy_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
    #                                output_size=(512,256),augment_eulers=np.array([-0.05,-0.05,0.0]))
    
    # img_yuv = transform_img(img_yuv, from_intr=FLIR_INTRINSIC, to_intr=medmodel_intrinsics, yuv=True,
    #                                 output_size=(512,256),augment_eulers=np.array([0.025,0.07,0.0]))
    frame_tensors = frames_to_tensor(np.array([img_yuv])).astype(np.float32)
    frame_list.append(dict(frame=frame_tensors, time=time))
    # print(datetime.now())
    
  except:
    import traceback
    traceback.print_exc()
    break

  

  # img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  # inputs = [np.vstack(frame_tensors,frame_tensors_prev)[None], desire, state]
  # print(frame_tensors.shape)
#   if i == 0:
#     frame_tensors_prev = frame_tensors
#     continue
frame_tensors = frame_list[0]['frame']
frame_tensors_prev = frame_tensors

for i in tqdm(range(0,len(frame_list))):
  frame_tensors = frame_list[i]['frame']
  input_dict = {'input_imgs':np.vstack([frame_tensors_prev[0],frame_tensors[0]])[None], 'desire':desire, 'initial_state':state, 'traffic_convention':traffic_convention}
  frame_tensors_prev = frame_tensors
  time = frame_list[i]['time']
  label_name = onnx_session.get_outputs()[0].name
  outs = onnx_session.run([label_name], input_dict)
  outarray = outs[0][0]
  # img = torch.from_numpy(np.vstack(frame_tensors[i:i+2])[None])
  # des = torch.from_numpy(desire)
  # sta = torch.from_numpy(state)
  # tc = torch.from_numpy(traffic_convention)
  # # print(img.size())
  # # print(des.size())
  # # print(sta.size())
  # # print(tc.size())
  # torch_out = torch_model(img.cuda(),des.cuda(), tc.cuda(),sta.cuda()).cpu()
  # outarray = torch_out.detach().numpy()[0]
  # print(outarray.shape)
  # desire = np.array([outarray[DESIRE_STATE_IDX:DESIRE_STATE_IDX+DESIRE_LEN]])
  # print(desire)
  # state = np.array([outarray[-512:]])

  # with open('./data/input{}.npy'.format(i), 'wb') as f:
  #   np.save(f, np.vstack(frame_tensors[i:i+2])[None])

  # with open('./data/desire{}.npy'.format(i), 'wb') as f:
  #   np.save(f, desire)

  # with open('./data/traffic{}.npy'.format(i), 'wb') as f:
  #   np.save(f, traffic_convention)

  # with open('./data/initial{}.npy'.format(i), 'wb') as f:
  #   np.save(f, state)

  # outs = supercombo.predict(inputs)
  # parsed = parser(outs)
  # state = outs[-1]
  # pose = outs[-2]
  # print(outarray.shape)
  planarray = outarray[0:LL_IDX]
  best_plan = get_plan_data(planarray)
  # print(len(outarray[LEAD_IDX:LEAD_PROB_IDX]))
  lead0 = lead(outarray[LEAD_IDX:LEAD_PROB_IDX], outarray[LEAD_PROB_IDX:DESIRE_STATE_IDX],0)
  lead2 = lead(outarray[LEAD_IDX:LEAD_PROB_IDX], outarray[LEAD_PROB_IDX:DESIRE_STATE_IDX],1)
  lead4 = lead(outarray[LEAD_IDX:LEAD_PROB_IDX], outarray[LEAD_PROB_IDX:DESIRE_STATE_IDX],2)
  path = pathxyzt(best_plan)
  vel = velxyzt(best_plan)
  ll = lanexyzt(outarray, LL_IDX)
  l = lanexyzt(outarray, LL_IDX + 66)
  r = lanexyzt(outarray, LL_IDX + 66*2)
  rr = lanexyzt(outarray, LL_IDX + 66*3)
  print("TRANS : ",outarray[10803:10806])
  print("ROT : ", outarray[POSE_IDX+3:POSE_IDX+6])
  # print(1/(1 + np.exp(-outarray[LL_PROB_IDX:LL_PROB_IDX+4])))
  # t0 = xyzt(outarray, LL_IDX + 66*4)
  # t1 = xyzt(outarray, LL_IDX + 66*5)
  # t2 = xyzt(outarray, LL_IDX + 66*6)
  # t3 = xyzt(outarray, LL_IDX + 66*7)
  le = lanexyzt(outarray, LL_IDX + 4 + 66*8)
  re = lanexyzt(outarray, LL_IDX + 4+ 66*9)
  # t6 = xyzt(outarray, LL_IDX + 4 +66*10)
  # t7 = xyzt(outarray, LL_IDX + 66*11)
  # t8 = xyzt(outarray, LL_IDX + 66*12)
  # t9 = xyzt(outarray, LL_IDX + 66*13)
  # t10 = xyzt(outarray, LL_IDX + 33*14)
  # t11 = xyzt(outarray, LL_IDX + 33*15) 
  # print(vel)
  pose = np.hstack([outarray[POSE_IDX:POSE_IDX+6]])
  try:
    llystdsave = np.vstack((llystdsave, ll['std_y']))
    lystdsave = np.vstack((lystdsave, l['std_y']))
    rystdsave = np.vstack((rystdsave, r['std_y']))
    rrystdsave = np.vstack((rrystdsave, rr['std_y']))
    llprobsave = np.vstack((llprobsave, ll['prob']))
    lprobsave = np.vstack((lprobsave, l['prob']))
    rprobsave = np.vstack((rprobsave, r['prob']))
    rrprobsave = np.vstack((rrprobsave, rr['prob']))
    llysave = np.vstack((llysave, ll['y']))
    # llxsave = np.vstack((llxsave, ll['x']))
    lysave = np.vstack((lysave, l['y']))
    # lxsave = np.vstack((lxsave, l['x']))
    rysave = np.vstack((rysave, r['y']))
    # rxsave = np.vstack((rxsave, r['x']))
    rrysave = np.vstack((rrysave, rr['y']))
    # rrxsave = np.vstack((rrxsave, rr['x']))
    lesave = np.vstack((lesave, le['y']))
    resave = np.vstack((resave, re['y']))
    posesave = np.vstack((posesave, pose))
    timesave = np.vstack((timesave, time))
  except:
    llystdsave = ll['std_y']
    lystdsave = l['std_y']
    rystdsave = r['std_y']
    rrystdsave = rr['std_y']
    llprobsave = ll['prob']
    lprobsave = l['prob']
    rprobsave = r['prob']
    rrprobsave = rr['prob']
    llysave = ll['y']
    # llxsave = ll['x']
    lysave = l['y']
    # lxsave = l['x']
    rysave = r['y'] 
    # rxsave = r['x']
    rrysave = rr['y']
    # rrxsave = rr['x']
    posesave = pose
    resave = re['y']
    lesave = le['y']
    timesave = time
  # ret, frame = cap.read()
  # h, w, c = frame.shape
  # # frame = cv2.flip(frame,0)
  # # frame = cv2.flip(frame,1)

  # # display_width = int( w * DISPLAY_HEIGHT / h)
  # # dim = (display_width, DISPLAY_HEIGHT)
  # # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)  
  # # frame = frame[y: y + h, x+xoff: x + w +xoff]
  # # Show raw camera image
  # # cv2.namedWindow("Input Image")
  # # cv2.moveWindow("Input Image",960,0)
  # # cv2.imshow("Input Image", frame)
  # # Clean plot for next frame
  # plt.clf()
  # plt.title("lanes and path")
  
  # # plt.xlim([-20, 20])
  # # plt.ylim([0, 200])
  # # thismanager = plt.get_current_fig_manager()
  # # thismanager.window.wm_geometry("+960+540")

  # # lll = left lane line
  # # print(parsed.keys())

  # '''PLOT TOPVIEW'''

  # cv2.namedWindow("Input Image")
  # cv2.moveWindow("Input Image",960,0)
 
  # cv2.imshow("Input Image", cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB_I420))
  # plt.clf()
  # plt.xlim(-20, 20)
  # plt.ylim(0, 200)
  # plt.plot(path['y'], path['x'], linewidth=3)
  # leada = outarray[LEAD_IDX:LEAD_PROB_IDX]
  # plt.plot(ll['y'], ll['x'], linewidth=1)
  # plt.plot(l['y'], l['x'], linewidth=3)
  # plt.plot(r['y'], r['x'], linewidth=3)
  # plt.plot(rr['y'], rr['x'], linewidth=1)
  # plt.plot(lead0['y'],lead0['x'],'o')
  # plt.plot(lead2['y'],lead2['x'],'o')
  # plt.plot(lead4['y'],lead4['x'],'o')
  # plt.plot(le['y'], le['x'], linewidth=2)
  # plt.plot(re['y'], re['x'], linewidth=2)

  # '''PLOT TOPVIEW END'''

  # '''PLOT OVERLAY VIEW'''
  # plt.xlim(0, 1164)
  # plt.ylim(874, 0)
  # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
  
  # plotoffset = 20

  # ll_2d = {'x':[],'y':[]}
  # ll_2d['x'], ll_2d['y'] = transform_points(ll['x'], ll['y'])

  # l_2d = {'x':[],'y':[]}
  # l_2d['x'], l_2d['y'] = transform_points(l['x'], l['y'])

  # r_2d = {'x':[],'y':[]}
  # r_2d['x'], r_2d['y'] = transform_points(r['x'], r['y'])

  # rr_2d = {'x':[],'y':[]}
  # rr_2d['x'], rr_2d['y'] = transform_points(rr['x'], rr['y'])

  # path_2d = {'x':[],'y':[]}
  # path_2d['x'], path_2d['y'] = transform_points(path['x'], path['y'])

  # le_2d = {'x':[],'y':[]}
  # le_2d['x'], le_2d['y'] = transform_points(le['x'], le['y'])

  # re_2d = {'x':[],'y':[]}
  # re_2d['x'], re_2d['y'] = transform_points(re['x'], re['y'])

  # plt.plot(ll_2d['x'], ll_2d['y'],label='transformed',  color='w')
  # plt.plot(l_2d['x'], l_2d['y'],label='transformed',  color='w')
  # plt.plot(r_2d['x'], r_2d['y'],label='transformed',  color='w')
  # plt.plot(rr_2d['x'], rr_2d['y'],label='transformed',  color='w')
  # plt.plot(path_2d['x'], path_2d['y'],label='transformed',  color='w')
  # plt.plot(le_2d['x'], ll_2d['y'],label='transformed',  color='w')
  # plt.plot(re_2d['x'], re_2d['y'],label='transformed', color='w')


  '''PLOT OVERLAY END'''
  

  # rll = right lane line
  # plt.plot(parsed["rll"][0], range(0, 192), "r-", linewidth=1)

  # plt.plot(parsed["lll"][0], range(0, 192), "b-", linewidth=1)

  # # path = path cool isn't it ?
  # plt.plot(parsed["path"][0], range(0, 192), "g-", linewidth=1)
  
  # print(np.array(pose[0,:3]).shape)
  # plt.scatter(pose[0,:3], range(3), c="y")
  
  # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis
  # plt.gca().invert_xaxis()
  print(datetime.now())
  plt.pause(0.0002)
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break

filename = camerafile.replace("gpmf.mp4","result.json")

with open(input,'r') as g:
  gpmf = json.load(g)
gps_data = gpmf['GPS data']
accl_data = gpmf['ACCL data']
gyro_data = gpmf['GYRO data']

f = open(filename, 'w')
result_data = list()
video_metadata = dict(Video_metadata=gpmf['Video metadata'])
camera_intrinsics = dict(Camera_Intrinsics=gpmf['Camera Intrinsics'])
result_data.append(video_metadata)
result_data.append(camera_intrinsics)

j=0
k=0
for i2 in range(0, len(accl_data['Time data'])):
  gpmf_time = accl_data['Time data'][i2]
  if(len(gps_data['Time data']) != 0):
    if(k==len(gps_data['Time data'])-1):
        k=k
    elif(gpmf_time > gps_data['Time data'][k+1]):
        k = k + 1
        if(k==len(gps_data['Time data'])):
            k = k - 1
  if(j==len(timesave)-1):
    j=j
  elif(gpmf_time > timesave[j+1]*1000):
    j = j + 1
    if(j==len(timesave)):
        j = j - 1
  
  #gps data
  gpmf_data = dict()
  gpmf_data['Time'] = gpmf_time
  if(len(gps_data['Time data']) != 0):
    gpmf_data['GPS Time'] = gps_data['Time data'][k]
    gpmf_data['GPS Latitude'] = gps_data['Latitude data'][k]
    gpmf_data['GPS Longitude'] = gps_data['Longitude data'][k]
    gpmf_data['GPS Altitude'] = gps_data['Altitude data'][k]
    gpmf_data['GPS 2d speed'] = gps_data['2d speed data'][k]
    gpmf_data['GPS Bearing'] = gps_data['Bearing data'][k]
  #accl data
  gpmf_data['ACCL x'] = accl_data['Accl x data'][i2]
  gpmf_data['ACCL y'] = accl_data['Accl y data'][i2]
  gpmf_data['ACCL z'] = accl_data['Accl z data'][i2]
  #gyro data
  gpmf_data['GYRO x'] = gyro_data['Gyro x data'][i2]
  gpmf_data['GYRO y'] = gyro_data['Gyro y data'][i2]
  gpmf_data['GYRO z'] = gyro_data['Gyro z data'][i2]
  #script data
  gpmf_data['frame'] = j
  gpmf_data['frame time'] = float(timesave[j])*1000
  gpmf_data['lly'] = llysave[j].tolist()
  gpmf_data['llstd'] = llystdsave[j].tolist()
  gpmf_data['llprob'] = llprobsave[j].tolist()
  gpmf_data['ly'] = lysave[j].tolist()
  gpmf_data['lstd'] = lystdsave[j].tolist()
  gpmf_data['lprob'] = lprobsave[j].tolist()
  gpmf_data['ry'] = rysave[j].tolist()
  gpmf_data['rstd'] = rystdsave[j].tolist()
  gpmf_data['rprob'] = rprobsave[j].tolist()
  gpmf_data['rry'] = rrysave[j].tolist()
  gpmf_data['rrstd'] = rrystdsave[j].tolist()
  gpmf_data['rrprob'] = rrprobsave[j].tolist()
  gpmf_data['rey'] = resave[j].tolist()
  gpmf_data['ley'] = lesave[j].tolist()
  gpmf_data['pose'] = posesave[j].tolist()
  result_data.append(gpmf_data)
json.dump(result_data, f, indent=4)
f.close()

# np.savetxt("llprob.csv", llprobsave, delimiter=",")
# np.savetxt('lprob.csv', lprobsave, delimiter=',')
# np.savetxt('rprob.csv', rprobsave, delimiter=',')
# np.savetxt('rrprob.csv', rrprobsave, delimiter=',')
# np.savetxt('rrstd.csv', rrystdsave, delimiter=',')
# np.savetxt("rstd.csv", rystdsave, delimiter=",")
# np.savetxt("lstd.csv", lystdsave, delimiter=",")
# np.savetxt("llstd.csv", llystdsave, delimiter=",")
# # np.savetxt("llx.csv", llxsave, delimiter=",")
# np.savetxt("lly.csv", llysave, delimiter=",")
# # np.savetxt("lx.csv", lxsave, delimiter=",")
# np.savetxt("ly.csv", lysave, delimiter=",")
# # np.savetxt("rx.csv", rxsave, delimiter=",")
# np.savetxt("ry.csv", rysave, delimiter=",")
# # np.savetxt("rrx.csv", rrxsave, delimiter=",")
# np.savetxt("rry.csv", rrysave, delimiter=",")
# np.savetxt("rey.csv", resave, delimiter=",")
# np.savetxt("ley.csv", lesave, delimiter=",")
# np.savetxt("pose.csv", posesave, delimiter=",")
plt.show()
