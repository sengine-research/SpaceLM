#!/bin/bash

######################################################################################
# set the img_dir below to the directory of the set of images you want to reconstruct
# set the postfix below to the format of the rgb images in the img_dir
######################################################################################
TEST_DATASET="Seq_Data(img_dir='data/Replica_demo/room0', postfix='.jpg', \
img_size=224, silent=False, sample_freq=1, \
start_idx=0, num_views=-1, start_freq=1, to_tensor=True)"

######################################################################################
# set the parameters for whole scene reconstruction below
# for defination of these parameters, please refer to the recon.py
######################################################################################
TEST_NAME="Replica_demo"
KEYFRAME_STRIDE=20
UPDATE_BUFFER_INTV=3
MAX_NUM_REGISTER=10
WIN_R=5
NUM_SCENE_FRAME=10
INITIAL_WINSIZE=5 
CONF_THRES_L2W=10
CONF_THRES_I2P=1.5
NUM_POINTS_SAVE=1000000

GPU_ID=-1


python recon.py \
--test_name $TEST_NAME \
--dataset "${TEST_DATASET}" \
--gpu_id $GPU_ID \
--keyframe_stride $KEYFRAME_STRIDE \
--win_r $WIN_R \
--num_scene_frame $NUM_SCENE_FRAME \
--initial_winsize $INITIAL_WINSIZE \
--conf_thres_l2w $CONF_THRES_L2W \
--conf_thres_i2p $CONF_THRES_I2P \
--num_points_save $NUM_POINTS_SAVE \
--update_buffer_intv $UPDATE_BUFFER_INTV \
--max_num_register $MAX_NUM_REGISTER
