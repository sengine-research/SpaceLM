#!/bin/bash

######################################################################################
# set the parameters for whole scene reconstruction below
# for defination of these parameters, please refer to the recon.py
######################################################################################
KEYFRAME_STRIDE=20
UPDATE_BUFFER_INTV=3
MAX_NUM_REGISTER=10
WIN_R=5
NUM_SCENE_FRAME=10
INITIAL_WINSIZE=5 
CONF_THRES_I2P=1.5

# the parameter below have nothing to do with the evaluation
NUM_POINTS_SAVE=1000000
CONF_THRES_L2W=10
GPU_ID=-1

SCENE_NAMES=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2") 

for SCENE_NAME in ${SCENE_NAMES[@]};
do

TEST_NAME="Replica_${SCENE_NAME}"

echo "--------Start reconstructing scene ${SCENE_NAME} with test name ${TEST_NAME}--------"

python recon.py \
--test_name "${TEST_NAME}" \
--img_dir "data/Replica/${SCENE_NAME}/results" \
--gpu_id $GPU_ID \
--keyframe_stride $KEYFRAME_STRIDE \
--win_r $WIN_R \
--num_scene_frame $NUM_SCENE_FRAME \
--initial_winsize $INITIAL_WINSIZE \
--conf_thres_l2w $CONF_THRES_L2W \
--conf_thres_i2p $CONF_THRES_I2P \
--num_points_save $NUM_POINTS_SAVE \
--update_buffer_intv $UPDATE_BUFFER_INTV \
--max_num_register $MAX_NUM_REGISTER \
--save_for_eval

echo "--------Start evaluating scene ${SCENE_NAME} with test name ${TEST_NAME}--------"

python evaluation/eval_recon.py \
--test_name="${TEST_NAME}" \
--gt_pcd="results/gt/replica/${SCENE_NAME}_pcds.npy"

done