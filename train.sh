# python dataset/preprocess_data_scene_script.py --dataset_name Scannet --dataset_dir preprocessed_data_scene_script_clean
python train.py --dataset_dir preprocessed_data_scene_script_clean \
--dataset_name Scannet --model_path ./Qwen2.5-0.5B-Instruct --exp_path ./exp \
--exp_name space_lm_model_qwen_llm_lr_5e-6_point_lr_2e-4_bs_16_full_finetune_stage_1_epochs_4_stage_2_epochs_10 \
--stage_1_epochs 4 --stage_2_epochs 10 --batch_size 1 --gradient_accumulation_steps 16 --learning_rate 5e-6 --save_per_epoch 2