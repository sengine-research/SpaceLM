ply_file="/root/autodl-tmp/SpaceLM/sample_data/scene0000_00/scene0000_00_pc_result.ply"
model_path="./exp/space_lm_model_qwen_llm_lr_2e-6_point_lr_2e-5_bs_16_full_finetune/space_lm_model_qwen_llm_lr_2e-6_point_lr_2e-5_bs_16_full_finetune/stage_2/epoch_4"
tokenizer_path="Qwen2.5-0.5B-Instruct"

python inference.py --model_path $model_path --tokenizer_path $tokenizer_path -p $ply_file -o test.txt
python visualize.py --point_cloud $ply_file --layout test.txt --save test.rrd