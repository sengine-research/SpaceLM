ply_file="/root/autodl-tmp/SpaceLM/sample_data/00d42bed-778d-2ac6-86a7-0e0e5f5f5660/00d42bed-778d-2ac6-86a7-0e0e5f5f5660_pc_result.ply"
model_path="/root/autodl-tmp/SpaceLM/exp/space_lm_model_qwen_llm_lr_2e-6_point_lr_2e-5_bs_16_full_finetune_Scannet/stage_2/epoch_8"
tokenizer_path="Qwen2.5-0.5B-Instruct"

python inference.py --model_path $model_path --tokenizer_path $tokenizer_path -p $ply_file -o test.txt
python visualize.py --point_cloud $ply_file --layout test.txt --save test.rrd