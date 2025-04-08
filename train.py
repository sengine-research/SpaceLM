import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
import datasets
from peft import LoraConfig, get_peft_model

from layout.layout import Layout
from model.spacelm_qwen import SpaceLMQwenForCausalLM
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from train_utils import MyTrainer, custom_collate_fn, load_point_backbone_parameters
import json
import os
import numpy as np

#########################################################
#                                                       #
#                   Train Settings                      #
#                                                       #
#########################################################

# set the number of threads to 1 for dataset map num_proc > 1
torch.set_num_threads(1)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen2.5-0.5B-Instruct")
parser.add_argument("--dataset_dir", type=str, default="preprocessed_data_scene_script")
parser.add_argument("--dataset_name", type=str, default="3RScan")
parser.add_argument("--exp_path", type=str, default="exp")
parser.add_argument("--exp_name", type=str, default="space_lm_model_qwen_llm_lr_1e-5_point_lr_1e-4_no_stage_1") 
parser.add_argument("--stage_1_epochs", type=int, default=4)
parser.add_argument("--stage_2_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--save_per_epoch", type=int, default=5)

args = parser.parse_args()

dataset_name = args.dataset_name
model_name = args.model_path
save_per_epoch = args.save_per_epoch
dataset_dir = args.dataset_dir
exp_path = args.exp_path
exp_name = args.exp_name
stage_1_epochs = args.stage_1_epochs
stage_2_epochs = args.stage_2_epochs
batch_size = args.batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
learning_rate = args.learning_rate


#########################################################
#                                                       #
#                   Load the Model                      #
#                                                       #
#########################################################

# load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = SpaceLMQwenForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    use_cache=False,
    ignore_mismatched_sizes=True
)

# load the point backbone parameters
model = load_point_backbone_parameters(model)


#########################################################
#                                                       #
#                   Data Preprocessing                   #
#                                                       #
#########################################################

# load the dataset
full_dataset = datasets.load_dataset("json", data_files=f"{dataset_dir}/{dataset_name}/scene_info_list.json")['train']
split_dataset = full_dataset.train_test_split(
    test_size=0.1,
    seed=42,
)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# data preprocessing
def process_func(example):
    # point cloud preprocessing
    with open(example['ply_json_path'], "r") as f:
        data = json.load(f)

    with open("code_template.txt", "r") as f:
        code_template = f.read()

    # point cloud preprocessing
    coord = torch.tensor(data['coord'], dtype=torch.float32)
    grid_coord = torch.tensor(data['grid_coord'], dtype=torch.int64)
    color = torch.tensor(data['color'], dtype=torch.float32)
    offset = torch.tensor(data['offset'], dtype=torch.int64)
    inverse = torch.tensor(data['inverse'], dtype=torch.int64)
    feat = torch.tensor(data['feat'], dtype=torch.float32)
    if 'point_feat' in data:
        point_feat = torch.tensor(data['point_feat'], dtype=torch.float32)
        point_feat_coord = torch.tensor(data['point_feat_coord'], dtype=torch.float32)
        min_extent = torch.tensor(data['min_extent'], dtype=torch.float32)
    else:
        point_feat = None
        point_feat_coord = None
        min_extent = None
    
    MAX_LENGTH = 4096

    # load the code template
    example['prompt'] = example['prompt'].replace('\\n', '\n')

    # build the dialog message list
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"<|point_start|><|point_pad|><|point_end|>Detect boxes. the output explanation and the code template is as follows: {code_template}"},
        {"role": "assistant", "content": example['prompt']}
    ]
    # use apply_chat_template to generate the input sequence
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    # generate the attention_mask
    attention_mask = [1] * len(input_ids)
    # build the labels
    instruction_length = len(tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": f"<|point_start|><|point_pad|><|point_end|>Detect boxes. the output explanation and the code template is as follows: {code_template}"}],
        tokenize=True
    ))
    labels = [-100] * instruction_length + input_ids[instruction_length:]

    # do the truncation
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "coord": coord,
        "grid_coord": grid_coord,
        "color": color,
        "offset": offset,
        "inverse": inverse,
        "feat": feat,
        "point_feat": point_feat,
        "point_feat_coord": point_feat_coord,
        "min_extent": min_extent
    }

train_tokenized = train_dataset.map(process_func, num_proc=16)
eval_tokenized = eval_dataset.map(process_func, num_proc=16)

# define the save model callback
class SaveModelCallback(TrainerCallback):
    """save the model after each epoch"""
    def on_epoch_end(self, args, state, control, **kwargs):
        # check if it is the last epoch, if not, save the model
        if not state.is_local_process_zero:
            return
        if not state.is_world_process_zero:
            return
        # save the model every 100 epochs
        if int(state.epoch) % save_per_epoch == 0:
            model_path = f"{output_dir}/epoch_{int(state.epoch)}"
            # merged_model = kwargs['model'].merge_and_unload()
            kwargs['model'].save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"model saved to {model_path}")

# define the experiment directory
exp_dir = os.path.join(exp_path, exp_name)

if not os.path.exists(f"{exp_dir}/{exp_name}"):
    os.makedirs(f"{exp_dir}/{exp_name}")

# save the split dataset
save_path = f"{exp_dir}/{exp_name}/scene_info_list_split.json"
with open(save_path, "w") as f:
    json.dump({
        "train": split_dataset["train"].to_dict(),
        "test": split_dataset["test"].to_dict()
    }, f)

#########################################################
#                                                       #
#                   Stage 1 Training                    #
#                                                       #
#########################################################

# configure the training parameters
output_dir = os.path.join(exp_dir, exp_name, "stage_1")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=stage_1_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="no",
    save_total_limit=10,
    eval_strategy="epoch",
    logging_steps=1,
    learning_rate=learning_rate,
    bf16=True,
    remove_unused_columns=False
)

for name, param in model.named_parameters():
    if "point_backbone" in name:
        param.requires_grad = True
        print(name)
    else:
        param.requires_grad = False

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=lambda batch: custom_collate_fn(batch, tokenizer),
    callbacks=[SaveModelCallback()]  # add the callback
)

# start training
trainer.train()

model.save_pretrained(output_dir)

#########################################################
#                                                       #
#                   Stage 2 Training                    #
#                                                       #
#########################################################

# configure the training parameters
output_dir = os.path.join(exp_dir, exp_name, "stage_2")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=stage_2_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="no",
    eval_strategy="epoch",
    save_total_limit=10,
    logging_steps=1,
    learning_rate=learning_rate,
    bf16=True,
    remove_unused_columns=False
)

for name, param in model.named_parameters():
    param.requires_grad = True
    print(name)

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=lambda batch: custom_collate_fn(batch, tokenizer),
    callbacks=[SaveModelCallback()]
)
# start training
trainer.train()

model.save_pretrained(output_dir)
