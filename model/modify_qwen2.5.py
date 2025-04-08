# download the Qwen2.5-0.5B-Instruct model

from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen2.5-0.5B-Instruct")
args = parser.parse_args()

# modify the config
with open(os.path.join(args.model_path, "config.json"), "r") as f:
    config = json.load(f)

config["point_start_token_id"] = 151652
config["point_end_token_id"] = 151653
config["point_pad_token_id"] = 151654

with open(os.path.join(args.model_path, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

# modify the tokenizer
with open(os.path.join(args.model_path, "tokenizer_config.json"), "r") as f:
    tokenizer_config = json.load(f)

tokenizer_config["added_tokens_decoder"]["151652"]["content"] = "<|point_start|>"
tokenizer_config["added_tokens_decoder"]["151653"]["content"] = "<|point_end|>"
tokenizer_config["added_tokens_decoder"]["151654"]["content"] = "<|point_pad|>"

with open(os.path.join(args.model_path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, indent=4)

with open(os.path.join(args.model_path, "tokenizer.json"), "r") as f:
    tokenizer = json.load(f)

# modify the tokenizer
output_token_list = []
for token in tokenizer["added_tokens"]:
    if token["id"] == 151652:
        token["content"] = "<|point_start|>"
    elif token["id"] == 151653:
        token["content"] = "<|point_end|>"
    elif token["id"] == 151654:
        token["content"] = "<|point_pad|>"
    output_token_list.append(token)

tokenizer["added_tokens"] = output_token_list

with open(os.path.join(args.model_path, "tokenizer.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer, f, indent=4)
