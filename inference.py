import os
import glob
import argparse

import torch
import numpy as np
from tqdm import tqdm
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from layout.layout import Layout
from model.spacelm_qwen import SpaceLMQwenForCausalLM
from pcd.pcd_loader import load_o3d_pcd, get_points_and_colors, get_points_and_colors_and_normals, cleanup_pcd, Compose
from dataset.preprocess_ply import preprocess_point_cloud_and_normals

def generate_layout(
    model,
    coord, grid_coord, color, offset, inverse, feat, min_extent,
    tokenizer,
    code_template_file,
    top_k=10,
    top_p=0.95,
    temperature=0.6,
    num_beams=1,
    max_new_tokens=4096,
):
    # load the code template
    with open(code_template_file, "r") as f:
        code_template = f.read()

    prompt = f"<|point_start|><|point_pad|><|point_end|>Detect boxes. the output explanation and the code template is as follows: {code_template}"

    # prepare the conversation data
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = {
        "input_ids": input_ids,
        "coord": coord.unsqueeze(0),
        "grid_coord": grid_coord.unsqueeze(0),
        "color": color.unsqueeze(0),
        "offset": offset.unsqueeze(0),
        "inverse": inverse.unsqueeze(0),
        "feat": feat.unsqueeze(0),
        "min_extent": min_extent.unsqueeze(0),
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "use_cache": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_beams": num_beams,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    print("Generating layout...\n")
    generate_texts = []
    for text in streamer:
        generate_texts.append(text)
        print(text, end="", flush=True)
    print("\nDone!")

    layout_str = "".join(generate_texts)
    layout = Layout(layout_str)
    return layout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--point_cloud",
        type=str,
        required=True,
        help="Path to the input point cloud file or a folder containing multiple point cloud files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output layout txt file or a folder to save multiple layout txt files",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="fine_tuned_space_lm_model_qwen_lr_5e-6_test/epoch4.0",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "-t",
        "--code_template_file",
        type=str,
        default="code_template.txt",
        help="Path to the code template file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="The number of highest probability vocabulary tokens to keep for top-k filtering",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The value used to module the next token probabilities",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams for beam search",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen2.5-0.5B-Instruct",
        help="Path to the tokenizer checkpoint",
    )
    args = parser.parse_args()

    # load the model
    print("loading model")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = SpaceLMQwenForCausalLM.from_pretrained(args.model_path)

    model.to("cuda")
    model.set_point_backbone_dtype(torch.float32)
    model.eval()

    # check if the input is a single point cloud file or a folder containing multiple point cloud files
    if os.path.isfile(args.point_cloud):
        point_cloud_files = [args.point_cloud]
    else:
        point_cloud_files = glob.glob(os.path.join(args.point_cloud, "*.ply"))

    for point_cloud_file in tqdm(point_cloud_files):
        # load the point cloud
        point_cloud = load_o3d_pcd(point_cloud_file)
        # point_cloud = cleanup_pcd(point_cloud)
        points, colors, normals = get_points_and_colors_and_normals(point_cloud)
        min_extent = np.min(points, axis=0)
        max_extent = np.max(points, axis=0)
        mid_extent = np.array([(min_extent[0] + max_extent[0]) / 2, (min_extent[1] + max_extent[1]) / 2, min_extent[2]])

        # preprocess the point cloud to tensor features
        grid_size = Layout.get_grid_size()
        num_bins = Layout.get_num_bins()
        coord, grid_coord, color, offset, inverse, feat = preprocess_point_cloud_and_normals(points, colors, normals, grid_size)
        # generate the layout
        layout = generate_layout(
            model,
            coord, grid_coord, color, offset, inverse, feat, torch.tensor(min_extent, dtype=torch.float32).cuda(),
            tokenizer,
            args.code_template_file,
            args.top_k,
            args.top_p,
            args.temperature,
            args.num_beams,
        )
        
        layout.undiscretize_and_unnormalize()
        layout.translate(min_extent)
        pred_language_string = layout.to_language_string()

        # check if the output path is a file or directory
        if os.path.splitext(args.output)[-1]:
            with open(args.output, "w") as f:
                f.write(pred_language_string)
        else:
            output_filename = os.path.basename(point_cloud_file).replace(".ply", ".txt")
            os.makedirs(args.output, exist_ok=True)
            with open(os.path.join(args.output, output_filename), "w") as f:
                f.write(pred_language_string)
