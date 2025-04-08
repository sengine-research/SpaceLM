from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
import numpy as np
import math

import torch
from torch.nn.utils.rnn import pad_sequence

class MyTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            # get all parameter names
            all_parameters = set([name for name, _ in opt_model.named_parameters()])
            # filter out point_backbone parameters
            point_backbone_parameters = set([name for name, _ in opt_model.named_parameters() if "point_proj" in name or "point_attention" in name])

            # other parameters are non point_backbone parameters
            other_parameters = all_parameters - point_backbone_parameters

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if n in point_backbone_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 10  # set a specific learning rate for point_backbone parameters
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if n in other_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate  # set a common learning rate for other parameters
                }
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    
        return self.optimizer

def custom_collate_fn(batch, tokenizer):
    input_ids = pad_sequence([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.tensor(item['labels'], dtype=torch.long) for item in batch], batch_first=True, padding_value=-100)
    coord = pad_sequence([torch.tensor(item['coord'], dtype=torch.float32) for item in batch], batch_first=True, padding_value=np.nan)
    grid_coord = pad_sequence([torch.tensor(item['grid_coord'], dtype=torch.int64) for item in batch], batch_first=True, padding_value=0)
    color = pad_sequence([torch.tensor(item['color'], dtype=torch.float32) for item in batch], batch_first=True, padding_value=np.nan)
    inverse = pad_sequence([torch.tensor(item['inverse'], dtype=torch.int64) for item in batch], batch_first=True, padding_value=0)
    offset = pad_sequence([torch.tensor(item['offset'], dtype=torch.int64) for item in batch], batch_first=True, padding_value=0)
    feat = pad_sequence([torch.tensor(item['feat'], dtype=torch.float32) for item in batch], batch_first=True, padding_value=np.nan)
    if batch[0]['point_feat']:
        point_feat = pad_sequence([torch.tensor(item['point_feat'], dtype=torch.float32) for item in batch], batch_first=True, padding_value=np.nan)
        point_feat_coord = pad_sequence([torch.tensor(item['point_feat_coord'], dtype=torch.float32) for item in batch], batch_first=True, padding_value=np.nan)
        min_extent = pad_sequence([torch.tensor(item['min_extent'], dtype=torch.float32) for item in batch], batch_first=True, padding_value=np.nan)
    else:
        point_feat = None
        point_feat_coord = None
        min_extent = None
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "coord": coord,
        "grid_coord": grid_coord,
        "color": color,
        "inverse": inverse,
        "offset": offset,
        "feat": feat,
        "point_feat": point_feat,
        "point_feat_coord": point_feat_coord,
        "min_extent": min_extent
    }

def load_point_backbone_parameters(model_wrapper):
    
    if model_wrapper.config.point_backbone == "scenescript":
        point_backbone_model = torch.load("scenescript_model_ase.ckpt")
        
        # directly load parameters (ensure key names are correctly matched)
        model_wrapper.load_state_dict(
            {k.replace("encoder.", "point_backbone."): v 
            for k, v in point_backbone_model["model_state_dict"].items() if "stem.0.0.kernel" not in k}, strict=False)
        for param in model_wrapper.point_backbone.sparse_resnet.parameters():
            param.requires_grad = False
    else:
        import sonata
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,  # reduce patch size if necessary
        )
        sonata_model = sonata.load("facebook/sonata", custom_config=custom_config)
        model_wrapper.point_backbone.encoder_model.load_state_dict(sonata_model.state_dict())
        for param in model_wrapper.point_backbone.encoder_model.parameters():
            param.requires_grad = False

    # initialize the parameters randomly
    def init_weights(param, layer_type='default'):
        if param.dim() >= 2:
            if 'attention' in layer_type:
                # use a more precise initialization scale
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
                if 'out_proj' in layer_type:
                    param.data.mul_(1.0 / math.sqrt(2.0))  # shrink the initialization range for the output projection
            else:
                torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.constant_(param, 0.0)

    for name, param in model_wrapper.named_parameters():
        if 'point_attention' in name:
            if 'in_proj_weight' in name:
                # get the actual parameter dimension
                hidden_size = model_wrapper.config.hidden_size
                num_heads = 8  # need to be consistent with the num_heads in the model definition
                head_dim = hidden_size // num_heads
                
                # when calculating the standard deviation, consider head_dim
                std = 1.0 / math.sqrt(head_dim)
                # initialize the query/key/value separately
                torch.nn.init.normal_(param[:hidden_size], std=std)  # query
                torch.nn.init.normal_(param[hidden_size:2*hidden_size], std=std)  # key
                torch.nn.init.normal_(param[2*hidden_size:], std=0.02)  # value
            elif 'in_proj_bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'out_proj' in name:
                init_weights(param, layer_type='attention_out')
        elif 'point_proj' in name or 'stem.0.0.kernel' in name:
            init_weights(param, layer_type='proj')

    return model_wrapper.to("cuda")