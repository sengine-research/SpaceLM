from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torchsparse
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers import (
    Qwen2Model,
    Qwen2ForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
)
from torchsparse.utils.collate import sparse_collate
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from torch.nn import MultiheadAttention

IGNORE_INDEX = -100
logger = logging.get_logger(__name__)

class SpaceLMQwenConfig(Qwen2Config):
    model_type = "spacelm_qwen"
    
    def __init__(
        self,
        use_point_attention=False,
        full_tune_point_backbone=False,
        fuse_coord=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_point_attention = use_point_attention
        self.full_tune_point_backbone = full_tune_point_backbone
        self.fuse_coord = fuse_coord
        self.point_backbone = "scenescript"
        self.point_config = {
            "conv_layers": [
                16,
                32,
                64,
                128,
                256
            ],
            "embed_channels": 512,
            "input_channels": 6,
            "num_bins": 640,
            "token_hidden_size": 896
        }

class SpaceLMQwenForCausalLM(Qwen2ForCausalLM):
    config_class = SpaceLMQwenConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.point_backbone == "scenescript":
            from model.pcd_encoder_scene_script import PointCloudEncoder as PointCloudEncoderSceneScript
            input_channels = config.point_config["input_channels"]
            d_model = config.point_config["embed_channels"]
            conv_layers = config.point_config["conv_layers"]
            num_bins = config.point_config["num_bins"]
            token_hidden_size = config.point_config["token_hidden_size"]
            self.point_backbone = PointCloudEncoderSceneScript(input_channels, d_model, conv_layers, num_bins, token_hidden_size)
        else:
            from model.pcd_encoder import PointCloudEncoder
            self.point_backbone = PointCloudEncoder(config)

        self.full_tune_point_backbone = config.full_tune_point_backbone

        self.point_start_token_id = self.config.point_start_token_id
        self.point_end_token_id = self.config.point_end_token_id
        self.point_pad_token_id = self.config.point_pad_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def forward_point_cloud(self, coord, grid_coord, color, inverse, offset, feat, point_feat, point_feat_coord, min_extent, device, dtype):

        if self.config.point_backbone == "scenescript":
            # point cloud has shape (n_points, n_features)
            # find the points that have nan values
            self.point_backbone.to(torch.float32)
            nan_mask = torch.isnan(coord).any(dim=1)
            grid_coords = grid_coord[~nan_mask].contiguous().to(torch.int32)
            xyz = coord[~nan_mask]
            rgb = color[~nan_mask]
            feats = torch.cat([xyz, rgb], dim=1).contiguous().to(torch.float32)
            # feats = xyz.contiguous().to(torch.float32)
            pc_sparse_tensor = torchsparse.SparseTensor(coords=grid_coords, feats=feats)
            pc_sparse_tensor = sparse_collate([pc_sparse_tensor])  # batch_size = 1
            pc_sparse_tensor = pc_sparse_tensor.to(device)
            encoded_features = self.point_backbone(pc_sparse_tensor)
            return encoded_features["context"].to(dtype)
        
        # train mode
        if point_feat is not None and not self.full_tune_point_backbone:
            point_feat_nan_mask = torch.isnan(point_feat).any(dim=1)
            point_feat = point_feat[~point_feat_nan_mask]
            point_feat_coord = point_feat_coord[~point_feat_nan_mask]
            encoded_features = self.point_backbone.forward_feat(point_feat, point_feat_coord, min_extent)
            return encoded_features
        
        self.point_backbone.to(torch.float32)
        nan_mask = torch.isnan(coord).any(dim=1)
        coord = coord[~nan_mask]
        grid_coord = grid_coord[~nan_mask]
        color = color[~nan_mask]
        # inverse = inverse[~nan_mask]
        # offset = offset[~nan_mask]
        feat = feat[~nan_mask]
        point = {
            "coord": coord,
            "grid_coord": grid_coord,
            "color": color,
            # "inverse": inverse,
            "offset": offset,
            "feat": feat
        }

        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)

        encoded_features = self.point_backbone(point, min_extent)
        return encoded_features

    def set_point_backbone_dtype(self, dtype: torch.dtype):
        for param in self.point_backbone.parameters():
            param.data = param.data.to(dtype)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        coord: Optional[torch.Tensor] = None,
        grid_coord: Optional[torch.Tensor] = None,
        color: Optional[torch.Tensor] = None,
        inverse: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        feat: Optional[torch.Tensor] = None,
        point_feat: Optional[torch.Tensor] = None,
        point_feat_coord: Optional[torch.Tensor] = None,
        min_extent: Optional[torch.Tensor] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # compute point cloud embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if (
            self.point_backbone is not None
            and (input_ids.shape[1] != 1 or self.training)
            and coord is not None
        ):
            n_point_clouds = len(coord)
            point_features = []
            for i in range(n_point_clouds):  # * iterate over batch
                coord_i = coord[i]
                grid_coord_i = grid_coord[i]
                color_i = color[i]
                inverse_i = inverse[i]
                offset_i = offset[i]
                feat_i = feat[i]
                
                if point_feat is not None:
                    point_feat_i = point_feat[i]
                    point_feat_coord_i = point_feat_coord[i]
                else:
                    point_feat_i = None
                    point_feat_coord_i = None

                if min_extent is not None:
                    min_extent_i = min_extent[i]
                else:
                    min_extent_i = None
                point_feature = self.forward_point_cloud(
                    coord_i, grid_coord_i, color_i, inverse_i, offset_i, feat_i, point_feat_i, point_feat_coord_i, min_extent_i, inputs_embeds.device, inputs_embeds.dtype
                )
                point_features.append(point_feature)

            # Insert point cloud features into the input ids
            point_start_end_token_pos = []
            new_input_embeds = []
            new_attention_mask = []
            cur_point_idx = 0
            max_num_tokens = 0
            for cur_input_ids, cur_input_embeds, cur_attention_mask in zip(
                input_ids, inputs_embeds, attention_mask
            ):  # * input_ids: B, L; input_embeds: B, L, C
                cur_point_features = (
                    point_features[cur_point_idx]
                    .to(device=cur_input_embeds.device)
                    .squeeze(0)
                )
                num_patches = cur_point_features.shape[0]  # * number of point tokens
                num_point_start_tokens = (
                    (cur_input_ids == self.config.point_start_token_id).sum().item()
                )
                num_point_end_tokens = (
                    (cur_input_ids == self.config.point_end_token_id).sum().item()
                )
                # currently, we only support one point start and one point end token
                assert num_point_start_tokens == num_point_end_tokens == 1, (
                    "The number of point start tokens and point end tokens should be 1, "
                    f"but got {num_point_start_tokens} and {num_point_end_tokens}."
                )
                point_start_token_pos = torch.where(
                    cur_input_ids == self.config.point_start_token_id
                )[0][0]
                point_end_token_pos = torch.where(
                    cur_input_ids == self.config.point_end_token_id
                )[0][0]
                cur_new_input_embeds = torch.cat(
                    (
                        cur_input_embeds[: point_start_token_pos + 1],
                        cur_point_features,
                        cur_input_embeds[point_end_token_pos:],
                    ),
                    dim=0,
                )
                cur_new_attention_mask = torch.cat(
                    (
                        cur_attention_mask[: point_start_token_pos + 1],
                        torch.ones(num_patches, device=cur_attention_mask.device),
                        cur_attention_mask[point_end_token_pos:],
                    ),
                    dim=0,
                )

                cur_point_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
                new_attention_mask.append(cur_new_attention_mask)
                point_start_end_token_pos.append(
                    (point_start_token_pos, num_patches, point_end_token_pos)
                )
                if cur_new_input_embeds.shape[0] > max_num_tokens:
                    max_num_tokens = cur_new_input_embeds.shape[0]
            # pad the new input embeds and attention mask to the max dimension
            for i in range(len(new_input_embeds)):
                cur_input_embeds = new_input_embeds[i]
                last_row = cur_input_embeds[-1]
                padding = last_row.repeat(max_num_tokens - cur_input_embeds.shape[0], 1)
                new_input_embeds[i] = torch.cat([cur_input_embeds, padding], dim=0)

                cur_attention_mask = new_attention_mask[i]
                new_attention_mask[i] = F.pad(
                    cur_attention_mask,
                    (0, max_num_tokens - cur_attention_mask.shape[0]),
                    value=0,
                )
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)

            assert (
                attention_mask.shape[1] == inputs_embeds.shape[1]
            ), "The length of attention mask and inputs embeds should be the same"

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # prepare new labels
            new_labels = []
            max_num_tokens = logits.shape[1]
            for i in range(len(point_start_end_token_pos)):
                cur_labels = labels[i]
                (
                    cur_point_start_token_pos,
                    num_patches,
                    cur_point_end_token_pos,
                ) = point_start_end_token_pos[i]
                cur_new_labels = torch.cat(
                    (
                        cur_labels[: cur_point_start_token_pos + 1],
                        torch.full(
                            (num_patches,),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                        ),
                        cur_labels[cur_point_end_token_pos:],
                    ),
                    dim=0,
                )
                cur_new_labels = F.pad(
                    cur_new_labels,
                    (0, max_num_tokens - cur_new_labels.shape[0]),
                    value=IGNORE_INDEX,
                )
                new_labels.append(cur_new_labels)
            labels = torch.stack(new_labels, dim=0)

            assert (
                labels.shape[1] == logits.shape[1]
            ), "The length of labels and logits should be the same"

            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "coord": kwargs.get("coord"),
            "grid_coord": kwargs.get("grid_coord"),
            "color": kwargs.get("color"),
            "inverse": kwargs.get("inverse"),
            "offset": kwargs.get("offset"),
            "feat": kwargs.get("feat"),
            "min_extent": kwargs.get("min_extent")
        }

        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            
        return model_inputs


AutoConfig.register("spacelm_qwen", SpaceLMQwenConfig)
AutoModelForCausalLM.register(SpaceLMQwenConfig, SpaceLMQwenForCausalLM)
