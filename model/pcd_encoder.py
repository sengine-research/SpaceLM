import torch
import torch.nn as nn
from sonata.model import PointTransformerV3
import json
from torch.nn import MultiheadAttention
from torchsparse import SparseTensor
from layout.entity import NORMALIZATION_PRESET

class PointCloudEncoder(nn.Module):
    def __init__(self, config, num_bins=NORMALIZATION_PRESET["num_bins"], num_bands=10):
        super(PointCloudEncoder, self).__init__()
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,  # reduce patch size if necessary
        )
        sonata_model_config = "sonata/config.json"
        sonata_model_config = json.load(open(sonata_model_config))
        if custom_config is not None:
            for key, value in custom_config.items():
                sonata_model_config[key] = value
        self.encoder_model = PointTransformerV3(**sonata_model_config)

        embed_channels = sonata_model_config["enc_channels"][-1]

        self.fuse_coord = config.fuse_coord

        if config.use_point_attention:
            if config.fuse_coord:
                self.point_proj = nn.Linear(embed_channels + 63, config.hidden_size)
            else:
                self.point_proj = nn.Linear(embed_channels, config.hidden_size)
            self.point_attention = MultiheadAttention(embed_dim=config.hidden_size, num_heads=8)
        else:
            if config.fuse_coord:
                self.point_proj = nn.Linear(embed_channels + 63, config.hidden_size)
            else:
                self.point_proj = nn.Linear(embed_channels, config.hidden_size)
            self.point_attention = None
        self.num_bins = num_bins  # set the number of bins
        self.num_bands = num_bands  # set the number of bands

    @staticmethod
    def fourier_encode_vector(vec, num_bands=10, sample_rate=60):
        """Fourier encoding method"""
        b, n, d = vec.shape
        samples = torch.linspace(1, sample_rate/2, num_bands).to(vec.device) * torch.pi
        sines = torch.sin(samples[None, None, :, None] * vec[:, :, None, :])
        cosines = torch.cos(samples[None, None, :, None] * vec[:, :, None, :])
        
        encoding = torch.stack([sines, cosines], dim=3).reshape(b, n, 2*num_bands, d)
        encoding = torch.cat([vec[:, :, None, :], encoding], dim=2)
        return encoding.flatten(2)

    def forward_feat(self, point_cloud_feat, point_cloud_feat_coord, min_extent):
        if self.fuse_coord:
            encoded_coords = self.fourier_encode_vector(
                point_cloud_feat_coord.unsqueeze(0) - min_extent,  # add batch dimension
                num_bands=self.num_bands
            ).squeeze(0)  # [N, 63]
            fused_feat = torch.cat([point_cloud_feat, encoded_coords], dim=-1)
        else:
            fused_feat = point_cloud_feat
            
        point_cloud_feat = self.point_proj(fused_feat)

        if self.point_attention is not None:
            # 添加残差连接和层归一化
            residual = point_cloud_feat
            # 调整维度顺序适配多头注意力 (seq_len, batch_size, embed_dim)
            point_cloud_feat = point_cloud_feat.unsqueeze(0)  # 新增维度 [1, N, C]
            point_cloud_feat = point_cloud_feat.permute(1, 0, 2)
            point_cloud_feat, _ = self.point_attention(
                point_cloud_feat, 
                point_cloud_feat, 
                point_cloud_feat
            )
            # 恢复原始维度并添加残差
            point_cloud_feat = point_cloud_feat.permute(1, 0, 2).squeeze(0)  # 移除新增维度 [N, C]
            point_cloud_feat = point_cloud_feat + residual
            point_cloud_feat = nn.LayerNorm(point_cloud_feat.shape[-1]).to(point_cloud_feat.device)(point_cloud_feat)

        return point_cloud_feat

    def forward(self, point_cloud, min_extent):
        """
        point_cloud: dict,
            "coord": torch.Tensor,
            "color": torch.Tensor,
            "normal": torch.Tensor,
            "feat": torch.Tensor,
            "point_feat": torch.Tensor,
            "point_feat_coord": torch.Tensor,

        output: torch.Tensor,
            "feat": torch.Tensor, (grid_size, hidden_size)
        """
        point_cloud_output = self.encoder_model(point_cloud)

        point_cloud_feat = point_cloud_output["feat"]
        coord = point_cloud_output["coord"] - min_extent

        if self.fuse_coord:
            encoded_coords = self.fourier_encode_vector(
                coord.unsqueeze(0),  # add batch dimension
                num_bands=self.num_bands
            ).squeeze(0)

            # concatenate the point cloud feature and the Fourier encoded coordinates [B, N, 512][B, N, 63] -> [B, N, 575]
            fused_feat = torch.cat([point_cloud_feat, encoded_coords], dim=-1)
        else:
            fused_feat = point_cloud_feat

        point_cloud_feat = self.point_proj(fused_feat)

        if self.point_attention is not None:
            # 添加残差连接和层归一化
            residual = point_cloud_feat
            # 调整维度顺序适配多头注意力 (seq_len, batch_size, embed_dim)
            point_cloud_feat = point_cloud_feat.unsqueeze(0)  # 新增维度 [1, N, C]
            point_cloud_feat = point_cloud_feat.permute(1, 0, 2)
            point_cloud_feat, _ = self.point_attention(
                point_cloud_feat, 
                point_cloud_feat, 
                point_cloud_feat
            )
            # 恢复原始维度并添加残差
            point_cloud_feat = point_cloud_feat.permute(1, 0, 2).squeeze(0)  # 移除新增维度 [N, C]
            point_cloud_feat = point_cloud_feat + residual
            point_cloud_feat = nn.LayerNorm(point_cloud_feat.shape[-1]).to(point_cloud_feat.device)(point_cloud_feat)

        return point_cloud_feat



