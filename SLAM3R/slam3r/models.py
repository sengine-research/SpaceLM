# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Multiview DUSt3R model class
# support arbitray number of input views,:one reference view and serveral sources views
# --------------------------------------------------------

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from .pos_embed import get_2d_sincos_pos_embed, RoPE2D 
from .patch_embed import get_patch_embed

from .blocks import Block, Mlp, MultiviewDecoderBlock_max

from .heads import head_factory
from .heads.postprocess import reg_dense_conf

from .utils.device import MyNvtxRange
from .utils.misc import freeze_all_params, transpose_to_landscape

from huggingface_hub import PyTorchModelHubMixin

inf = float('inf')

       
class Multiview3D(nn.Module, PyTorchModelHubMixin):
    """Backbone of SLAM3R model, with the following components:
    - patch embeddings
    - positional embeddings
    - encoder and decoder 
    - downstream heads for 3D point and confidence map prediction
    """
    def __init__(self,
                 img_size=224,           # input image size
                 patch_size=16,          # patch_size 
                 enc_embed_dim=768,      # encoder feature dimension
                 enc_depth=12,           # encoder depth 
                 enc_num_heads=12,       # encoder number of heads in the transformer block 
                 dec_embed_dim=512,      # decoder feature dimension 
                 dec_depth=8,            # decoder depth 
                 dec_num_heads=16,       # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='RoPE100',     # positional embedding (either cosine or RoPE100)
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 need_encoder=True,      # whether to create the encoder, or omit it
                 mv_dec1 = "MultiviewDecoderBlock_max", # type of decoder block 
                 mv_dec2 = "MultiviewDecoderBlock_max",
                 enc_minibatch = 4,  # minibatch size for encoding multiple views
                 input_type = 'img',
                ):    

        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim)
        # positional embeddings in the encoder and decoder
        self._set_pos_embed(pos_embed, enc_embed_dim, dec_embed_dim, 
                            self.patch_embed.num_patches)
        # transformer for the encoder 
        self.need_encoder = need_encoder
        if need_encoder:
            self._set_encoder(enc_embed_dim, enc_depth, enc_num_heads, 
                              mlp_ratio, norm_layer)
        else:
            self.enc_norm = norm_layer(enc_embed_dim) 
        # transformer for the decoder
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, 
                          mlp_ratio, norm_layer, norm_im2_in_dec, 
                          mv_dec1=mv_dec1, mv_dec2=mv_dec2)
        # dust3r specific initialization
        self._set_downstream_head(output_mode, head_type, landscape_only, 
                                  depth_mode, conf_mode, patch_size, img_size)
        self.set_freeze(freeze)
        
        self.enc_minibatch= enc_minibatch
        self.input_type = input_type

    def _set_patch_embed(self, patch_embed_cls, img_size=224, patch_size=16, 
                         enc_embed_dim=768):
        self.patch_size = patch_size
        self.patch_embed = get_patch_embed(patch_embed_cls, img_size, 
                                           patch_size, enc_embed_dim)
        
    def _set_encoder(self, enc_embed_dim, enc_depth, enc_num_heads, 
                     mlp_ratio, norm_layer):
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)    
    
    def _set_pos_embed(self, pos_embed, enc_embed_dim, 
                       dec_embed_dim, num_patches):
        self.pos_embed = pos_embed
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, int(num_patches**.5), n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, int(num_patches**.5), n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, 
                     dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec, 
                     mv_dec1, mv_dec2):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the two ssymmetric decoders 
        self.mv_dec_blocks1 = nn.ModuleList([
            eval(mv_dec1)(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        self.mv_dec_blocks2 = nn.ModuleList([
            eval(mv_dec2)(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        self.mv_dec1_str = mv_dec1 
        self.mv_dec2_str = mv_dec2
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_downstream_head(self, output_mode, head_type, landscape_only, 
                             depth_mode, conf_mode, patch_size, img_size):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def load_state_dict(self, ckpt, ckpt_type="slam3r", **kw):
        if not self.need_encoder:
            ckpt_wo_enc = {k: v for k, v in ckpt.items() if not k.startswith('enc_blocks')}
            ckpt = ckpt_wo_enc
            
        # if already in the slam3r format, just load it
        if ckpt_type == "slam3r":
            assert any(k.startswith('mv_dec_blocks') for k in ckpt)
            return super().load_state_dict(ckpt, **kw)
        
        # if in croco format, convert to dust3r format first
        if ckpt_type == "croco":
            assert not any(k.startswith('dec_blocks2') for k in ckpt)
            dust3r_ckpt = dict(ckpt)
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    dust3r_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        elif ckpt_type == "dust3r":
            assert any(k.startswith('dec_blocks2') for k in ckpt)
            dust3r_ckpt = dict(ckpt)
        else:
            raise ValueError(f"Unknown ckpt_type {ckpt_type}")
        
        # convert from dust3r format to slam3r format
        slam3r_ckpt = deepcopy(dust3r_ckpt)
        for key, value in dust3r_ckpt.items():
            if key.startswith('dec_blocks2'):
                slam3r_ckpt[key.replace('dec_blocks2', 'mv_dec_blocks2')] = value
                del slam3r_ckpt[key]
            elif key.startswith('dec_blocks'):
                slam3r_ckpt[key.replace('dec_blocks', 'mv_dec_blocks1')] = value
                del slam3r_ckpt[key]    
        
        # now load the converted ckpt in slam3r format
        return super().load_state_dict(slam3r_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        if freeze == 'none':
            return
        if self.need_encoder and freeze == 'encoder':
            freeze_all_params([self.patch_embed, self.enc_blocks])
        elif freeze == 'corr_score_head_only':
            for param in self.parameters():
                param.requires_grad = False
            for param in self.corr_score_proj.parameters():
                param.requires_grad = True
            for param in self.corr_score_norm.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError(f"freeze={freeze} not implemented")

    def _encode_image(self, image, true_shape, normalize=True):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # add positional embedding without cls token
        if(self.pos_embed != 'cosine'):
            assert self.enc_pos_embed is None 
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)
        if normalize:
            x = self.enc_norm(x)
        return x, pos, None
    
    def _encode_multiview(self, views:list, view_batchsize=None, normalize=True, silent=True):
        """encode multiple views in a minibatch
        For example, if there are 6 views, and view_batchsize=3, 
        then the first 3 views are encoded together, and the last 3 views are encoded together.
        
        Warnning!!: it only works when shapes of views in each minibatch is the same
        
        Args:
            views: list of dictionaries, each containing the input view
            view_batchsize: number of views to encode in a single batch
            normalize: whether to normalize the output token with self.enc_norm. Specifically,
                if the input view['img_tokens'] are already normalized, set it to False.
        """
        input_type = self.input_type if self.input_type in views[0] else 'img'
        # if img tokens output by encoder are already precomputed and saved, just return them
        if "img_tokens" in views[0]:
            res_shapes, res_feats, res_poses = [], [], []
            for i, view in enumerate(views):
                tokens = self.enc_norm(view["img_tokens"]) if normalize else view["img_tokens"]
                # res_feats.append(view["img_tokens"]) # (B, S, D)
                res_feats.append(tokens) # (B, S, D)
                res_shapes.append(view['true_shape']) # (B, 2)
                if "img_pos" in view:
                    res_poses.append(view["img_pos"]) #(B, S, 2)
                else:
                    img = view[input_type]
                    res_poses.append(self.position_getter(B, img.size(2), img.size(3), img.device))
            return res_shapes, res_feats, res_poses
                
                
        if view_batchsize is None: 
            view_batchsize = self.enc_minibatch

        B = views[0][input_type].shape[0]
        res_shapes, res_feats, res_poses = [],[],[]
        minibatch_num = (len(views)-1)//view_batchsize+1
        
        with tqdm(total=len(views), disable=silent, desc="encoding images") as pbar:   
            for i in range(0,minibatch_num):
                batch_views = views[i*view_batchsize:(i+1)*view_batchsize]
                batch_imgs = [view[input_type] for view in batch_views]
                batch_shapes = [view.get('true_shape', 
                                        torch.tensor(view[input_type].shape[-2:])[None].repeat(B, 1))
                                for view in batch_views]  # vb*(B,2)
                res_shapes += batch_shapes
                batch_imgs = torch.cat(batch_imgs, dim=0)  # (vb*B, 3, H, W)
                batch_shapes = torch.cat(batch_shapes, dim=0)  # (vb*B, 2)
                out, pos, _ = self._encode_image(batch_imgs,batch_shapes,normalize) # (vb*B, S, D), (vb*B, S, 2)
                res_feats += out.chunk(len(batch_views), dim=0) # V*(B, S, D)
                res_poses += pos.chunk(len(batch_views), dim=0) # V*(B, S, 2)
               
                pbar.update(len(batch_views))
                
        return res_shapes, res_feats, res_poses    

    def _decode_multiview(self, ref_feats:torch.Tensor, src_feats:torch.Tensor, 
                          ref_poses:torch.Tensor, src_poses:torch.Tensor, 
                          ref_pes:torch.Tensor|None, src_pes:torch.Tensor|None):
        """exchange information between reference and source views in the decoder

        About naming convention:
            reference views: views that define the coordinate system.
            source views: views that need to be transformed to the coordinate system of the reference views.

        Args:
            ref_feats (R, B, S, D_enc): img tokens of reference views 
            src_feats (V-R, B, S, D_enc): img tokens of source views
            ref_poses (R, B, S, 2): positions of tokens of reference views
            src_poses (V-R, B, S, 2): positions of tokens of source views
            ref_pes (R, B, S, D_dec): pointmap tokens of reference views
            src_pes:(V-R, B, S, D_dec): pointmap tokens of source views
        
        Returns:
            final_refs: list of R*(B, S, D_dec)
            final_srcs: list of (V-R)*(B, S, D_dec)
        """
        # R: number of reference views
        # V: total number of reference and source views
        # S: number of tokens
        num_ref = ref_feats.shape[0]
        num_src = src_feats.shape[0]
        
        final_refs = [ref_feats]  # before projection
        final_srcs = [src_feats]
        # project to decoder dim
        final_refs.append(self.decoder_embed(ref_feats)) 
        final_srcs.append(self.decoder_embed(src_feats))
        
        # define how each views interact with each other
        # here we use a simple way: ref views and src views exchange information bidirectionally
        # for more detail, please refer to the class MultiviewDecoderBlock_max in blocks/multiview_blocks.py
        src_rel_ids_d = torch.arange(num_ref, device=final_refs[0].device, dtype=torch.long)
        src_rel_ids_d = src_rel_ids_d[None].expand(src_feats.shape[0], -1).reshape(-1) # (V-R * R)
        ref_rel_ids_d = torch.arange(num_src, device=final_refs[0].device, dtype=torch.long)
        ref_rel_ids_d = ref_rel_ids_d[None].expand(ref_feats.shape[0], -1).reshape(-1) # (R * V-R)
        
        for i in range(self.dec_depth):
            # (R, B, S, D),  (V-R, B, S, D)
            # add pointmap tokens if available(used in Local2WorldModel)
            ref_inputs = final_refs[-1] + ref_pes if ref_pes is not None else final_refs[-1]
            src_inputs = final_srcs[-1] + src_pes if src_pes is not None else final_srcs[-1]

            ref_blk:MultiviewDecoderBlock_max = self.mv_dec_blocks1[i]
            src_blk = self.mv_dec_blocks2[i]
            # reference image side
            ref_outputs = ref_blk(ref_inputs, src_inputs, 
                                     ref_poses, src_poses, 
                                     ref_rel_ids_d, num_src) # (R, B, S, D)
            # source image side
            src_outputs = src_blk(src_inputs, ref_inputs, 
                                     src_poses, ref_poses, 
                                     src_rel_ids_d, num_ref) # (V-R, B, S, D)
            # store the result
            final_refs.append(ref_outputs)
            final_srcs.append(src_outputs)

        # normalize last output
        del final_srcs[1]  # duplicate with final_output[0]
        del final_refs[1]
        final_refs[-1] = self.dec_norm(final_refs[-1])  #(R, B, S, D)
        final_srcs[-1] = self.dec_norm(final_srcs[-1])

        for i in range(len(final_refs)):
            R, B, S, D = final_refs[i].shape
            assert R == num_ref
            final_refs[i] = final_refs[i].reshape(R*B, S, D)  #(R*B, S, D)
            final_srcs[i] = final_srcs[i].reshape(num_src*B, S, D)  #((V-R)*B, S, D/D')

        return final_refs, final_srcs  #list: [depth*(R*B, S, D/D')], [depth*((V-R)*B, S, D/D')]

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def split_stack_ref_src(self, data:list, ref_ids:list, src_ids:list, stack_up=True):
        """Split the list containing data of different views into 2 lists by ref_ids 
        Args:
            data: a list of length num_views, 
                containing data(enc_feat, dec_feat pos or pes) of all views
            ref_ids: list of indices of reference views
            src_ids: list of indices of source views
            stack_up: whether to stack up the result
        """
        ref_data = [data[i] for i in ref_ids]
        src_data = [data[i] for i in src_ids]
        if stack_up:
            ref_data = torch.stack(ref_data, dim=0) # (R, B, S, D)
            src_data = torch.stack(src_data, dim=0) # (V-R, B, S, D)
        return ref_data, src_data

    def forward():
        raise NotImplementedError
        
        
class Image2PointsModel(Multiview3D):
    """Image2Point Model, with a retrieval module attached to it.
    Take multiple views as input, and recover 3D pointmaps directly. 
    All the pointmaps are in the coordinate system of a designated view.
    """
    def __init__(self, corr_depth=2, **args):
        super().__init__( **args)
        self.corr_score_depth = corr_depth
        self.corr_score_norm = nn.LayerNorm(self.dec_embed_dim)
        self.corr_score_proj = Mlp(in_features=self.dec_embed_dim, out_features=1)

    def get_corr_score(self, views, ref_id, depth=-1):
        """Get the correlation score between the reference view and each source view
        Use the first serveral decoder blocks, followed by a layernorm and a mlp. 
        Modified from _decode_multiview() function.
        
        Args:
            ref_id: index of the reference view
            depth: number of decoder blocks to use. If -1, use self.corr_score_depth
        
        Returns:
            patch_corr_scores: correlation scores between the reference view 
            and each source view tokens
        
        """
        if depth < 0:
            depth = self.corr_score_depth
        shapes, enc_feats, poses = self._encode_multiview(views)
        assert ref_id < len(views) and ref_id >= 0
        src_ids = [i for i in range(len(views)) if i != ref_id]
        ref_ids = [ref_id] 
        
        # select and stacck up ref and src elements. R=1
        ref_feats, src_feats = self.split_stack_ref_src(enc_feats, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_poses, src_poses = self.split_stack_ref_src(poses, ref_ids, src_ids)  # (R, B, S, 2), (V-R, B, S, 2)
        
        num_ref = ref_feats.shape[0]
        num_src = src_feats.shape[0]
        num_views = num_ref + num_src
        
        final_refs = [ref_feats]  # before projection
        final_srcs = [src_feats]
        # project to decoder dim
        final_refs.append(self.decoder_embed(ref_feats))
        final_srcs.append(self.decoder_embed(src_feats))
        
        ref_rel_ids_d = torch.arange(num_views-1, device=final_refs[0].device, dtype=torch.long)
        src_rel_ids_d = torch.zeros(num_views-1, device=final_srcs[0].device, dtype=torch.long)  
        
        for i in range(depth):
            ref_input = final_refs[-1]  # (1, B, S, D)
            src_inputs = final_srcs[-1]  # (V-1, B, S, D)

            ref_blk = self.mv_dec_blocks1[i]
            src_blk = self.mv_dec_blocks2[i]
            # reference image side
            if i < depth-1:
                ref_outputs = ref_blk(ref_input, src_inputs, 
                                        ref_poses, src_poses, 
                                        ref_rel_ids_d, num_views-1)
                final_refs.append(ref_outputs)
            # source image side
            src_outputs = src_blk(src_inputs, ref_input, 
                                     src_poses, ref_poses, 
                                     src_rel_ids_d, 1)
            final_srcs.append(src_outputs)

        dec_feats_shallow = final_srcs[-1] #output of the depth_th block (src, B, S, D)
        dec_feats_shallow = self.corr_score_norm(dec_feats_shallow)
        patch_corr_scores = self.corr_score_proj(dec_feats_shallow)[..., 0]  # (src, B, S)
        patch_corr_scores = reg_dense_conf(patch_corr_scores, mode=self.conf_mode)  # (src, B, S)   

        return patch_corr_scores
    
    def forward(self, views:list, ref_id, return_corr_score=False):
        """ 
        naming convention:
            reference views: views that define the coordinate system.
            source views: views that need to be transformed to the coordinate system of the reference views.
        Args:
            views: list of dictionaries, each containing:
                - 'img': input image tensor (B, 3, H, W) or 'img_tokens': image tokens (B, S, D)
                - 'true_shape': true shape of the input image (B, 2)
            ref_id: index of the reference view in input view list
        """
        # decide which views are reference views and which are source views
        assert ref_id < len(views) and ref_id >= 0
        src_ids = [i for i in range(len(views)) if i != ref_id]
        ref_ids = [ref_id] 
            
        with MyNvtxRange('encode'):
            shapes, enc_feats, poses = self._encode_multiview(views)
        
        # select and stacck up ref and src elements. R=1 in the I2P model.
        ref_feats, src_feats = self.split_stack_ref_src(enc_feats, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_poses, src_poses = self.split_stack_ref_src(poses, ref_ids, src_ids)  # (R, B, S, 2), (V-R, B, S, 2)
        ref_shapes, src_shapes = self.split_stack_ref_src(shapes, ref_ids, src_ids) # (R, B, 2), (V-R, B, 2)
        
        # let all reference view and source view tokens interact with each other
        with MyNvtxRange('decode'):
            dec_feats_ref, dec_feats_src = self._decode_multiview(ref_feats, src_feats, 
                                                                  ref_poses, src_poses,
                                                                  None,None) 
        # print(len(dec_feats_ref), len(dec_feats_src)) #list: [depth*(R*B, S, D/D')], [depth*((V-R)*B, S, D/D')]
        
        with MyNvtxRange('head'):
            with torch.cuda.amp.autocast(enabled=False):
                # conf: ((V-R)*B, H, W)  pts3d: ((V-R)*B, H, W, 3)
                res_ref = self._downstream_head(1, [tok.float() for tok in dec_feats_ref], ref_shapes.reshape(-1,2))
                res_src = self._downstream_head(2, [tok.float() for tok in dec_feats_src], src_shapes.reshape(-1,2))
                # print(res_ref['pts3d'].shape, res_src['pts3d'].shape, res_ref['conf'].shape, res_src['conf'].shape)
        
        if return_corr_score:
            dec_feats_shallow = dec_feats_src[self.corr_score_depth] # (src*B, S, D)
            dec_feats_shallow = self.corr_score_norm(dec_feats_shallow)
            patch_corr_scores = self.corr_score_proj(dec_feats_shallow)[..., 0]  # (src*B, S)
            # patch_confs = reg_dense_conf(patch_confs, mode=self.conf_mode)  # (src*B, S)        
        
        # split the results back to each view
        results = [] 
        B = res_ref['pts3d'].shape[0]  #因为这里num_ref=1
        for id in range(len(views)):
            res = {}
            if id in ref_ids:
                rel_id = ref_ids.index(id)
                res['pts3d'] = res_ref['pts3d'][rel_id*B:(rel_id+1)*B]
                res['conf'] = res_ref['conf'][rel_id*B:(rel_id+1)*B]
            else:
                rel_id = src_ids.index(id)
                res['pts3d_in_other_view'] = res_src['pts3d'][rel_id*B:(rel_id+1)*B]
                res['conf'] = res_src['conf'][rel_id*B:(rel_id+1)*B] # (B, H, W)
                if return_corr_score:
                    res['pseudo_conf'] = patch_corr_scores[rel_id*B:(rel_id+1)*B] # (B, S)
            results.append(res)
        return results
    

class Local2WorldModel(Multiview3D):
    """Local2World Model
    Take arbitrary number of refernce views('scene frames' in paper) 
    and source views('keyframes' in paper) as input
    1. refine the input 3D pointmaps of the reference views
    2. transform the input 3D pointmaps of the source views to the coordinate system of the reference views
    """
    def __init__(self, **args):
        super().__init__(**args)
        self.dec_embed_dim = self.decoder_embed.out_features
        self.void_pe_token = nn.Parameter(torch.randn(1,1,self.dec_embed_dim), requires_grad=True)
        self.set_pointmap_embedder()

    def set_pointmap_embedder(self):
        self.ponit_embedder = nn.Conv2d(3, self.dec_embed_dim, 
                                        kernel_size=self.patch_size, stride=self.patch_size)
        
    def get_pe(self, views, ref_ids):
        """embed 3D points with a single conv layer
        landscape_only not tested yet"""
        pes = []
        for id, view in enumerate(views):
            if id in ref_ids:
                pos = view['pts3d_world']
            else:
                pos = view['pts3d_cam']
                
            if pos.shape[-1] == 3:
                pos = pos.permute(0,3, 1, 2)
                
            pts_embedding = self.ponit_embedder(pos).permute(0,2,3,1).reshape(pos.shape[0], -1, self.dec_embed_dim) # (B, S, D)
            if 'patch_mask' in view:
                patch_mask = view['patch_mask'].reshape(pos.shape[0], -1, 1) # (B, S, 1)
                pts_embedding = pts_embedding*(~patch_mask) + self.void_pe_token*patch_mask
                
            pes.append(pts_embedding)
        
        return pes
    
    def forward(self, views:list, ref_ids = 0):
        """ 
        naming convention:
            reference views: views that define the coordinate system.
            source views: views that need to be transformed to the coordinate system of the reference views.
        
        Args:
            views: list of dictionaries, each containing:
                    - 'img': input image tensor (B, 3, H, W) or 'img_tokens': image tokens (B, S, D)
                    - 'true_shape': true shape of the input image (B, 2)
                    - 'pts3d_world' (reference view only): 3D pointmaps in the world coordinate system (B, H, W, 3)
                    - 'pts3d_cam' (source view only): 3D pointmaps in the camera coordinate system (B, H, W, 3)
            ref_ids: indexes of the reference views in the input view list
        """
        # decide which views are reference views and which are source views
        if isinstance(ref_ids, int):
            ref_ids = [ref_ids]
        for ref_id in ref_ids:
            assert ref_id < len(views) and ref_id >= 0
        src_ids = [i for i in range(len(views)) if i not in ref_ids]            

        # #feat: B x S x D  pos: B x S x 2
        with MyNvtxRange('encode'):
            shapes, enc_feats, poses = self._encode_multiview(views)
            pes = self.get_pe(views, ref_ids=ref_ids)
        
        # select and stacck up ref and src elements
        ref_feats, src_feats = self.split_stack_ref_src(enc_feats, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_poses, src_poses = self.split_stack_ref_src(poses, ref_ids, src_ids)  # (R, B, S, 2), (V-R, B, S, 2)
        ref_pes, src_pes = self.split_stack_ref_src(pes, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_shapes, src_shapes = self.split_stack_ref_src(shapes, ref_ids, src_ids) # (R, B, 2), (V-R, B, 2)
        
        # combine all ref images into object-centric representation
        with MyNvtxRange('decode'):
            dec_feats_ref, dec_feats_src = self._decode_multiview(ref_feats, src_feats, 
                                                                  ref_poses, src_poses, 
                                                                  ref_pes, src_pes)
        
        with MyNvtxRange('head'):
            with torch.cuda.amp.autocast(enabled=False):
                # conf: ((V-R)*B, H, W)  pts3d: ((V-R)*B, H, W, 3)
                res_ref = self._downstream_head(1, [tok.float() for tok in dec_feats_ref], ref_shapes)
                res_src = self._downstream_head(2, [tok.float() for tok in dec_feats_src], src_shapes)
                # print(res_ref['pts3d'].shape, res_src['pts3d'].shape, res_ref['conf'].shape, res_src['conf'].shape)
        
        # split the results back to each view
        results = [] 
        B = res_ref['pts3d'].shape[0] // len(ref_ids)
        for id in range(len(views)):
            res = {}
            if id in ref_ids:
                rel_id = ref_ids.index(id)
                res['pts3d'] = res_ref['pts3d'][rel_id*B:(rel_id+1)*B]
                res['conf'] = res_ref['conf'][rel_id*B:(rel_id+1)*B]
            else:
                rel_id = src_ids.index(id)
                res['pts3d_in_other_view'] = res_src['pts3d'][rel_id*B:(rel_id+1)*B]
                res['conf'] = res_src['conf'][rel_id*B:(rel_id+1)*B]
            results.append(res)
        return results