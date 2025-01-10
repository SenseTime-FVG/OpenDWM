import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Optional, Any, List
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero
from dwm.models.bevw_vae_mm_model import VAEBevMultiModality
from diffusers.models.attention import BasicTransformerBlock
import pdb

ID_TO_ACTION_MAP = {
    0: "turn left",
    1: "turn right",
    2: "speed up",
    3: "slow down",
    4: "keep normal",
}
ACTION_TO_ID_MAP = {v: k for k, v in ID_TO_ACTION_MAP.items()}

# helper function to load testing data
def load_dict_items(load_dir="/mnt/storage/user/liuyichen/repo/DWM/scripts/lyc_noupload/test_data"):
    import numpy as np
    """
    Load items from files, converting NumPy arrays to PyTorch tensors.
    Handles nested directory structures created by save_dict_items.
    
    Args:
    load_dir (str): The directory to load the items from.
    
    Returns:
    dict: A dictionary containing the loaded items.
    """
    loaded_dict = {}
    
    for item in os.listdir(load_dir):
        item_path = os.path.join(load_dir, item)
        
        if item.endswith('.npy'):
            # Single tensor
            key = item[:-4]  # Remove .npy extension
            loaded_dict[key] = torch.from_numpy(np.load(item_path))
        
        elif os.path.isdir(item_path):
            # List of tensors or list of lists of tensors
            sub_items = os.listdir(item_path)
            if any(subitem.endswith('.npy') for subitem in sub_items):
                # List of tensors
                tensor_list = []
                for i in range(len(sub_items)):
                    file_path = os.path.join(item_path, f"{i}.npy")
                    if os.path.exists(file_path):
                        tensor_list.append(torch.from_numpy(np.load(file_path)))
                loaded_dict[item] = tensor_list
            else:
                # List of lists of tensors
                nested_tensor_list = []
                for subdir in sorted(sub_items):
                    subdir_path = os.path.join(item_path, subdir)
                    if os.path.isdir(subdir_path):
                        sub_tensor_list = []
                        for j in range(len(os.listdir(subdir_path))):
                            file_path = os.path.join(subdir_path, f"{j}.npy")
                            if os.path.exists(file_path):
                                sub_tensor_list.append(torch.from_numpy(np.load(file_path)))
                        nested_tensor_list.append(sub_tensor_list)
                loaded_dict[item] = nested_tensor_list
    
    print(f"All items loaded from {load_dir}")
    return loaded_dict

class AdaLN(nn.Module):
    def __init__(self, dim, num_embeds_ada_norm):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.emb_proj = nn.Linear(num_embeds_ada_norm, dim * 3)

    def forward(self, emb):
        embed = self.emb_proj(emb).unsqueeze(1)
        scale, shift, gate = embed.chunk(3, dim=2)
        x = self.norm(x)
        return x * (1 + scale) + shift, gate  # Return gate as well

# class BasicTransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, dropout=0., attention_dropout=0., num_embeds_ada_norm=None, is_temporal=False):
#         super().__init__()
#         self.norm1 = AdaLN(dim, num_embeds_ada_norm)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attention_dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, condition, causal_mask=None):
#         normed_x, attn_gate = self.norm1(x, condition)
#         attn_output = self.attn(normed_x, normed_x, normed_x, attn_mask=causal_mask)[0]
#         x = x + attn_output * attn_gate
#         normed_x = self.norm2(x)
#         x = x + self.mlp(normed_x)
#         return x



class DiffusionBevMultiModality(nn.Module):
    def __init__(
        self,
        action_threshold: List[float] = [15 / 90 * torch.pi, 0.1, 0.1],
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
        input_resolution: List[int] = [80, 80],
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        video_length: int = 16,
    ):
        super(DiffusionBevMultiModality, self).__init__()
        self.action_threshold = action_threshold
        self.action_condition_embedding = nn.Embedding(len(ID_TO_ACTION_MAP), 512)
        self.video_length = video_length
        inner_dim = num_attention_heads * attention_head_dim

        # Spatial transformer blocks
        self.spatial_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    final_dropout=dropout,
                    attention_bias = True,
                    upcast_attention = True,
                    num_embeds_ada_norm=len(ID_TO_ACTION_MAP), # num of embeddings for action condition
                    norm_type = "ada_norm_zero"
                )
                for _ in range(num_layers)
            ]
        )

        # Temporal transformer blocks
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    final_dropout=dropout,
                    attention_bias = True,
                    upcast_attention = True,
                    num_embeds_ada_norm=len(ID_TO_ACTION_MAP),  # num of embeddings for action condition
                    norm_type = "ada_norm_zero"
                )
                for _ in range(num_layers)
            ]
        )

        # Input and output projections
        self.input_proj = nn.Linear(in_channels, inner_dim)
        self.output_norm = AdaLayerNorm(inner_dim)
        self.output_proj = nn.Linear(inner_dim, in_channels)

        # Temporal positional embedding
        temp_pos_embed = self.get_1d_sincos_temp_embed(inner_dim, video_length)
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)

        # Add 2D positional embedding
        pos_embed_2d = self.get_2d_sincos_pos_embed(inner_dim, input_resolution)
        self.register_buffer("pos_embed_2d", torch.from_numpy(pos_embed_2d).float().unsqueeze(0), persistent=False)
        self.use_checkpoint = use_checkpoint

    def get_1d_sincos_temp_embed(self, embed_dim, length):
        # To des
        pos = torch.arange(0, length).unsqueeze(1)
        return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        # x_grid = torch.arange(0, grid_size[0])
        # y_grid = torch.arange(0, grid_size[1])
        # x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
        # x_grid = x_grid.flatten(0, 1).unsqueeze(1)
        # y_grid = y_grid.flatten(0, 1).unsqueeze(1)
        pos_embed_2d = get_2d_sincos_pos_embed(embed_dim, grid_size)
        return pos_embed_2d

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        action_condition: Optional[torch.Tensor] = None,
        ego_transforms: Optional[torch.Tensor] = None,
        enable_temporal_attentions: bool = True,
    ):
        batch_size, frame, c, h, w = hidden_states.shape

        # Project input
        hidden_states = rearrange(hidden_states, 'b f c h w -> (b f h w) c')        
        hidden_states = self.input_proj(hidden_states)

        # Add 2D positional embedding
        hidden_states = rearrange(hidden_states, '(b f h w) c -> (b f) (h w) c', b=batch_size, f=frame, h=h, w=w)
        hidden_states = hidden_states + self.pos_embed_2d
        # hidden_states = rearrange(hidden_states, '(b f) h w c -> (b f) (h w) c', b=batch_size, f=frame, h=h, w=w)

        # Prepare action condition       
        if action_condition is None and ego_transforms is not None:
            action_condition = DiffusionBevMultiModality.get_action_condition(ego_transforms)

        # Create causal mask for temporal attention
        attention_mask = torch.triu(torch.ones(frame, frame, device=hidden_states.device), diagonal=1).bool().logical_not()
        temporal_attention_mask = attention_mask.unsqueeze(0).expand(batch_size * h * w, -1, -1)
        

        # Iteratively apply spatial and temporal blocks
        for spatial_block, temporal_block in zip(self.spatial_transformer_blocks, self.temporal_transformer_blocks):
            if enable_temporal_attentions:
                hidden_states = rearrange(hidden_states, '(b f) (h w) c -> (b h w) f c', b=batch_size, f=frame, h=h, w=w)
                expanded_timestep = timestep.unsqueeze(1).expand(-1, h * w).flatten(0, 1)
                import pdb; pdb.set_trace()
                if self.training and self.use_checkpoint:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        temporal_block,
                        hidden_states,
                        temporal_attention_mask,
                        None, # encoder_hidden_states: Optional[torch.FloatTensor] = None,
                        None, # encoder_attention_mask: Optional[torch.FloatTensor] = None,
                        expanded_timestep, # timestep: Optional[torch.LongTensor] = None,
                        None, # cross_attention_kwargs: Dict[str, Any] = None,
                        action_condition, # class_labels: Optional[torch.LongTensor] = None,
                        None,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = temporal_block(hidden_states, 
                                                   class_labels = action_condition, 
                                                   attention_mask=attention_mask,
                                                   timestep = expanded_timestep)
                hidden_states = rearrange(hidden_states, '(b h w) f c -> (b f) (h w) c', b=batch_size, h=h, w=w)
            # Spatial block
            expanded_timestep = timestep.unsqueeze(1).expand(-1, frame).flatten(0, 1)
            if self.training and self.use_checkpoint:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    None, # attention_mask
                    None, # encoder_hidden_states
                    None, # encoder_attention_mask
                    expanded_timestep, # timestep
                    None, # cross_attention_kwargs
                    action_condition, # class_labels
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = spatial_block(hidden_states, 
                                             class_labels = action_condition,
                                             timestep = expanded_timestep)

        # Project output
        hidden_states = rearrange(hidden_states, '(b f) (h w) c -> (b c h w) c', b=batch_size, f=frame, h=h, w=w)
        shift_hidden_states, gate = self.output_norm(hidden_states)
        hidden_states = hidden_states * gate + self.output_proj(shift_hidden_states)
        hidden_states = rearrange(hidden_states, '(b c h w) c -> b f c h w', b=batch_size, f=frame, h=h, w=w)

        return hidden_states
    @staticmethod
    def get_action_condition(ego_transform, turn_threshold= 15 / 90 * torch.pi, speed_up_threshold=0.5, slow_down_threshold=0.5):
        """
        Determine the actions of the car based on the ego_transform tensor.
        
        Args:
            ego_transform: [batch_size, seq_len, 4, 4] - Representing the transform 
                        from the car coordinate to the world coordinate.
            turn_threshold (float): Threshold for determining left/right turns.
            speed_threshold (float): Threshold for determining speed up/slow down.
        
        Returns:
            list: List of actions for each timestamp.
        """
        bs, sequence_length = ego_transform.shape[:2]
        actions = []

        for i in range(1, sequence_length):
            # Extract the rotation and translation components
            prev_rotation = ego_transform[:, i-1, :3, :3]
            curr_rotation = ego_transform[:, i, :3, :3]
            prev_translation = ego_transform[:, i-1, :3, 3]
            curr_translation = ego_transform[:, i, :3, 3]
            if i > 1:
                prev_prev_translation = ego_transform[:, i-2, :3, 3]
            # Calculate the change in rotation (yaw). We can ignore the pitch and roll rotation.
            # Turn left: yaw_change < 0, Turn right: yaw_change > 0
            rotation_change = torch.atan2(curr_rotation[:, 1, 0], curr_rotation[:, 0, 0]) - \
                            torch.atan2(prev_rotation[:, 1, 0], prev_rotation[:, 0, 0])
            
            # Calculate the change in position (speed)
            speed = torch.norm(curr_translation - prev_translation)
            if i > 1:
                prev_speed = torch.norm(prev_prev_translation - prev_translation)
                speed_change = (speed - prev_speed) / prev_speed

            # Determine the action based on thresholds
            # Determine the action based on thresholds using PyTorch operations
            action = torch.full((bs,), 4, dtype=torch.long, device=ego_transform.device)  # Default action: keep normal
            action = torch.where(rotation_change > turn_threshold, torch.tensor(0), action) # turn left
            action = torch.where(rotation_change < -turn_threshold, torch.tensor(1), action) # turn right
            
            if i > 1:
                action = torch.where((speed_change > speed_up_threshold) & (action == 4), torch.tensor(2), action) # speed up
                action = torch.where((speed_change < -slow_down_threshold) & (action == 4), torch.tensor(3), action) # slow down
            actions.append(action.unsqueeze(1))
        return torch.cat(actions, dim=1)

# Example usage:
if __name__ == "__main__":
    batch = load_dict_items("/mnt/storage/user/liuyichen/repo/DWM/scripts/lyc_noupload/test_data")
    from dwm.functional import make_homogeneous_vector
    def get_points(batch, device):
            return [
                [
                    (make_homogeneous_vector(
                        p_j.to(device)) @ t_j.permute(1, 0))[:, :3]
                    for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
                ]
                for p_i, t_i in zip(
                    batch["lidar_points"],
                    batch["lidar_transforms"].to(device))
            ]
    points = get_points(
            batch, 'cuda')
    ray_cast_center = [
                1.0,
                0.0,
                2.0
            ]
    batch_size = len(batch["lidar_points"])
    ray_cast_center = torch.tensor([ray_cast_center])\
                    .repeat(batch_size, 1)
    from dwm.models.voxelizer import Voxelizer
    from dwm.models.bevw_vae_mm_model import SwimTransformerEncoder, DeformableAttention, BEVDecoder, ConvDecoder, VAEBevMultiModality

    # Set up Voxelizer
    voxelizer = Voxelizer(
        x_min=-50.0,
        x_max=50.0,
        y_min=-50.0,
        y_max=50.0,
        z_min=-3.0,
        z_max=5.0,
        step=0.15625,
        z_step=0.125
    )

    # Set up Image Encoder
    img_encoder = SwimTransformerEncoder(
        img_size=[256, 448],
        out_channels=512,
        in_channels=3,
        embed_dim=512,
        num_heads=8,
        upcast=True,
        flatten=False,
        use_checkpoint=True
    )

    # Set up LiDAR Encoder
    lidar_encoder = SwimTransformerEncoder(
        img_size=[640, 640],
        out_channels=512,
        in_channels=64,
        embed_dim=256,
        num_heads=8,
        upcast=True,
        use_checkpoint=True
    )

    # Set up Deformable Transformer
    deformable_transformer = DeformableAttention(
        in_channels=512,
        out_channels=512,
        num_heads=8
    )

    # Set up Image Decoder
    img_decoder = ConvDecoder(
        input_size=[32, 56],
        in_channels=64,
        embed_dim=256,
        out_channels=3,
        depth=[4, 6, 4, 3]
    )

    # Set up BEV Decoder
    bev_decoder = BEVDecoder(
        img_size=[256, 448],
        lidar_size=[640, 640],
        img_latent_size=[32, 56],
        num_patches=6400,
        img_decoder=img_decoder,
        feature_depth=64,
        voxel_depth=64,
        embed_dim=512,
        num_heads=8,
        in_channels=512,
        bias_init=-6.9,
        use_gt_voxel=False,
        visual_grid_feat_dim=64,
        use_voxel_decoder=True,
        grid_size_offset=[[0,0,0], [0,0,3]],
        downsample_voxel=[1, 1, 1],
        upsample_style="pixel_shuffle",
        use_checkpoint=True
    )

    # Set up VAEBevMultiModality
    bevw_vae = VAEBevMultiModality(
        voxelizer=voxelizer,
        img_encoder=img_encoder,
        lidar_encoder=lidar_encoder,
        deformable_transformer=deformable_transformer,
        bev_decoder=bev_decoder,
        latent_dim=512,
        num_sample_per_pillar=4
    )
    bevw_vae.train()
    bevw_vae.cuda()
    batch_size, sequence_length, num_views, c, h, w = batch["vae_images"].shape
    image_tensor = batch["vae_images"].clone().flatten(0, 1).cuda()
    import itertools
    old_points = points
    points = list(itertools.chain.from_iterable(points))
    points = [[p] for p in points]
    camera_transforms = batch["camera_transforms"].flatten(0, 1).cuda()
    camera_intrinsics = batch["camera_intrinsics"].flatten(0, 1).cuda()
    ego_transforms = batch["ego_transforms"].flatten(0, 1).cuda()
    voxels = bevw_vae.voxelizer(points)[:, 0]
    bevw_feats = bevw_vae.encode(
        voxels = voxels,
        images = image_tensor,
        camera_transforms = camera_transforms,
        camera_intrinsics = camera_intrinsics,
        ego_transforms = ego_transforms,
    )
    N, L, C = bevw_feats.shape
    hw = int(L ** 0.5)
    bevw_feats = bevw_feats.view(N, hw, hw, C).permute(0, 3, 1, 2)
    bevw_feats = bevw_feats.view(batch_size, sequence_length, C, hw, hw)
    print(bevw_feats.shape)

    model = DiffusionBevMultiModality()
    model.train()
    model.cuda()
    model(bevw_feats,
          timestep = torch.randint(0, 50, (batch_size,)).cuda(),
          ego_transforms=ego_transforms)
    pdb.set_trace()