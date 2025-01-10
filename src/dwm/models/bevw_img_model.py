import torch
import torch.nn as nn
from typing import Union, List, Optional
from diffusers.models.autoencoders.vae import Encoder, Decoder
from typing import Tuple


class AutoKLEncoder(nn.Module):
    def __init__(self,
                 input_size: Union[int, List[int]],
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
                 block_out_channels: Tuple[int, ...] = (64,),
                 layers_per_block: int = 2,
                 norm_num_groups: int = 32,
                 act_fn: str = "silu",
                 double_z: bool = False,
                 mid_block_add_attention=True,
                 use_checkpoint: bool = True,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.input_size = input_size
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=double_z,
            mid_block_add_attention=mid_block_add_attention
        )
        self.encoder.gradient_checkpointing = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class AutoKLDecoder(nn.Module):
    def __init__(self,
                 input_size: Union[int, List[int]],
                 in_channels: int = 4,
                 out_channels: int = 3,
                 up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
                 block_out_channels: Tuple[int, ...] = (64,),
                 layers_per_block: int = 2,
                 norm_num_groups: int = 32,
                 act_fn: str = "silu",
                 norm_type: str = "group",  # group, spatial
                 mid_block_add_attention=True,
                 use_checkpoint: bool = True,

                 ):
        super().__init__()
        self.in_channels = in_channels
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.input_size = input_size
        self.decoder = Decoder(
            in_channels=in_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            norm_type=norm_type,
            mid_block_add_attention=mid_block_add_attention
        )
        self.decoder.gradient_checkpointing = use_checkpoint
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        x = self.activation(x)
        return x


class ConvDecoderV1(nn.Module):
    def __init__(
        self,
        input_size: Union[int, List[int]],
        in_channels: int = 16,
        embed_dim: int = 512,
        out_channels: int = 3,
        num_layers: int = 5,
        depth: int = [4, 6, 4, 3],
        groups: int = 32,
    ):
        super().__init__()
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.input_size = input_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # Initial projection
        self.initial_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, padding=1)

        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        current_dim = embed_dim
        for i in range(num_layers):
            out_dim = current_dim // 2 if i < num_layers - 1 else current_dim

            upsample_block = nn.Sequential(
                nn.ConvTranspose2d(current_dim, out_dim,
                                   kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(groups, out_dim),
                nn.SiLU(inplace=True),
                *[ResNetBlock(out_dim, groups) for _ in range(depth[i])]
            )
            self.upsample_layers.append(upsample_block)
            current_dim = out_dim

        # Final convolution
        self.final_conv = nn.Conv2d(
            current_dim, out_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_channels,
                   self.input_size[0], self.input_size[1])
        x = self.initial_proj(x)

        for layer in self.upsample_layers:
            x = layer(x)

        x = self.final_conv(x)
        x = self.activation(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x += residual
        x = self.activation(x)
        return x


class BasicConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        norm_layer: nn.Module = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: torch.tensor) -> torch.tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + identity
        out = self.relu(out)

        return out


class ConvDecoderV0(nn.Module):
    def __init__(
        self, input_size: Union[int, list[int]],
        in_channels: int = 16,
        embed_dim: int = 512,
        depth: list[int] = [4, 6, 4],
        out_channels: int = 3,
        use_checkpoint: bool = False,
        norm_type: str = "batch_norm",
    ):
        """
        Description:
            CNN-based Decoder model for 2D input
        Args:
            img_size: int or list[int]. The size of input image.
            patch_size: int. The OVERALL size of the patch. In this model, it uses patch_size // 2 in patch
                        embedding and then merges the neighboring patches into a single patch.
            in_channels: int. The number of input channels.
            embed_dim: int. The dimension of the embedding.
            depth: list[int]. The depth of each block.
            out_channels: int. The dimension of the output.
            use_checkpoint: bool. Whether to use checkpoint.

        """
        super().__init__()
        if norm_type == "batch_norm":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "group_norm":
            norm_layer = nn.GroupNorm
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        # input size of the first transformer block, i.e. the size after the patchification
        self.in_h = input_size[0]
        self.in_w = input_size[1]

        self.out_channels = out_channels
        self.use_checkpoint = use_checkpoint
        self.first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1
        )
        self.conv_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        for d in range(len(depth) - 1):
            cur_conv_blocks = nn.ModuleList([
                BasicConvBlock(
                    in_channels=embed_dim // (2 ** d),
                    out_channels=embed_dim // (2 ** d),
                    norm_layer=norm_layer
                )
                for _ in range(depth[d])
            ])
            cur_conv_blocks = nn.Sequential(*cur_conv_blocks)
            self.conv_blocks.append(cur_conv_blocks)
            self.upsampling_blocks.append(
                nn.ConvTranspose2d(embed_dim // (2 ** d),
                                   embed_dim // (2 ** (d + 1)), 2, stride=2)
            )
        final_conv = nn.ModuleList([
            BasicConvBlock(
                in_channels=embed_dim // (2 ** (len(depth) - 1)),
                out_channels=embed_dim // (2 ** (len(depth) - 1)),
                norm_layer=norm_layer
            )
            for _ in range(depth[-1])
        ])
        self.final_conv = nn.Sequential(*final_conv)
        self.final_pred = nn.Sequential(
            nn.Conv2d(embed_dim // (2 ** (len(depth) - 1)),
                      embed_dim // (2 ** len(depth)), 1, 1, bias=False),
            norm_layer(embed_dim // (2 ** len(depth))),
            nn.ReLU(),
            nn.Conv2d(embed_dim // (2 ** len(depth)),
                      out_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # nn.init.constant_(self.pre_quant.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x) -> torch.tensor:
        x = self.first_conv(x)
        for (blk, upsamp) in zip(self.conv_blocks, self.upsampling_blocks):
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(
                    upsamp, x, use_reentrant=False)
            else:
                x = blk(x)
                x = upsamp(x)

        x = self.final_conv(x)
        x = self.final_pred(x)

        return x
