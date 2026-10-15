"""Configurable U-Net for 2D or 3D inputs.

- Encoder/decoder of `ConvBlock`s (two conv-norm-activation layers, optional residual connection,
  optional SE or CBAM attention, optional dropout).
- Skip connections, optionally filtered by attention gates (Oktay et al., Attention U-Net).
- Bottleneck: plain conv block, transformer (multi-head self-attention over all bottleneck voxels with a
  convolutional positional encoding), or ASPP (parallel dilated convolutions + global context).
- Deep supervision: 1x1 heads on the highest-resolution decoder levels.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False  # benchmarking 3D convs briefly needs >2x the training memory and gave no speed-up


def conv_nd(dims):
    return nn.Conv3d if dims == 3 else nn.Conv2d


def make_norm(kind, channels, dims):
    if kind == "instance":
        return (nn.InstanceNorm3d if dims == 3 else nn.InstanceNorm2d)(channels, affine=True)
    if kind == "batch":
        return (nn.BatchNorm3d if dims == 3 else nn.BatchNorm2d)(channels)
    if kind == "group":
        return nn.GroupNorm(math.gcd(8, channels), channels)
    raise ValueError(kind)


def make_act(kind):
    return {"leakyrelu": lambda: nn.LeakyReLU(0.01, inplace=True), "relu": lambda: nn.ReLU(inplace=True), "gelu": nn.GELU}[kind]()


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, dims, reduction=8):
        super().__init__()
        Conv, hidden = conv_nd(dims), max(channels // reduction, 4)
        self.fc = nn.Sequential(Conv(channels, hidden, 1), nn.ReLU(inplace=True), Conv(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(dim=tuple(range(2, x.ndim)), keepdim=True))


class CBAM(nn.Module):
    def __init__(self, channels, dims, reduction=8):
        super().__init__()
        Conv, hidden = conv_nd(dims), max(channels // reduction, 4)
        self.mlp = nn.Sequential(Conv(channels, hidden, 1), nn.ReLU(inplace=True), Conv(hidden, channels, 1))
        self.spatial = Conv(2, 1, 7, padding=3)

    def forward(self, x):
        spatial_dims = tuple(range(2, x.ndim))
        x = x * torch.sigmoid(self.mlp(x.mean(spatial_dims, keepdim=True)) + self.mlp(x.amax(spatial_dims, keepdim=True)))
        return x * torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dims, cfg, stride=1):
        super().__init__()
        Conv = conv_nd(dims)
        self.conv1 = Conv(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1, self.act1 = make_norm(cfg["norm"], out_ch, dims), make_act(cfg["activation"])
        self.conv2 = Conv(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2, self.act2 = make_norm(cfg["norm"], out_ch, dims), make_act(cfg["activation"])
        self.dropout = (nn.Dropout3d if dims == 3 else nn.Dropout2d)(cfg["dropout"]) if cfg["dropout"] > 0 else nn.Identity()
        self.attention = {None: nn.Identity, "se": lambda: SqueezeExcitation(out_ch, dims), "cbam": lambda: CBAM(out_ch, dims)}[cfg["block_attention"]]()
        self.residual = cfg["residual"]
        self.projection = None
        if self.residual and (in_ch != out_ch or stride != 1):
            self.projection = nn.Sequential(Conv(in_ch, out_ch, 1, stride=stride, bias=False), make_norm(cfg["norm"], out_ch, dims))

    def forward(self, x):
        y = self.dropout(self.act1(self.norm1(self.conv1(x))))
        y = self.attention(self.norm2(self.conv2(y)))
        if self.residual:
            y = y + (self.projection(x) if self.projection is not None else x)
        return self.act2(y)


class AttentionGate(nn.Module):
    def __init__(self, skip_ch, gate_ch, dims, norm):
        super().__init__()
        Conv, inter = conv_nd(dims), max(skip_ch // 2, 1)
        self.w_skip = nn.Sequential(Conv(skip_ch, inter, 1, bias=False), make_norm(norm, inter, dims))
        self.w_gate = nn.Sequential(Conv(gate_ch, inter, 1, bias=False), make_norm(norm, inter, dims))
        self.psi = nn.Sequential(Conv(inter, 1, 1), nn.Sigmoid())

    def forward(self, skip, gate):
        return skip * self.psi(F.relu(self.w_skip(skip) + self.w_gate(gate)))


class TransformerBottleneck(nn.Module):
    def __init__(self, channels, dims, cfg):
        super().__init__()
        assert channels % cfg["transformer_heads"] == 0, "bottleneck channels must be divisible by transformer_heads"
        self.position = conv_nd(dims)(channels, channels, 3, padding=1, groups=channels)  # conditional positional encoding
        layer = nn.TransformerEncoderLayer(channels, cfg["transformer_heads"], int(channels * cfg["transformer_mlp_ratio"]),
                                           dropout=cfg["dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg["transformer_layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x + self.position(x)
        tokens = self.norm(self.encoder(x.flatten(2).transpose(1, 2)))
        return tokens.transpose(1, 2).reshape(x.shape)


class ASPPBottleneck(nn.Module):
    def __init__(self, channels, dims, cfg):
        super().__init__()
        Conv = conv_nd(dims)
        branch = lambda k, rate: nn.Sequential(Conv(channels, channels, k, padding=rate if k == 3 else 0, dilation=rate, bias=False),
                                               make_norm(cfg["norm"], channels, dims), make_act(cfg["activation"]))
        self.branches = nn.ModuleList([branch(1, 1)] + [branch(3, r) for r in cfg["aspp_rates"]])
        self.pool = nn.Sequential(Conv(channels, channels, 1), make_act(cfg["activation"]))
        self.project = nn.Sequential(Conv(channels * (len(self.branches) + 1), channels, 1, bias=False),
                                     make_norm(cfg["norm"], channels, dims), make_act(cfg["activation"]))

    def forward(self, x):
        pooled = self.pool(x.mean(dim=tuple(range(2, x.ndim)), keepdim=True)).expand_as(x)
        return x + self.project(torch.cat([b(x) for b in self.branches] + [pooled], 1))


class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dims, depth = cfg["spatial_dims"], cfg["depth"]
        Conv = conv_nd(dims)
        channels = [min(cfg["base_channels"] * 2 ** i, cfg["max_channels"]) for i in range(depth)]
        self.encoders = nn.ModuleList()
        for i in range(depth):
            in_ch = cfg["in_channels"] if i == 0 else channels[i - 1]
            if i == 0:
                self.encoders.append(ConvBlock(in_ch, channels[0], dims, cfg))
            elif cfg["downsample"] == "strided_conv":
                self.encoders.append(ConvBlock(in_ch, channels[i], dims, cfg, stride=2))
            else:
                self.encoders.append(nn.Sequential((nn.MaxPool3d if dims == 3 else nn.MaxPool2d)(2), ConvBlock(in_ch, channels[i], dims, cfg)))
        self.bottleneck = {"conv": nn.Identity, "transformer": lambda: TransformerBottleneck(channels[-1], dims, cfg),
                           "aspp": lambda: ASPPBottleneck(channels[-1], dims, cfg)}[cfg["bottleneck"]]()
        self.ups, self.gates, self.decoders, self.heads = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(depth - 1)):
            if cfg["upsample"] == "transpose":
                self.ups.append((nn.ConvTranspose3d if dims == 3 else nn.ConvTranspose2d)(channels[i + 1], channels[i], 2, stride=2))
            else:
                self.ups.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="trilinear" if dims == 3 else "bilinear", align_corners=False),
                                              Conv(channels[i + 1], channels[i], 1)))
            self.gates.append(AttentionGate(channels[i], channels[i], dims, cfg["norm"]) if cfg["skip_attention"] else nn.Identity())
            self.decoders.append(ConvBlock(2 * channels[i], channels[i], dims, cfg))
            self.heads.append(Conv(channels[i], N_CLASSES, 1))
        self.skip_attention = cfg["skip_attention"]
        self.ds_levels = max(1, min(cfg["deep_supervision_levels"], depth - 1)) if cfg["deep_supervision"] else 1

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        x = self.bottleneck(skips.pop())
        outputs = []
        for level, (up, gate, decoder, head) in enumerate(zip(self.ups, self.gates, self.decoders, self.heads)):
            x = up(x)
            skip = skips.pop()
            if self.skip_attention:
                skip = gate(skip, x)
            x = decoder(torch.cat([x, skip], 1))
            if level >= len(self.ups) - self.ds_levels:  # only the needed heads are evaluated
                outputs.append(head(x))
        outputs = outputs[::-1]  # full resolution first
        return outputs if self.training and self.ds_levels > 1 else outputs[0]


def memory_format(cfg):
    if not cfg.get("channels_last", False):
        return torch.contiguous_format
    return torch.channels_last_3d if cfg["spatial_dims"] == 3 else torch.channels_last


def autocast(cfg):
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, None: None}[cfg["amp"]]
    return torch.autocast(DEVICE.type, dtype=amp_dtype, enabled=amp_dtype is not None)


def build_model(cfg):
    return UNet(cfg).to(DEVICE, memory_format=memory_format(cfg))


def check_model(cfg, batch=None):
    """Build the model from a resolved config, run one forward/backward pass at the patch size, report memory."""
    model = build_model(cfg).train()
    patch = cfg["patch_size_3d"] if cfg["spatial_dims"] == 3 else cfg["patch_size_2d"]
    batch = batch or (cfg["batch_size_3d"] if cfg["spatial_dims"] == 3 else cfg["batch_size_2d"])
    x = torch.randn(batch, cfg["in_channels"], *patch, device=DEVICE).contiguous(memory_format=memory_format(cfg))
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with autocast(cfg):
        out = model(x)
    outputs = out if isinstance(out, list) else [out]
    sum(o.float().mean() for o in outputs).backward()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    memory = torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else float("nan")
    print(f"{cfg['experiment_name']}: {params:.1f} M parameters; outputs {[tuple(o.shape) for o in outputs]}; "
          f"peak GPU memory {memory:.1f} GiB (batch {batch})")
    del model, x, out, outputs
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
