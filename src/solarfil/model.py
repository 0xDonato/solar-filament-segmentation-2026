"""Dense network: an encoder-decoder that maps the flat field (and optionally x/R, y/R) to logits per pixel.

Channel 0 is the filament probability ``p`` (the chance that a random annotator marks the pixel), the optional
channel 1 the spine probability. Encoders come from timm or torchvision through segmentation_models_pytorch;
every pretrained weight used is under a permissive licence (Apache-2.0 / BSD / MIT).
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch import nn

IN_CHANNELS = 3
OUT_CHANNELS = 2


class StemUnet(nn.Module):
    """U-Net for an encoder whose first feature map is at 1/4 scale (ConvNeXt): smp gives such an encoder an empty
    1/2-scale feature, so the decoder's last skip carries nothing. A learned stride-2 stem on the input fills it,
    as in the 1st-place solution of SenNet + HOA 2024 (Kaggle topic 475522). Decoder and head are smp's U-Net
    defaults."""

    def __init__(self, encoder: str, weights: str | None, in_channels: int, classes: int, stem_channels: int = 32):
        super().__init__()
        base = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=in_channels, classes=classes)
        channels = list(base.encoder.out_channels)
        if channels[1] != 0:
            raise ValueError(f"{encoder} has a 1/2-scale feature already; use arch 'unet'")
        channels[1] = stem_channels
        self.encoder = base.encoder
        self.stem = nn.Sequential(nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(stem_channels), nn.ReLU(inplace=True))
        self.decoder = UnetDecoder(encoder_channels=channels, decoder_channels=(256, 128, 64, 32, 16), n_blocks=5)
        self.segmentation_head = base.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits per pixel, with the stem's features in place of the encoder's empty 1/2-scale feature."""
        features = self.encoder(x)
        features[1] = self.stem(x)
        return self.segmentation_head(self.decoder(features))


class InnerScale(nn.Module):
    """A network that sees its input enlarged by `factor` and returns its logits on the input's own grid (bilinear
    both ways), as in the 5th-place solution of SenNet + HOA 2024 (Kaggle topic 475288): thin structures are sampled
    more finely inside the network, while the targets, the component tree and the scorer stay at native resolution.
    The enlarged size must still divide by the encoder's stride (a 1024 px crop at 1.5 is 1536 px)."""

    def __init__(self, net: nn.Module, factor: float):
        super().__init__()
        self.net, self.factor = net, factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The wrapped network's logits of the enlarged input, resized back to the input's size."""
        size = x.shape[-2:]
        y = self.net(F.interpolate(x, scale_factor=self.factor, mode="bilinear", align_corners=False))
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class PopulationMap(nn.Module):
    """A network trained on the latent filament mask (`annotators.GroupAnnotators`) that returns the population
    annotator's logits: channel 0 becomes logit(alpha pi + (1 - beta)(1 - pi)), with pi the network's latent
    probability and (alpha, beta) the buffer `annotator`, which the training sets at its end. Training reads the
    latent logits from `net` directly."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.register_buffer("annotator", torch.ones(2))  # alpha, beta; (1, 1) maps pi to itself

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The population annotator's logit in channel 0, the wrapped network's other channels unchanged."""
        z = self.net(x)
        alpha, beta = self.annotator.float()
        p = (1 - beta) + (alpha + beta - 1) * torch.sigmoid(z[:, :1].float())
        return torch.cat([torch.logit(p, eps=1e-6).to(z.dtype), z[:, 1:]], 1)


class Softmax3(nn.Module):
    """A network with three softmax classes (background, filament body without its contact borders, contact border;
    DSB 2018 1st, Kaggle topic 54741). Training reads the three raw logits; in
    eval mode it returns two logits whose sigmoids are P(filament) = P(body) + P(contact) and P(contact), so the
    maps keep the members' format (channel 0 the filament map, channel 1 beside it)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw (N, 3) logits in training mode, else the two logits of P(filament) and P(contact)."""
        z = self.net(x)
        if self.training:
            return z
        z = z.float()
        fil = torch.logsumexp(z[:, 1:3], 1, keepdim=True) - z[:, :1]
        con = z[:, 2:3] - torch.logsumexp(z[:, 0:2], 1, keepdim=True)
        return torch.cat([fil, con], 1)


class _GNBlock(nn.Module):
    """A U-Net decoder block: upsample x2, concatenate the skip, two 3x3 convolutions with GroupNorm and GELU."""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        g = lambda c: nn.GroupNorm(min(32, c // 4) if c >= 8 else 1, c)  # noqa: E731
        self.body = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), g(cout), nn.GELU(),
                                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), g(cout), nn.GELU())

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.body(x if skip is None else torch.cat([x, skip], 1))


class DeepSupUnet(nn.Module):
    """U-Net with a GroupNorm/GELU decoder and deep supervision at 1/2 and 1/4 scale (SenNet 1st,
    Contrails 2nd): training returns the full-scale logits and the logits of the heads at 1/2 and 1/4, eval mode
    the full-scale logits only, so the maps keep the members' format."""

    def __init__(self, encoder: str, weights: str | None, in_channels: int, classes: int,
                 decoder_channels: tuple = (256, 128, 64, 32, 16)):
        super().__init__()
        from segmentation_models_pytorch.encoders import get_encoder
        self.encoder = get_encoder(encoder, in_channels=in_channels, depth=5, weights=weights)
        enc = list(self.encoder.out_channels)[1:][::-1]  # deepest first
        skips = enc[1:] + [0]
        ins = [enc[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList(_GNBlock(i, s, o) for i, s, o in zip(ins, skips, decoder_channels))
        self.head = nn.Conv2d(decoder_channels[-1], classes, 3, padding=1)
        self.aux = nn.ModuleList([nn.Conv2d(decoder_channels[3], classes, 1),  # 1/2 scale
                                  nn.Conv2d(decoder_channels[2], classes, 1)])  # 1/4 scale

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)[1:][::-1]
        y, skips, outs = feats[0], feats[1:] + [None], []
        for blk, s in zip(self.blocks, skips):
            y = blk(y, s)
            outs.append(y)
        main = self.head(y)
        if not self.training:
            return main
        return main, self.aux[0](outs[3]), self.aux[1](outs[2])


def build_model(arch: str = "unet", encoder: str = "resnet34", weights: str | None = "imagenet",
                in_channels: int = IN_CHANNELS, classes: int = OUT_CHANNELS, inner_scale: float = 1.0,
                population: bool = False) -> nn.Module:
    """A dense network: smp's U-Net (``unet``), U-Net++ (``unetpp``), U-Net++ with scSE (``unetpp_scse``), or
    `StemUnet` (``unet_stem``), wrapped in
    `InnerScale` when `inner_scale` is not 1 and in `PopulationMap` with `population`."""
    kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=in_channels, classes=classes)
    if arch == "unet":
        net = smp.Unet(**kw)
    elif arch == "unetpp":
        net = smp.UnetPlusPlus(**kw)
    elif arch == "unetpp_scse":
        net = smp.UnetPlusPlus(**kw, decoder_attention_type="scse")
    elif arch == "unet_stem":
        net = StemUnet(encoder, weights, in_channels, classes)
    elif arch == "unet_sm3":  # three softmax classes, two sigmoid logits in eval mode (`Softmax3`)
        net = Softmax3(smp.Unet(**dict(kw, classes=3)))
    elif arch == "unet_ds":
        net = DeepSupUnet(encoder, weights, in_channels, classes)
    else:
        raise ValueError(arch)
    net = net if inner_scale == 1.0 else InnerScale(net, inner_scale)
    return PopulationMap(net) if population else net


def io_channels(cfg: dict) -> dict:
    """Input planes and output maps of a recipe: the coordinate planes and the spine map are optional."""
    return dict(in_channels=IN_CHANNELS if cfg["coords"] else 1, classes=OUT_CHANNELS if cfg["spine_w"] > 0 else 1)


def model_from_cfg(cfg: dict, weights: str | None = None) -> nn.Module:
    """The network of a saved recipe (a run's `cfg.json`); runs saved before the channel counts were recorded have
    three inputs and two outputs."""
    return build_model(cfg["arch"], cfg["encoder"], weights, in_channels=cfg.get("in_channels", IN_CHANNELS),
                       classes=cfg.get("classes", OUT_CHANNELS), inner_scale=cfg.get("inner_scale", 1.0),
                       population=cfg.get("hier_em", False))


def uses_coords(cfg: dict) -> bool:
    """Whether a saved recipe's network reads the coordinate planes."""
    return cfg.get("in_channels", IN_CHANNELS) > 1


def device() -> torch.device:
    """The best available device: CUDA, then MPS, then the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inference_dtype(dev: torch.device) -> torch.dtype | None:
    """The autocast type for the dense maps: bfloat16 on a CUDA GPU that computes it natively (Ampere or newer),
    else None (float32)."""
    if dev.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return None


class EMA:
    """Exponential moving average of parameters and buffers (BatchNorm statistics included)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Move the average towards the model's current state; integer buffers are copied."""
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        """The averaged state, loadable into the model."""
        return self.shadow


class WeightAverage:
    """Uniform average of parameters and buffers over the steps it is updated: stochastic weight averaging of one
    training run's tail (Izmailov et al., arXiv 1803.05407), BatchNorm statistics averaged as the EMA does."""

    def __init__(self):
        self.n = 0
        self.shadow: dict | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Add the model's current state to the running mean; integer buffers are copied."""
        self.n += 1
        if self.shadow is None:
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.add_(v.detach() - s, alpha=1.0 / self.n)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        """The averaged state (None before the first update), loadable into the model."""
        return self.shadow
