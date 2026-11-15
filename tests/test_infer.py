"""Inference with a one-input, one-output network (no coordinate planes, no spine map)."""
import numpy as np
import torch

from solarfil.frames import Frame
from solarfil.infer import predict_frame, read_map_png, write_map_png
from solarfil.model import InnerScale, build_model, io_channels
from solarfil.preprocess import Disk


def test_io_channels_follow_the_recipe():
    assert io_channels(dict(coords=True, spine_w=0.3)) == dict(in_channels=3, classes=2)
    assert io_channels(dict(coords=False, spine_w=0.0)) == dict(in_channels=1, classes=1)


def test_stem_unet_fills_the_empty_half_scale_skip_of_convnext():
    torch.manual_seed(0)
    m = build_model("unet_stem", "tu-convnext_atto", weights=None, in_channels=1, classes=1).eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        assert m.encoder(x)[1].shape[1] == 0  # smp's placeholder for the missing 1/2-scale feature
        assert m.stem(x).shape == (2, 32, 32, 32)
        y = m(x)
        assert y.shape == (2, 1, 64, 64)
        m.stem[0].weight.zero_()  # the stem feeds the output: silencing it changes the map
        assert not torch.allclose(m(x), y)


def test_one_map_network_predicts_and_stores_a_map_without_spine(tmp_path):
    torch.manual_seed(0)
    m = build_model("unet", "resnet18", weights=None, in_channels=1, classes=1).eval()
    rng = np.random.default_rng(0)
    fr = Frame("x", Disk(32.0, 32.0, 25.0), 1, rng.integers(0, 256, (64, 64), dtype=np.uint8), coords=False)
    pm = predict_frame([m], fr, torch.device("cpu"), n_flips=4)
    assert pm.shape == (1, 64, 64) and np.all((pm > 0) & (pm < 1))
    write_map_png(tmp_path / "m.png", pm)
    p, s = read_map_png(tmp_path / "m.png", native=False)
    np.testing.assert_allclose(p, pm[0], atol=1 / 255)
    assert not s.any()


def test_a_label_dy_network_has_its_map_moved_up():
    torch.manual_seed(0)
    m = build_model("unet", "resnet18", weights=None, in_channels=1, classes=1).eval()
    rng = np.random.default_rng(0)
    fr = Frame("x", Disk(32.0, 32.0, 25.0), 1, rng.integers(0, 256, (64, 64), dtype=np.uint8), coords=False)
    plain = predict_frame([m], fr, torch.device("cpu"), n_flips=2)[0]
    moved = predict_frame([m], fr, torch.device("cpu"), n_flips=2, label_dys=[0.25])[0]
    np.testing.assert_allclose(moved[:-1], 0.75 * plain[:-1] + 0.25 * plain[1:], atol=1e-6)
    both = predict_frame([m, m], fr, torch.device("cpu"), n_flips=2, label_dys=[0.25, 0.0])[0]
    np.testing.assert_allclose(both, (moved + plain) / 2, atol=1e-6)


def test_zoomed_inference_returns_the_model_scale_map():
    torch.manual_seed(0)
    m = torch.nn.Conv2d(1, 1, 1).eval()  # pointwise: zooming in and back out only blurs a smooth map
    yy, xx = np.mgrid[0:64, 0:64]
    img = (127 + 100 * np.sin(xx / 9.0) * np.cos(yy / 13.0)).astype(np.uint8)
    fr = Frame("x", Disk(32.0, 32.0, 25.0), 1, img, coords=False)
    plain = predict_frame([m], fr, torch.device("cpu"), n_flips=2)
    zoomed = predict_frame([m], fr, torch.device("cpu"), n_flips=2, zoom=1.5)
    assert zoomed.shape == plain.shape == (1, 64, 64)
    np.testing.assert_allclose(zoomed[:, 2:-2, 2:-2], plain[:, 2:-2, 2:-2], atol=0.01)
    assert not np.allclose(zoomed, plain, atol=1e-6)  # the zoom did take place


def test_an_inner_scale_network_returns_logits_on_its_input_grid():
    torch.manual_seed(0)
    m = build_model("unet", "resnet18", weights=None, in_channels=1, classes=1, inner_scale=1.5).eval()
    assert isinstance(m, InnerScale) and m.factor == 1.5
    with torch.no_grad():
        y = m(torch.zeros(1, 1, 64, 64))  # seen at 96 px inside, a multiple of the encoder's stride 32
    assert y.shape == (1, 1, 64, 64)
    assert build_model("unet", "resnet18", weights=None, in_channels=1, classes=1).__class__.__name__ == "Unet"


def test_tiled_maps_equal_whole_maps_for_a_pointwise_network_and_cover_every_pixel():
    import torch

    from solarfil.infer import tiled_sigmoid
    torch.manual_seed(0)
    net = torch.nn.Conv2d(1, 1, 1)  # each output pixel sees only its own input pixel
    x = torch.randn(1, 1, 96, 80)
    whole = tiled_sigmoid(net, x, 0)
    for tile in (32, 40, 48):  # strides of half a tile, the last window flush with the far edge
        assert torch.allclose(tiled_sigmoid(net, x, tile), whole, atol=1e-6)
    assert torch.equal(tiled_sigmoid(net, x, 128), whole)  # a tile larger than the frame runs it whole
    ctx = torch.nn.Conv2d(1, 1, 9, padding=4)  # with context, windows differ only near their borders
    y = tiled_sigmoid(ctx, x, 48)
    assert y.shape == whole.shape and torch.isfinite(y).all()
