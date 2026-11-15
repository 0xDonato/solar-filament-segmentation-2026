"""The Dice terms of the training loss: DML1 is the soft Dice loss on hard targets, and only DML1 is minimised by
the soft target itself."""
import torch

from solarfil.train import dml1_loss, soft_dice_loss


def test_dml1_equals_the_soft_dice_loss_on_hard_targets():
    g = torch.Generator().manual_seed(0)
    prob = torch.rand(2, 64, 64, generator=g)
    target = (torch.rand(2, 64, 64, generator=g) > 0.7).float()
    torch.testing.assert_close(dml1_loss(prob, target), soft_dice_loss(prob, target))


def test_only_dml1_is_minimised_by_a_soft_target():
    target = torch.zeros(64, 64)
    target[20:30, 10:50] = 1 / 3  # a filament one annotator of three drew
    target[23:27, 10:50] = 1.0
    sharpened = (target > 0.5).float()
    assert dml1_loss(target, target) == 0
    assert dml1_loss(target, target) < dml1_loss(sharpened, target)
    assert soft_dice_loss(sharpened, target) < soft_dice_loss(target, target)
