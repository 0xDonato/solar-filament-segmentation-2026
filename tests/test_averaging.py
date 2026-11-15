"""Weight averages kept during training: the tail average is the plain mean of the states it saw."""
import torch

from solarfil.model import WeightAverage


def test_weight_average_is_the_uniform_mean_of_the_states_it_saw():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.BatchNorm2d(2))
    avg, states = WeightAverage(), []
    for _ in range(3):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p))
            m[1].running_mean.add_(1.0)
            m[1].num_batches_tracked += 1
        avg.update(m)
        states.append({k: v.clone() for k, v in m.state_dict().items()})
    assert avg.n == 3
    for k, v in avg.state_dict().items():
        if v.dtype.is_floating_point:
            torch.testing.assert_close(v, torch.stack([s[k] for s in states]).mean(0))
        else:  # step counters follow the latest state
            assert torch.equal(v, states[-1][k])
