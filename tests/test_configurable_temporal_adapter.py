import pytest
import torch

from dire.configurable_temporal_adapter import ARCHITECTURES, architecture_flags, build_configurable_adapter


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_configurable_adapter_returns_normalized_embeddings_and_attention(architecture):
    model = build_configurable_adapter(4, 3, 3, architecture)
    output = model(torch.randn(2, 3, 4))
    assert output.features.shape == (2, 4)
    assert torch.allclose(output.features.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output.attention.sum(dim=1), torch.ones(2), atol=1e-5)


def test_architecture_flags_reject_unknown_name():
    with pytest.raises(ValueError, match="Unknown architecture"):
        architecture_flags("unknown")
