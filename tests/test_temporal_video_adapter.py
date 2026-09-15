import pytest
import torch

from dire.temporal_video_adapter import TemporalVideoAdapter


def test_temporal_adapter_returns_normalized_features_and_frame_weights():
    model = TemporalVideoAdapter(dim=4, hidden_dim=3, max_frames=3)
    output = model(torch.randn(2, 3, 4))
    assert output.features.shape == (2, 4)
    assert output.attention.shape == (2, 3)
    assert torch.allclose(output.features.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output.attention.sum(dim=1), torch.ones(2), atol=1e-5)
    assert model.scaled_similarity(output.features, output.features).shape == (2, 2)


def test_temporal_adapter_rejects_incompatible_frame_shapes():
    model = TemporalVideoAdapter(dim=4, hidden_dim=3, max_frames=3)
    with pytest.raises(ValueError, match="feature dimension"):
        model(torch.randn(2, 3, 5))
    with pytest.raises(ValueError, match="between 1 and 3"):
        model(torch.randn(2, 4, 4))
