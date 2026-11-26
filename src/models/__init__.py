"""PyTorch neural network architectures for raw inertial HAR."""

from src.models.cnn1d import CompactCNN1D, CompactResCNN1D, build_cnn_model

__all__ = ["CompactCNN1D", "CompactResCNN1D", "build_cnn_model"]
