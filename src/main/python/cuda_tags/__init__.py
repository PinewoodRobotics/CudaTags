"""Typed AprilTag API. Native symbols load only when requested."""
from importlib import import_module

__all__ = ["CameraMatrix", "DistCoeffs", "TagType", "Pose", "DetectionResult",
           "CudaTagsWrapper", "tag36h11", "to_gray", "cuda_available"]

def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    try:
        native = import_module(".cuda_tags", __name__)
    except ImportError as exc:
        raise ImportError(
            "CUDA Tags native backend is unavailable. Type information remains "
            "available; install a native build on a CUDA host to run detection."
        ) from exc
    value = getattr(native, name)
    globals()[name] = value
    return value
