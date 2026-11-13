from enum import Enum
from typing import Literal, overload
import numpy as np
import numpy.typing as npt

class TagType(Enum):
    tag36h11 = 1
tag36h11: TagType

class CameraMatrix:
    """Pinhole intrinsics in pixels. Fields default to 0."""
    fx: float
    cx: float
    fy: float
    cy: float
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, fx: float, cx: float, fy: float, cy: float) -> None: ...

class DistCoeffs:
    """OpenCV (Brown-Conrady) distortion. Fields default to 0."""
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, k1: float, k2: float, p1: float, p2: float, k3: float) -> None: ...

class Pose:
    """Camera-from-tag pose; tag frame x right, y down, z into the tag."""
    @property
    def rotation(self) -> list[list[float]]: ...
    @property
    def translation(self) -> list[float]: ...

class DetectionResult:
    def __init__(self) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def hamming(self) -> int: ...
    @property
    def decision_margin(self) -> float: ...
    @property
    def homography(self) -> list[list[float]]: ...
    @property
    def center(self) -> list[float]: ...
    @property
    def corners(self) -> list[list[float]]: ...
    @property
    def pose(self) -> Pose | None: ...

class CudaTagsWrapper:
    """A GPU detector for one image size. Use it from one thread at a time."""
    def __init__(self, tagType: TagType, camMatrix: CameraMatrix,
                 camDistCoeffs: DistCoeffs, nthreads: int,
                 imgWidth: int, imgHeight: int, *,
                 tag_size_m: float = 0.0, decimate: int = 2,
                 min_white_black_diff: int = 5, max_hamming: int = 1,
                 refine_edges: bool = True,
                 decode_sharpening: float = 0.25,
                 color_conversion: Literal["before_detecting", "while_detecting"] = "before_detecting") -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def tag_size_m(self) -> float: ...
    @property
    def color_conversion(self) -> Literal["while_detecting", "before_detecting"]: ...
    def input_buffer(self, channels: Literal[1, 3, 4] = 3) -> npt.NDArray[np.uint8]: ...
    def process(self, image: npt.NDArray[np.uint8],
                color_order: Literal["bgr", "rgb"] = "bgr") -> list[DetectionResult]: ...

def to_gray(image: npt.NDArray[np.uint8],
            color_order: Literal["bgr", "rgb"] = "bgr") -> npt.NDArray[np.uint8]: ...
def cuda_available() -> bool: ...
