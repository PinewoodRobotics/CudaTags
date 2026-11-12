from enum import Enum
from typing import overload
import numpy as np
import numpy.typing as npt

class TagType(Enum):
    tag36h11 = 1
tag36h11: TagType

class CameraMatrix:
    fx: float
    cx: float
    fy: float
    cy: float
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, fx: float, cx: float, fy: float, cy: float) -> None: ...

class DistCoeffs:
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, k1: float, k2: float, p1: float, p2: float, k3: float) -> None: ...

class DetectionResult:
    id: int
    hamming: int
    decision_margin: float
    def __init__(self) -> None: ...
    @property
    def homography(self) -> list[list[float]]: ...
    @property
    def center(self) -> list[float]: ...
    @property
    def corners(self) -> list[list[float]]: ...

class CudaTagsWrapper:
    def __init__(self, tagType: TagType, camMatrix: CameraMatrix,
                 camDistCoeffs: DistCoeffs, nthreads: int,
                 imgWidth: int, imgHeight: int) -> None: ...
    def process(self, image: npt.NDArray[np.uint8]) -> list[DetectionResult]: ...
