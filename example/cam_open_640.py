import cuda_tags
import numpy as np
import cv2
import time

camera_matrix = cuda_tags.CameraMatrix(1000, 1000, 1000, 1000)
dist_coeffs = cuda_tags.DistCoeffs(0, 0, 0, 0, 0)
tags_wrapper = cuda_tags.CudaTagsWrapper(
    cuda_tags.TagType.tag36h11, camera_matrix, dist_coeffs, 1, 640, 480
)

camera_feed = cv2.VideoCapture(0)
camera_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
fps = 0.0
fps_alpha = 0.1  # Smoothing factor for FPS calculation
smoothed_fps = 0.0

# Custom colors for drawings
COLORS = [
    (0, 255, 0),  # Green
    (0, 255, 255),  # Yellow
    (255, 0, 0),  # Blue
    (0, 128, 255),  # Orange
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 0, 255),  # Red
    (180, 105, 255),  # Pink
    (255, 255, 255),  # White
]


def draw_tag_info(img, detection, detection_idx):
    # Draw detected corners
    corners = np.array(detection.corners, dtype=np.float32).reshape((4, 2))
    color = COLORS[detection_idx % len(COLORS)]
    for i in range(4):
        pt1 = tuple(np.int32(corners[i % 4]))
        pt2 = tuple(np.int32(corners[(i + 1) % 4]))
        cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)
        cv2.circle(img, pt1, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, pt1, 2, color, -1, cv2.LINE_AA)

    # Draw tag center as a cross
    center = tuple(np.int32(detection.center))
    cv2.drawMarker(
        img, center, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3
    )

    # Write tag ID & margin
    margin = detection.decision_margin
    tag_id = detection.id
    label = f"ID:{tag_id} m:{margin:.2f}"
    cv2.putText(
        img,
        label,
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    # Draw homography axes (estimate orientation)
    H = np.array(detection.homography, dtype=np.float64)
    # We'll use the top-left corner as the origin for axes
    origin = tuple(np.int32(corners[0]))
    # Homography columns: first two are direction vectors
    x_axis = (int(origin[0] + 40 * H[0][0]), int(origin[1] + 40 * H[1][0]))
    y_axis = (int(origin[0] + 40 * H[0][1]), int(origin[1] + 40 * H[1][1]))
    cv2.arrowedLine(img, origin, x_axis, (0, 0, 255), 3, tipLength=0.3)  # X axis: Red
    cv2.arrowedLine(img, origin, y_axis, (0, 255, 0), 3, tipLength=0.3)  # Y axis: Green


while True:
    ret, frame = camera_feed.read()
    if not ret:
        break

    # Measure detection time
    detection_start = time.time()
    results = tags_wrapper.process(frame)
    detection_time = time.time() - detection_start

    num_tags = len(results)
    display_frame = frame.copy()

    # Calculate FPS with smoothing
    curr_time = time.time()
    elapsed = curr_time - prev_time
    if elapsed > 0:
        fps = 1.0 / elapsed
        # Smooth the FPS using exponential moving average
        if smoothed_fps == 0.0:
            smoothed_fps = fps
        else:
            smoothed_fps = fps_alpha * fps + (1 - fps_alpha) * smoothed_fps
    prev_time = curr_time

    # Draw tags, axes, and info for each detection
    for idx, detection in enumerate(results):
        draw_tag_info(display_frame, detection, idx)

    # Put the number of tags detected, detection time, and FPS on the image (with background rectangles for better readability)
    txt1 = f"Tags detected: {num_tags}"
    txt2 = f"Detection: {detection_time*1000:.2f} ms"
    txt3 = f"FPS: {smoothed_fps:.2f}"
    for i, txt in enumerate([txt1, txt2, txt3]):
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        ty = 30 + i * 50
        cv2.rectangle(
            display_frame, (6, ty - th - 7), (16 + tw, ty + 12), (0, 0, 0), -1
        )
        cv2.putText(
            display_frame,
            txt,
            (10, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            COLORS[i % len(COLORS)],
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("CUDA Tags Cool Demo", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
