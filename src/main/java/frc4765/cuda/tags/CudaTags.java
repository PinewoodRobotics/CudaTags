package frc4765.cuda.tags;

import java.util.Objects;

/**
 * A GPU tag36h11 AprilTag detector for one image size (CudaTags).
 *
 * <p>Loads {@code libcuda_tags_jni} with {@link System#loadLibrary}. A detector owns GPU memory
 * and a CUDA stream: use it from the thread that created it and {@link #close()} it when done.
 * Frames of another size are rejected; build a new detector instead.
 */
public final class CudaTags implements AutoCloseable {
  static {
    System.loadLibrary("cuda_tags_jni");
  }

  /** Doubles per detection in the array returned by {@code nativeDetect}. */
  private static final int STRIDE = 35;

  /** Camera-from-tag pose; tag frame x right, y down, z into the tag. */
  public record Pose(double[] rotationRowMajor, double[] translationMeters) {}

  /**
   * One tag. Corners are pixels in the input image, bottom-left, bottom-right, top-right, top-left
   * of an upright tag. {@code pose} is null when the tag size is 0 or the solve failed.
   */
  public record Detection(
      int id,
      int hamming,
      float decisionMargin,
      double[] center,
      double[][] corners,
      double[] homographyRowMajor,
      Pose pose) {}

  private long handle;
  private final int width;
  private final int height;

  /**
   * @param distortion OpenCV k1, k2, p1, p2, k3 (null for none)
   * @param tagSizeMeters black border edge length; 0 disables pose estimation
   */
  public CudaTags(
      int width,
      int height,
      double fx,
      double fy,
      double cx,
      double cy,
      double[] distortion,
      int threads,
      double tagSizeMeters) {
    double[] coefficients = distortion == null ? new double[5] : distortion;
    if (coefficients.length != 5) {
      throw new IllegalArgumentException("distortion must hold k1, k2, p1, p2, k3");
    }
    this.width = width;
    this.height = height;
    this.handle =
        nativeCreate(width, height, fx, fy, cx, cy, coefficients, threads, tagSizeMeters);
  }

  /** True when the native library contains the CUDA detector. */
  public static native boolean cudaAvailable();

  public int width() {
    return width;
  }

  public int height() {
    return height;
  }

  /** Detects tags in a contiguous mono8 image of {@code width * height} bytes. */
  public synchronized Detection[] detect(byte[] gray) {
    return detect(gray, width, 1);
  }

  /**
   * Detects tags in a possibly padded 8-bit image.
   *
   * @param stride bytes per row
   * @param channels 1 (gray), 3 (BGR) or 4 (BGRA)
   */
  public synchronized Detection[] detect(byte[] pixels, int stride, int channels) {
    Objects.requireNonNull(pixels, "pixels");
    return unpack(nativeDetect(checkedHandle(), pixels, stride, channels));
  }

  /** Detects tags in an OpenCV {@code Mat} (CV_8UC1/3/4) given its native address. */
  public synchronized Detection[] detectMat(long matAddress) {
    return unpack(nativeDetectMat(checkedHandle(), matAddress));
  }

  @Override
  public synchronized void close() {
    if (handle != 0) {
      nativeDestroy(handle);
      handle = 0;
    }
  }

  private long checkedHandle() {
    if (handle == 0) {
      throw new IllegalStateException("detector is closed");
    }
    return handle;
  }

  private static Detection[] unpack(double[] packed) {
    Detection[] detections = new Detection[packed.length / STRIDE];
    for (int i = 0; i < detections.length; i++) {
      int o = i * STRIDE;
      double[] center = {packed[o + 3], packed[o + 4]};
      double[][] corners = new double[4][];
      for (int c = 0; c < 4; c++) {
        corners[c] = new double[] {packed[o + 5 + 2 * c], packed[o + 6 + 2 * c]};
      }
      double[] homography = new double[9];
      System.arraycopy(packed, o + 13, homography, 0, 9);
      Pose pose = null;
      if (packed[o + 22] != 0.0) {
        double[] rotation = new double[9];
        double[] translation = new double[3];
        System.arraycopy(packed, o + 23, rotation, 0, 9);
        System.arraycopy(packed, o + 32, translation, 0, 3);
        pose = new Pose(rotation, translation);
      }
      detections[i] =
          new Detection(
              (int) packed[o],
              (int) packed[o + 1],
              (float) packed[o + 2],
              center,
              corners,
              homography,
              pose);
    }
    return detections;
  }

  private static native long nativeCreate(
      int width,
      int height,
      double fx,
      double fy,
      double cx,
      double cy,
      double[] distortion,
      int threads,
      double tagSizeMeters);

  private static native double[] nativeDetect(
      long handle, byte[] pixels, int stride, int channels);

  private static native double[] nativeDetectMat(long handle, long matAddress);

  private static native void nativeDestroy(long handle);
}
