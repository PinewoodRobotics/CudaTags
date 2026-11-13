// JNI for frc4765.cuda.tags.CudaTags over the shared C++ core.
#include <jni.h>

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "cuda_tags/detector.hpp"

namespace {

constexpr int kStride = 35;  // must match CudaTags.STRIDE

void Throw(JNIEnv *env, const char *class_name, const char *message) {
  jclass type = env->FindClass(class_name);
  if (type != nullptr) env->ThrowNew(type, message);
}

// Converts the in-flight C++ exception into a Java one.
void ThrowJava(JNIEnv *env) {
  try {
    throw;
  } catch (const std::invalid_argument &error) {
    Throw(env, "java/lang/IllegalArgumentException", error.what());
  } catch (const std::exception &error) {
    Throw(env, "java/lang/RuntimeException", error.what());
  } catch (...) {
    Throw(env, "java/lang/RuntimeException", "unknown C++ exception");
  }
}

cuda_tags::Detector *FromHandle(jlong handle) {
  return reinterpret_cast<cuda_tags::Detector *>(static_cast<intptr_t>(handle));
}

cuda_tags::PixelFormat FormatFor(int channels, bool rgb = false) {
  switch (channels) {
    case 1:
      return cuda_tags::PixelFormat::kGray;
    case 3:
      return rgb ? cuda_tags::PixelFormat::kRgb : cuda_tags::PixelFormat::kBgr;
    case 4:
      return rgb ? cuda_tags::PixelFormat::kRgba : cuda_tags::PixelFormat::kBgra;
  }
  throw std::invalid_argument("channels must be 1, 3 or 4");
}

jdoubleArray Pack(JNIEnv *env,
                  const std::vector<cuda_tags::Detection> &detections) {
  std::vector<jdouble> packed(detections.size() * kStride, 0.0);
  for (size_t i = 0; i < detections.size(); ++i) {
    const cuda_tags::Detection &tag = detections[i];
    jdouble *out = packed.data() + i * kStride;
    out[0] = tag.id;
    out[1] = tag.hamming;
    out[2] = tag.decision_margin;
    out[3] = tag.center[0];
    out[4] = tag.center[1];
    for (int c = 0; c < 4; ++c) {
      out[5 + 2 * c] = tag.corners[c][0];
      out[6 + 2 * c] = tag.corners[c][1];
    }
    for (int k = 0; k < 9; ++k) out[13 + k] = tag.homography[k];
    if (tag.pose) {
      out[22] = 1.0;
      for (int k = 0; k < 9; ++k) out[23 + k] = tag.pose->rotation[k];
      for (int k = 0; k < 3; ++k) out[32 + k] = tag.pose->translation[k];
    }
  }
  jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(packed.size()));
  if (result != nullptr) {
    env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(packed.size()),
                              packed.data());
  }
  return result;
}

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL
Java_frc4765_cuda_tags_CudaTags_cudaAvailable(JNIEnv *, jclass) {
  return cuda_tags::CudaAvailable() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL Java_frc4765_cuda_tags_CudaTags_nativeCreate(
    JNIEnv *env, jclass, jint width, jint height, jdouble fx, jdouble fy,
    jdouble cx, jdouble cy, jdoubleArray distortion, jint threads,
    jdouble tag_size_m) {
  try {
    if (width <= 0 || height <= 0) {
      throw std::invalid_argument("width and height must be positive");
    }
    if (distortion == nullptr || env->GetArrayLength(distortion) != 5) {
      throw std::invalid_argument("distortion must hold k1, k2, p1, p2, k3");
    }
    jdouble k[5];
    env->GetDoubleArrayRegion(distortion, 0, 5, k);
    cuda_tags::Options options;
    options.width = static_cast<uint32_t>(width);
    options.height = static_cast<uint32_t>(height);
    options.threads = threads;
    options.tag_size_m = tag_size_m;
    auto *detector = new cuda_tags::Detector(
        cuda_tags::Camera{fx, fy, cx, cy},
        cuda_tags::Distortion{k[0], k[1], k[2], k[3], k[4]}, options);
    return static_cast<jlong>(reinterpret_cast<intptr_t>(detector));
  } catch (...) {
    ThrowJava(env);
  }
  return 0;
}

JNIEXPORT jdoubleArray JNICALL Java_frc4765_cuda_tags_CudaTags_nativeDetect(
    JNIEnv *env, jclass, jlong handle, jbyteArray pixels, jint stride,
    jint channels, jboolean rgb) {
  try {
    cuda_tags::Detector *detector = FromHandle(handle);
    const cuda_tags::Options &options = detector->options();
    if (stride <= 0) throw std::invalid_argument("stride must be positive");
    const jsize length = env->GetArrayLength(pixels);
    // Copy out of the Java heap (never pin it across GPU work), straight
    // into the detector's pinned input buffer when it fits, so the GPU then
    // reads the frame in place.
    std::vector<uint8_t> fallback;
    uint8_t *data = detector->InputBuffer();
    if (data == nullptr ||
        static_cast<size_t>(length) > detector->InputBufferSize()) {
      fallback.resize(static_cast<size_t>(length));
      data = fallback.data();
    }
    env->GetByteArrayRegion(pixels, 0, length,
                            reinterpret_cast<jbyte *>(data));
    cuda_tags::ImageView view;
    view.data = data;
    view.size = static_cast<size_t>(length);
    view.width = options.width;
    view.height = options.height;
    view.stride = static_cast<size_t>(stride);
    view.format = FormatFor(channels, rgb == JNI_TRUE);
    return Pack(env, detector->Detect(view));
  } catch (...) {
    ThrowJava(env);
  }
  return nullptr;
}

JNIEXPORT void JNICALL Java_frc4765_cuda_tags_CudaTags_nativeSetColorConversion(
    JNIEnv *env, jclass, jlong handle, jboolean before_detecting) {
  try {
    FromHandle(handle)->SetColorConversion(
        before_detecting == JNI_TRUE
            ? cuda_tags::ColorConversion::kBeforeDetecting
            : cuda_tags::ColorConversion::kWhileDetecting);
  } catch (...) {
    ThrowJava(env);
  }
}

JNIEXPORT jdoubleArray JNICALL Java_frc4765_cuda_tags_CudaTags_nativeDetectMat(
    JNIEnv *env, jclass, jlong handle, jlong mat_address) {
  try {
    if (mat_address == 0) throw std::invalid_argument("Mat address is 0");
    cuda_tags::Detector *detector = FromHandle(handle);
    const cv::Mat &mat =
        *reinterpret_cast<const cv::Mat *>(static_cast<intptr_t>(mat_address));
    if (mat.depth() != CV_8U || mat.dims != 2) {
      throw std::invalid_argument("Mat must be a 2-D 8-bit image");
    }
    cuda_tags::ImageView view;
    view.data = mat.data;
    view.width = static_cast<uint32_t>(mat.cols);
    view.height = static_cast<uint32_t>(mat.rows);
    view.stride = mat.step[0];
    view.size = mat.rows > 0 ? (mat.rows - 1) * mat.step[0] +
                                   mat.cols * mat.elemSize()
                             : 0;
    view.format = FormatFor(mat.channels());
    return Pack(env, detector->Detect(view));
  } catch (...) {
    ThrowJava(env);
  }
  return nullptr;
}

JNIEXPORT void JNICALL Java_frc4765_cuda_tags_CudaTags_nativeDestroy(
    JNIEnv *, jclass, jlong handle) {
  delete FromHandle(handle);
}

}  // extern "C"
