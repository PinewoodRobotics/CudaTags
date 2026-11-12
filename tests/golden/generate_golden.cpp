// Renders the golden test image: three tag36h11 tags at known poses seen by
// a known pinhole camera, with exact ground-truth corners and poses.
//
//   generate_golden <out.pgm> <out.txt>
//
// The committed tests/golden/tags_640x480.{pgm,txt} came from this program;
// rerun it only when changing the scene.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "golden/golden_scene.h"

namespace {

constexpr int kWidth = 640, kHeight = 480;
constexpr double kFx = 600, kFy = 600, kCx = 320, kCy = 240;
constexpr double kTagSize = 0.1651;  // 6.5 in, the FRC tag size
constexpr int kSupersample = 8;
constexpr uint8_t kBackground = 128;

struct TagPlacement {
  int id;
  double yaw_deg, pitch_deg, roll_deg;  // about camera y, x, z
  double x, y, z;                       // meters
};

cv::Matx33d Rotation(double yaw_deg, double pitch_deg, double roll_deg) {
  const double d = M_PI / 180.0;
  const double cy = std::cos(yaw_deg * d), sy = std::sin(yaw_deg * d);
  const double cp = std::cos(pitch_deg * d), sp = std::sin(pitch_deg * d);
  const double cr = std::cos(roll_deg * d), sr = std::sin(roll_deg * d);
  const cv::Matx33d ry(cy, 0, sy, 0, 1, 0, -sy, 0, cy);
  const cv::Matx33d rx(1, 0, 0, 0, cp, -sp, 0, sp, cp);
  const cv::Matx33d rz(cr, -sr, 0, sr, cr, 0, 0, 0, 1);
  return ry * rx * rz;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: generate_golden <out.pgm> <out.txt>\n";
    return 2;
  }
  const std::vector<TagPlacement> placements = {
      {0, 0.0, 0.0, 0.0, -0.30, -0.10, 1.3},
      {7, 35.0, 0.0, 0.0, 0.25, -0.05, 1.1},
      {42, 0.0, -25.0, 30.0, 0.0, 0.17, 0.9},
  };
  const cv::Matx33d camera(kFx, 0, kCx, 0, kFy, kCy, 0, 0, 1);
  apriltag_family_t *family = tag36h11_create();
  const double cell = kTagSize / family->width_at_border;
  const double center = family->total_width / 2.0;

  cv::Mat image(kHeight, kWidth, CV_8UC1, cv::Scalar(kBackground));
  std::ofstream text(argv[2]);
  text << std::setprecision(12);
  text << "# Golden scene for tests/golden/tags_640x480.pgm, written by "
          "tests/golden/generate_golden.cpp.\n";
  text << "image " << kWidth << " " << kHeight << "\n";
  text << "camera " << kFx << " " << kFy << " " << kCx << " " << kCy << "\n";
  text << "tag_size " << kTagSize << "\n";

  for (const TagPlacement &placement : placements) {
    image_u8_t *texture = apriltag_to_image(family, placement.id);
    const cv::Matx33d rotation =
        Rotation(placement.yaw_deg, placement.pitch_deg, placement.roll_deg);
    const cv::Vec3d translation(placement.x, placement.y, placement.z);
    // Plane-to-image homography for tag-frame points (X, Y, 0).
    const cv::Matx33d plane(rotation(0, 0), rotation(0, 1), translation[0],
                            rotation(1, 0), rotation(1, 1), translation[1],
                            rotation(2, 0), rotation(2, 1), translation[2]);
    const cv::Matx33d to_image = camera * plane;
    const cv::Matx33d to_plane = to_image.inv();

    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        int covered = 0, sum = 0;
        for (int sy = 0; sy < kSupersample; ++sy) {
          for (int sx = 0; sx < kSupersample; ++sx) {
            // Pixel centers sit at integer coordinates.
            const double px = x - 0.5 + (sx + 0.5) / kSupersample;
            const double py = y - 0.5 + (sy + 0.5) / kSupersample;
            const cv::Vec3d p = to_plane * cv::Vec3d(px, py, 1.0);
            if (p[2] == 0.0) continue;
            const double u = center + (p[0] / p[2]) / cell;
            const double v = center + (p[1] / p[2]) / cell;
            if (u < 0 || v < 0 || u >= family->total_width ||
                v >= family->total_width) {
              continue;
            }
            ++covered;
            sum += texture->buf[static_cast<int>(v) * texture->stride +
                                static_cast<int>(u)];
          }
        }
        if (covered == 0) continue;
        const int samples = kSupersample * kSupersample;
        image.at<uint8_t>(y, x) = static_cast<uint8_t>(std::lround(
            (sum + kBackground * (samples - covered)) /
            static_cast<double>(samples)));
      }
    }
    image_u8_destroy(texture);

    const double s = kTagSize / 2.0;
    const double object[4][2] = {{-s, s}, {s, s}, {s, -s}, {-s, -s}};
    text << "tag " << placement.id << " corners";
    for (const auto &point : object) {
      const cv::Vec3d p = to_image * cv::Vec3d(point[0], point[1], 1.0);
      text << " " << p[0] / p[2] << " " << p[1] / p[2];
    }
    text << " rotation";
    for (int i = 0; i < 9; ++i) text << " " << rotation.val[i];
    text << " translation " << translation[0] << " " << translation[1] << " "
         << translation[2] << "\n";
  }
  tag36h11_destroy(family);
  golden::WritePgm(argv[1], kWidth, kHeight, image.data);
  return 0;
}
