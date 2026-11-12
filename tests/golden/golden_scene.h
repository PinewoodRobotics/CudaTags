// Reads the golden scene description written by generate_golden.cpp.
//
// Format (whitespace separated, '#' starts a comment line):
//   image <width> <height>
//   camera <fx> <fy> <cx> <cy>
//   tag_size <meters>
//   tag <id> corners <8 numbers> rotation <9 numbers> translation <3 numbers>
// Corners are ground-truth pixels (pixel centers at integer coordinates) in
// AprilTag's order; rotation/translation are the camera-from-tag pose in
// AprilTag's tag frame.
#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace golden {

struct Tag {
  int id = 0;
  std::array<std::array<double, 2>, 4> corners{};
  std::array<double, 9> rotation{};
  std::array<double, 3> translation{};
};

struct Scene {
  int width = 0, height = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  double tag_size_m = 0;
  std::vector<Tag> tags;
};

inline void Expect(std::istream &in, const char *word) {
  std::string token;
  if (!(in >> token) || token != word) {
    throw std::runtime_error(std::string("golden scene: expected '") + word +
                             "', got '" + token + "'");
  }
}

// 8-bit binary PGM (P5), so the tests need no image codec library.
struct GrayImage {
  int width = 0, height = 0;
  std::vector<uint8_t> pixels;
};

inline GrayImage ReadPgm(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  GrayImage image;
  int max_value = 0;
  if (!(file >> magic >> image.width >> image.height >> max_value) ||
      magic != "P5" || max_value != 255 || image.width <= 0 ||
      image.height <= 0) {
    throw std::runtime_error("cannot read 8-bit binary PGM " + path);
  }
  file.get();  // the single whitespace byte after the header
  image.pixels.resize(static_cast<size_t>(image.width) * image.height);
  if (!file.read(reinterpret_cast<char *>(image.pixels.data()),
                 static_cast<std::streamsize>(image.pixels.size()))) {
    throw std::runtime_error("truncated PGM " + path);
  }
  return image;
}

inline void WritePgm(const std::string &path, int width, int height,
                     const uint8_t *pixels) {
  std::ofstream file(path, std::ios::binary);
  file << "P5\n" << width << " " << height << "\n255\n";
  file.write(reinterpret_cast<const char *>(pixels),
             static_cast<std::streamsize>(width) * height);
  if (!file) throw std::runtime_error("cannot write " + path);
}

inline Scene ReadScene(const std::string &path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open " + path);
  Scene scene;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream in(line);
    std::string kind;
    in >> kind;
    if (kind == "image") {
      in >> scene.width >> scene.height;
    } else if (kind == "camera") {
      in >> scene.fx >> scene.fy >> scene.cx >> scene.cy;
    } else if (kind == "tag_size") {
      in >> scene.tag_size_m;
    } else if (kind == "tag") {
      Tag tag;
      in >> tag.id;
      Expect(in, "corners");
      for (auto &corner : tag.corners) in >> corner[0] >> corner[1];
      Expect(in, "rotation");
      for (double &value : tag.rotation) in >> value;
      Expect(in, "translation");
      for (double &value : tag.translation) in >> value;
      scene.tags.push_back(tag);
    } else {
      throw std::runtime_error("golden scene: unknown line '" + line + "'");
    }
    if (in.fail()) throw std::runtime_error("golden scene: bad line '" + line + "'");
  }
  return scene;
}

}  // namespace golden
