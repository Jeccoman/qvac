#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "OcrTypes.hpp"

namespace qvac_lib_infer_ocr_ggml::ocr_fit {
struct OcrFitFootprint {
  uint64_t sourceBytes{};
  uint64_t elements{};
};

struct FitDevice {
  std::string id;
  std::string name;
  bool unified{};
  uint64_t usedBytes{};
  uint64_t freeBytes{};
  uint64_t totalBytes{};
};

struct OcrFitResult {
  // "fits", "does-not-fit", or "unknown" when no reliable verdict exists.
  std::string status{"unknown"};
  std::vector<FitDevice> devices;
};

// Inspect tensor descriptors only; no tensor data or image is loaded.
std::optional<OcrFitFootprint>
inspectOcrGguf(const std::string& path, const char* requiredTensor);

// Advisory weight-load check. A model that cannot be read, or no backend
// device, throws.
OcrFitResult assessOcrFit(
    const std::string& detectorPath, const std::string& recognizerPath,
    const OcrConfig& config, uint64_t marginBytes);

} // namespace qvac_lib_infer_ocr_ggml::ocr_fit
