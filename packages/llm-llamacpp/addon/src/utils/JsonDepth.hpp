#pragma once
#include <cstddef>
#include <string_view>

namespace qvac_lib_inference_addon_llama::utils {

/// Deepest array/object nesting accepted in request JSON. picojson and
/// fabric's `json_schema_to_grammar` and chat templates recurse once per
/// level with no limit; on a 512 KB thread stack, the default for worker
/// threads on macOS and iOS, fabric overflows past 512 levels.
inline constexpr size_t kMaxJsonDepth = 128;

/// Whether arrays and objects in @p json nest deeper than kMaxJsonDepth.
/// Brackets inside strings are skipped.
inline bool nestsTooDeep(std::string_view json) {
  size_t depth = 0;
  bool inString = false;
  bool escaped = false;
  for (const char c : json) {
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        inString = false;
      }
    } else if (c == '"') {
      inString = true;
    } else if (c == '[' || c == '{') {
      if (++depth > kMaxJsonDepth) {
        return true;
      }
    } else if ((c == ']' || c == '}') && depth > 0) {
      --depth;
    }
  }
  return false;
}

} // namespace qvac_lib_inference_addon_llama::utils
