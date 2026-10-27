#include <memory>
#include <sstream>

#include <gtest/gtest.h>

#include "inference-addon-cpp/Errors.hpp"
#include "model-interface/LlamaModelLoader.hpp"

TEST(
    AsyncWeightsLoaderTest,
    ShardedStreamingRejectsTensorsBeforeFirstGgufShard) {
  LlamaModelLoader loader(
      {}, "model-00001-of-00003.gguf", {{"device", "cpu"}}, "");
  EXPECT_THROW(
      loader.setWeightsForFile(
          "model.tensors.txt", std::make_unique<std::stringbuf>("tensors")),
      qvac_errors::StatusError);
  EXPECT_THROW(
      loader.setWeightsForFile(
          "model-00001-of-00003.gguf",
          std::make_unique<std::stringbuf>("weights")),
      qvac_errors::StatusError);
  EXPECT_FALSE(loader.isLoaded());
}
