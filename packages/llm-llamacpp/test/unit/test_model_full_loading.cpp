#include <filesystem>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "model-interface/LlamaModel.hpp"
#include "test_common.hpp"

namespace fs = std::filesystem;

class ModelFullLoadingTest : public ::testing::Test {
protected:
  void SetUp() override {
    using MP = test_common::TestModelPath;

    config_["device"] = test_common::getTestDevice();
    config_["ctx_size"] = "2048";
    config_["gpu_layers"] = test_common::getTestGpuLayers();
    config_["n_predict"] = "10";

    config_["backendsDir"] = test_common::getTestBackendsDir().string();

    singleModel_ =
        MP("Llama-3.2-1B-Instruct-Q4_0.gguf", nullptr, MP::OnMissing::Fail, "");

    shardedModel_ =
        MP("Qwen3-0.6B-UD-IQ1_S-00001-of-00003.gguf",
           "SHARDED_MODEL_FIRST_SHARD_PATH",
           MP::OnMissing::Fail,
           "https://huggingface.co/jmb95/Qwen3-0.6B-UD-IQ1_S-sharded",
           true /* isSharded */);

    largeShardedModel_ =
        MP("Llama-3.2-1B-Instruct-Q4_0-00001-of-00008.gguf",
           "LARGE_SHARDED_MODEL_FIRST_SHARD_PATH",
           MP::OnMissing::Skip,
           "https://huggingface.co/jmb95/Llama-3.2-1B-Instruct-Q4_0-sharded",
           true /* isSharded */);
  }

  LlamaModel loadModel(const std::string& modelPath) {
    std::string path = modelPath;
    std::string projection;
    auto cfg = config_;
    return LlamaModel(std::move(path), std::move(projection), std::move(cfg));
  }

  void streamShardsIntoModel(
      LlamaModel& model, const test_common::TestModelPath& mp) {
    std::string tensorsBasename =
        fs::path(mp.shards.tensors_file).filename().string();
    auto tensorsBuf =
        test_common::readFileToStreambufBinary(mp.shards.tensors_file);
    ASSERT_NE(tensorsBuf, nullptr)
        << "Failed to open: " << mp.shards.tensors_file;
    EXPECT_THROW(
        model.setWeightsForFile(tensorsBasename, std::move(tensorsBuf)),
        qvac_errors::StatusError);

    for (const auto& shardPath : mp.shards.gguf_files) {
      auto streambuf = test_common::readFileToStreambufBinary(shardPath);
      ASSERT_NE(streambuf, nullptr) << "Failed to open shard: " << shardPath;
      EXPECT_THROW(
          model.setWeightsForFile(
              fs::path(shardPath).filename().string(), std::move(streambuf)),
          qvac_errors::StatusError);
    }
  }

  std::unordered_map<std::string, std::string> config_;
  test_common::TestModelPath singleModel_;
  test_common::TestModelPath shardedModel_;
  test_common::TestModelPath largeShardedModel_;
};

TEST_F(ModelFullLoadingTest, SingleFile_LoadsSuccessfully) {
  REQUIRE_MODEL(singleModel_);
  LlamaModel model = loadModel(singleModel_.path);
  model.waitForLoadInitialization();
  EXPECT_TRUE(model.isLoaded());
}

TEST_F(ModelFullLoadingTest, RejectedStreamingSingleFile_StillLoadsFromDisk) {
  REQUIRE_MODEL(singleModel_);
  LlamaModel model = loadModel(singleModel_.path);
  std::string filename = fs::path(singleModel_.path).filename().string();
  auto streambuf = test_common::readFileToStreambufBinary(singleModel_.path);
  ASSERT_NE(streambuf, nullptr) << "Failed to open: " << singleModel_.path;
  EXPECT_THROW(
      model.setWeightsForFile(filename, std::move(streambuf)),
      qvac_errors::StatusError);
  model.waitForLoadInitialization();
  EXPECT_TRUE(model.isLoaded());
}

TEST_F(ModelFullLoadingTest, Sharded_LoadsSuccessfully) {
  REQUIRE_MODEL(shardedModel_);
  LlamaModel model = loadModel(shardedModel_.path);
  model.waitForLoadInitialization();
  EXPECT_TRUE(model.isLoaded());
}

TEST_F(ModelFullLoadingTest, Sharded_LoadFromLaterShardRequiresFirstShard) {
  REQUIRE_MODEL(shardedModel_);
  ASSERT_GE(shardedModel_.shards.gguf_files.size(), 2U);

  LlamaModel model = loadModel(shardedModel_.shards.gguf_files[1]);
  EXPECT_THROW(model.waitForLoadInitialization(), qvac_errors::StatusError);
  EXPECT_FALSE(model.isLoaded());
}

TEST_F(ModelFullLoadingTest, RejectedStreamingShards_StillLoadFromDisk) {
  REQUIRE_MODEL(shardedModel_);
  LlamaModel model = loadModel(shardedModel_.path);
  streamShardsIntoModel(model, shardedModel_);
  model.waitForLoadInitialization();
  EXPECT_TRUE(model.isLoaded());
}

TEST_F(ModelFullLoadingTest, LargeSharded_LoadsSuccessfully) {
  REQUIRE_MODEL(largeShardedModel_);
  LlamaModel model = loadModel(largeShardedModel_.path);
  model.waitForLoadInitialization();
  EXPECT_TRUE(model.isLoaded());
}

TEST_F(ModelFullLoadingTest, RejectedStreamingLargeShards_StillLoadFromDisk) {
  REQUIRE_MODEL(largeShardedModel_);
  LlamaModel model = loadModel(largeShardedModel_.path);
  streamShardsIntoModel(model, largeShardedModel_);
  model.waitForLoadInitialization();
  EXPECT_TRUE(model.isLoaded());
}
