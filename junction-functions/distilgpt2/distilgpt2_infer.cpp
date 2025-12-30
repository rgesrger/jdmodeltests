#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <array>

// Simple argmax function
int64_t argmax(const float* data, int64_t size) {
    int64_t idx = 0;
    float maxv = data[0];
    for (int64_t i = 1; i < size; ++i) {
        if (data[i] > maxv) {
            maxv = data[i];
            idx = i;
        }
    }
    return idx;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " distilgpt2.onnx\n";
        return 1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "distilgpt2");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, argv[1], opts);
    Ort::AllocatorWithDefaultOptions allocator;

    // Model configuration
    constexpr int NUM_LAYERS = 6;
    constexpr int NUM_HEADS = 12;
    constexpr int HEAD_DIM = 64;
    constexpr int PAST_SEQ = 10;
    constexpr int CUR_SEQ = 1;

    // Token input
    std::vector<int64_t> input_ids{50256};
    std::vector<int64_t> attention_mask(PAST_SEQ + CUR_SEQ, 1); // Correct mask length

    // Input shapes
    std::array<int64_t, 2> input_shape{1, CUR_SEQ};
    std::array<int64_t, 2> mask_shape{1, PAST_SEQ + CUR_SEQ};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Prepare inputs
    std::vector<Ort::Value> inputs;

    inputs.emplace_back(
        Ort::Value::CreateTensor<int64_t>(mem, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()));

    inputs.emplace_back(
        Ort::Value::CreateTensor<int64_t>(mem, attention_mask.data(), attention_mask.size(), mask_shape.data(), mask_shape.size()));

    // Past key/value cache
    std::vector<float> past_data(NUM_LAYERS * 2 * NUM_HEADS * PAST_SEQ * HEAD_DIM, 0.0f);
    std::array<int64_t, 4> past_shape{1, NUM_HEADS, PAST_SEQ, HEAD_DIM};

    size_t offset = 0;
    for (int i = 0; i < NUM_LAYERS * 2; ++i) {
        size_t size = NUM_HEADS * PAST_SEQ * HEAD_DIM;
        inputs.emplace_back(
            Ort::Value::CreateTensor<float>(mem, past_data.data() + offset, size, past_shape.data(), past_shape.size()));
        offset += size;
    }

    // Input and output names must exactly match ONNX export
    std::vector<const char*> input_names = {
        "input_ids",
        "attention_mask",
        "past.0.key", "past.0.value",
        "past.1.key", "past.1.value",
        "past.2.key", "past.2.value",
        "past.3.key", "past.3.value",
        "past.4.key", "past.4.value",
        "past.5.key", "past.5.value"
    };

    std::vector<const char*> output_names = {
        "logits",
        "present.0.key", "present.0.value",
        "present.1.key", "present.1.value",
        "present.2.key", "present.2.value",
        "present.3.key", "present.3.value",
        "present.4.key", "present.4.value",
        "present.5.key", "present.5.value"
    };

    // Run the model
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        inputs.data(),
        inputs.size(),
        output_names.data(),
        output_names.size()
    );

    // Get logits and compute next token
    auto shape_out = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t vocab_size = shape_out[2];  // [batch, seq, vocab]

    float* logits = outputs[0].GetTensorMutableData<float>();
    float* last_logits = logits + (shape_out[1] - 1) * vocab_size;

    int64_t next_token = argmax(last_logits, vocab_size);
    std::cout << "Next token id: " << next_token << "\n";

    return 0;
}
