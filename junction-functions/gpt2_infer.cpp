#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <stdexcept>

// Argmax sampling
int64_t argmax(const float* data, int64_t size) {
    int64_t max_i = 0;
    float max_v = data[0];
    for (int64_t i = 1; i < size; ++i) {
        if (data[i] > max_v) {
            max_v = data[i];
            max_i = i;
        }
    }
    return max_i;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " gpt2.onnx\n";
        return 1;
    }

    const char* model_path = argv[1];

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gpt2");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path, opts);
    Ort::AllocatorWithDefaultOptions allocator;

    // ---- Model metadata ----
    constexpr int NUM_LAYERS = 12;
    constexpr int NUM_HEADS = 12;
    constexpr int HEAD_DIM  = 64;
    constexpr int PAST_SEQ  = 10;

    // ---- Input token (dummy) ----
    std::vector<int64_t> input_ids{50256};  // EOS token
    std::vector<int64_t> attention_mask{1};

    std::array<int64_t, 2> ids_shape{1, 1};

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem, input_ids.data(), input_ids.size(),
        ids_shape.data(), ids_shape.size());

    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        mem, attention_mask.data(), attention_mask.size(),
        ids_shape.data(), ids_shape.size());

    // ---- Past KV cache ----
    std::vector<float> past_data(
        NUM_LAYERS * 2 * NUM_HEADS * PAST_SEQ * HEAD_DIM, 0.0f);

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_ids_tensor));
    inputs.push_back(std::move(attention_mask_tensor));

    std::array<int64_t, 4> past_shape{
        1, NUM_HEADS, PAST_SEQ, HEAD_DIM};

    size_t offset = 0;
    for (int i = 0; i < NUM_LAYERS * 2; ++i) {
        size_t size = NUM_HEADS * PAST_SEQ * HEAD_DIM;
        inputs.emplace_back(
            Ort::Value::CreateTensor<float>(
                mem,
                past_data.data() + offset,
                size,
                past_shape.data(),
                past_shape.size()));
        offset += size;
    }

    // ---- Input / output names ----
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    for (size_t i = 0; i < session.GetInputCount(); ++i)
        input_names.push_back(
            session.GetInputNameAllocated(i, allocator).release());

    for (size_t i = 0; i < session.GetOutputCount(); ++i)
        output_names.push_back(
            session.GetOutputNameAllocated(i, allocator).release());

    // ---- Run inference ----
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        inputs.data(),
        inputs.size(),
        output_names.data(),
        output_names.size());

    // ---- Extract logits ----
    float* logits = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t vocab_size = shape[2];

    int64_t next_token = argmax(logits, vocab_size);

    std::cout << "Next token id: " << next_token << "\n";

    return 0;
}
