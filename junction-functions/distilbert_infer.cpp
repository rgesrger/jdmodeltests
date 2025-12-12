// distilbert_infer.cpp
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

// Simple softmax over a small vector
std::vector<float> softmax(const std::vector<float>& logits) {
    float max_logit = logits[0];
    for (float v : logits) {
        if (v > max_logit) max_logit = v;
    }
    float sum = 0.0f;
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (float& p : probs) {
        p /= sum;
    }
    return probs;
}

// Helper to print sentiment
std::string label_from_logits(const std::vector<float>& logits) {
    // For SST-2: index 0 = NEGATIVE, index 1 = POSITIVE
    if (logits.size() != 2) return "unknown";
    return logits[1] > logits[0] ? "POSITIVE" : "NEGATIVE";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " /path/to/distilbert.onnx\n";
        return 1;
    }

    std::string model_path = argv[1];

    try {
        // 1. Create environment and session options
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "distilbert_infer");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 2. Create session
        Ort::Session session(env, model_path.c_str(), session_options);

        // 3. Describe input
        Ort::AllocatorWithDefaultOptions allocator;

        // Get input names
        size_t num_input_nodes = session.GetInputCount();
        std::vector<const char*> input_names(num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            char* name = session.GetInputNameAllocated(i, allocator).release();
            input_names[i] = name;
            std::cerr << "Input " << i << " name: " << input_names[i] << "\n";
        }

        // For DistilBERT SST-2 ONNX exported as described:
        // input_names should be {"input_ids", "attention_mask"}
        // Output should be "logits"
        size_t num_output_nodes = session.GetOutputCount();
        std::vector<const char*> output_names(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            char* name = session.GetOutputNameAllocated(i, allocator).release();
            output_names[i] = name;
            std::cerr << "Output " << i << " name: " << output_names[i] << "\n";
        }

        // For now, we hard-code a single pre-tokenized example:
        // Suppose tokenizer("this movie was great") produced something like:
        // [101, 2023, 3185, 2001, 2307, 102, 0, 0, ...] with padding
        // In reality, you will compute input_ids & attention_mask in Python
        // and send them to this program or embed them.
        const int64_t seq_len = 8;  // small demo; real models use 128, 256, etc.

        std::vector<int64_t> input_ids = {
            101, 2023, 3185, 2001, 2307, 102, 0, 0
        };
        std::vector<int64_t> attention_mask = {
            1, 1, 1, 1, 1, 1, 0, 0
        };

        if (input_ids.size() != seq_len || attention_mask.size() != seq_len) {
            throw std::runtime_error("Mismatched seq_len in demo input");
        }

        // 4. Create input tensors
        std::array<int64_t, 2> input_shape{1, seq_len};  // batch_size=1

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtDeviceAllocator, OrtMemTypeCPU);

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input_ids.data(),
            input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            attention_mask.data(),
            attention_mask.size(),
            input_shape.data(),
            input_shape.size()
        );

        std::array<Ort::Value, 2> input_tensors{
            std::move(input_ids_tensor),
            std::move(attention_mask_tensor)
        };

        // 5. Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        if (output_tensors.size() != 1) {
            throw std::runtime_error("Expected a single output tensor");
        }

        // 6. Extract logits
        Ort::Value& logits_tensor = output_tensors[0];

        float* logits_data = logits_tensor.GetTensorMutableData<float>();
        auto type_info = logits_tensor.GetTensorTypeAndShapeInfo();
        auto output_shape = type_info.GetShape();

        if (output_shape.size() != 2 || output_shape[0] != 1) {
            throw std::runtime_error("Unexpected logits shape");
        }

        int64_t num_classes = output_shape[1];
        std::vector<float> logits(num_classes);
        for (int64_t i = 0; i < num_classes; ++i) {
            logits[i] = logits_data[i];
        }

        std::vector<float> probs = softmax(logits);
        std::string label = label_from_logits(logits);

        std::cout << "Logits: ";
        for (float v : logits) std::cout << v << " ";
        std::cout << "\nProbs: ";
        for (float p : probs) std::cout << p << " ";
        std::cout << "\nPredicted label: " << label << "\n";

        // Free the names we allocated
        for (auto name : input_names) {
            allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(name)));
        }
        for (auto name : output_names) {
            allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(name)));
        }

        return 0;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
