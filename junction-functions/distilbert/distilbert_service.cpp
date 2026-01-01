#include <onnxruntime_cxx_api.h>

#include "../junctiond/httplib.h"
#include "../junctiond/json.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using json = nlohmann::json;

namespace {
struct Config {
    std::string model_path;
    std::string host = "0.0.0.0";
    int port = 9000;
};

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--model-path" || arg == "-m") && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if ((arg == "--host" || arg == "-H") && i + 1 < argc) {
            cfg.host = argv[++i];
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            cfg.port = std::stoi(argv[++i]);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (cfg.model_path.empty()) {
        throw std::runtime_error("--model-path is required");
    }
    return cfg;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    if (logits.empty()) return {};
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exps(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }
    for (float& v : exps) v /= sum;
    return exps;
}

std::string label_from_logits(const std::vector<float>& logits) {
    if (logits.size() != 2) return "unknown";
    return logits[1] > logits[0] ? "positive" : "negative";
}
}  // namespace

int main(int argc, char* argv[]) {
    Config cfg;
    try {
        cfg = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Usage: " << argv[0]
                  << " --model-path /path/to/distilbert.onnx [--host 0.0.0.0] [--port 9000]\n"
                  << "Error: " << e.what() << "\n";
        return 1;
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "distilbert_service");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, cfg.model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        std::vector<std::string> input_name_storage;
        std::vector<const char*> input_name_ptrs;
        size_t num_inputs = session.GetInputCount();
        input_name_storage.reserve(num_inputs);
        input_name_ptrs.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_name_storage.emplace_back(name.get());
            input_name_ptrs.push_back(input_name_storage.back().c_str());
        }

        std::vector<std::string> output_name_storage;
        std::vector<const char*> output_name_ptrs;
        size_t num_outputs = session.GetOutputCount();
        output_name_storage.reserve(num_outputs);
        output_name_ptrs.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_name_storage.emplace_back(name.get());
            output_name_ptrs.push_back(output_name_storage.back().c_str());
        }

        httplib::Server svr;
        svr.Post("/infer", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                if (!body.contains("input_ids") || !body.contains("attention_mask")) {
                    res.status = 400;
                    res.set_content("{\"error\":\"input_ids and attention_mask required\"}", "application/json");
                    return;
                }
                const auto& ids_j = body["input_ids"];
                const auto& mask_j = body["attention_mask"];
                if (!ids_j.is_array() || !mask_j.is_array()) {
                    res.status = 400;
                    res.set_content("{\"error\":\"input_ids and attention_mask must be arrays\"}", "application/json");
                    return;
                }
                if (ids_j.size() != mask_j.size() || ids_j.empty()) {
                    res.status = 400;
                    res.set_content("{\"error\":\"input_ids and attention_mask length mismatch or empty\"}", "application/json");
                    return;
                }

                std::vector<int64_t> input_ids;
                std::vector<int64_t> attention_mask;
                input_ids.reserve(ids_j.size());
                attention_mask.reserve(mask_j.size());
                for (size_t i = 0; i < ids_j.size(); ++i) {
                    input_ids.push_back(ids_j.at(i).get<int64_t>());
                    attention_mask.push_back(mask_j.at(i).get<int64_t>());
                }

                const int64_t seq_len = static_cast<int64_t>(input_ids.size());
                std::array<int64_t, 2> input_shape{1, seq_len};
                Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t>(
                    mem_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());
                Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
                    mem_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());

                std::array<Ort::Value, 2> input_tensors{
                    std::move(ids_tensor),
                    std::move(mask_tensor)
                };

                auto outputs = session.Run(
                    Ort::RunOptions{nullptr},
                    input_name_ptrs.data(),
                    input_tensors.data(),
                    input_tensors.size(),
                    output_name_ptrs.data(),
                    output_name_ptrs.size());

                if (outputs.empty()) {
                    throw std::runtime_error("No outputs returned from model");
                }

                auto& logits_tensor = outputs.front();
                float* logits_data = logits_tensor.GetTensorMutableData<float>();
                auto type_info = logits_tensor.GetTensorTypeAndShapeInfo();
                auto shape = type_info.GetShape();
                if (shape.size() != 2 || shape[0] != 1) {
                    throw std::runtime_error("Unexpected logits shape");
                }

                int64_t num_classes = shape[1];
                std::vector<float> logits(num_classes);
                for (int64_t i = 0; i < num_classes; ++i) {
                    logits[i] = logits_data[i];
                }

                std::vector<float> probs = softmax(logits);
                std::string label = label_from_logits(logits);

                json resp{{"logits", logits}, {"probs", probs}, {"label", label}};
                res.set_content(resp.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                json err{{"error", e.what()}};
                res.set_content(err.dump(), "application/json");
            }
        });

        std::cout << "distilbert_service listening on " << cfg.host << ":" << cfg.port << "\n";
        svr.listen(cfg.host, cfg.port);
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
