#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm> // For std::fill
#include <chrono>

// Helper to calculate product of a shape vector
int64_t GetElementCount(const std::vector<int64_t>& shape) {
    int64_t count = 1;
    for (int64_t dim : shape) count *= dim;
    return count;
}

int main(int argc, char* argv[]) {
    std::cout << "ENTERED MAIN" << std::endl; // ADD THIS
    auto start = std::chrono::high_resolution_clock::now();
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " distilgpt2.onnx\n";
        return 1;
    }

    // 1. Setup Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "distilgpt2");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, argv[1], opts);
    Ort::AllocatorWithDefaultOptions allocator;

    std::cout << "READY" << std::endl;
    std::cout.flush();

    // 2. Define Input Data
    std::vector<int64_t> current_token_id = {50256};
    std::vector<int64_t> current_mask = {1}; // Mask size 1 for inference

    // 3. Auto-Discovery Variables
    std::vector<std::string> input_names_strings;
    std::vector<std::vector<int64_t>> input_shapes;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    size_t num_inputs = session.GetInputCount();

    // --- PASS 1: Collect Names and Shapes (Stabilize Memory) ---
    for (size_t i = 0; i < num_inputs; i++) {
        // Get name
        auto name_ptr = session.GetInputNameAllocated(i, allocator);
        input_names_strings.push_back(name_ptr.get());

        // Get shape
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensor_info.GetShape();

        // Fix dynamic dimensions (-1 -> 1)
        for (size_t j = 0; j < shape.size(); j++) {
            if (shape[j] < 0) shape[j] = 1;
        }
        input_shapes.push_back(shape);
    }

    // --- PASS 2: Create Pointers and Tensors ---
    std::vector<const char*> input_names_ptrs;
    std::vector<Ort::Value> input_tensors;

    for (size_t i = 0; i < num_inputs; i++) {
        // Point to the stable string in the vector
        input_names_ptrs.push_back(input_names_strings[i].c_str());

        // Create Tensor based on Name
        if (input_names_strings[i] == "input_ids") {
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                mem, current_token_id.data(), current_token_id.size(), 
                input_shapes[i].data(), input_shapes[i].size()));
        } 
        else if (input_names_strings[i] == "attention_mask") {
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                mem, current_mask.data(), current_mask.size(), 
                input_shapes[i].data(), input_shapes[i].size()));
        }
        else {
            // PAST KEY/VALUES (Zero-filled)
            // Use Ort to allocate memory for us so we don't need to manage C++ vectors
            Ort::Value val = Ort::Value::CreateTensor<float>(
                allocator, input_shapes[i].data(), input_shapes[i].size());
            
            // Fill with zeros
            float* data_ptr = val.GetTensorMutableData<float>();
            size_t element_count = GetElementCount(input_shapes[i]);
            std::fill(data_ptr, data_ptr + element_count, 0.0f);
            
            input_tensors.push_back(std::move(val));
        }
    }

    // 4. Output Setup (Same 2-pass logic for safety, though less critical here)
    std::vector<std::string> output_names_strings;
    std::vector<const char*> output_names_ptrs;
    size_t num_outputs = session.GetOutputCount();

    for(size_t i = 0; i < num_outputs; i++) {
        auto name_ptr = session.GetOutputNameAllocated(i, allocator);
        output_names_strings.push_back(name_ptr.get());
    }
    for(size_t i = 0; i < num_outputs; i++) {
        output_names_ptrs.push_back(output_names_strings[i].c_str());
    }

    // 5. Run Inference
    try {
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_ptrs.data(),
            output_names_ptrs.size()
        );

        // 6. Print Result
        float* logits = outputs[0].GetTensorMutableData<float>();
        auto shape_out = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocab_size = shape_out[2];
        
        float* last_logits = logits + (shape_out[1] - 1) * vocab_size;
        int64_t best_idx = 0;
        float max_val = last_logits[0];
        
        for(int64_t i=1; i<vocab_size; ++i){
            if(last_logits[i] > max_val){
                max_val = last_logits[i];
                best_idx = i;
            }
        }
        std::cout << "Next token id: " << best_idx << std::endl;;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << "\n";
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Model runtime: " << elapsed.count() << " seconds" << std::endl;
    return 0;
}