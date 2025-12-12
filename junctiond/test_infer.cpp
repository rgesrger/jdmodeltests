#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Resolve binary and model paths. Allow overrides via env vars or CLI.
    std::string infer_bin = std::getenv("DISTILBERT_BIN")
        ? std::getenv("DISTILBERT_BIN")
        : std::string("../../C-and-D-final/junction-functions/distilbert_infer");
    std::string model_path = std::getenv("DISTILBERT_ONNX")
        ? std::getenv("DISTILBERT_ONNX")
        : std::string("../../C-and-D-final/models/distilbert-finetuned/distilbert.onnx");

    if (argc >= 2) {
        infer_bin = argv[1];
    }
    if (argc >= 3) {
        model_path = argv[2];
    }

    if (!std::filesystem::exists(infer_bin)) {
        std::cerr << "distilbert_infer not found at: " << infer_bin << "\n";
        std::cerr << "Set DISTILBERT_BIN or pass as first argument." << "\n";
        return 1;
    }
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "ONNX model not found at: " << model_path << "\n";
        std::cerr << "Set DISTILBERT_ONNX or pass as second argument." << "\n";
        return 1;
    }

    std::cout << "Running inference: " << infer_bin << " " << model_path << "\n";
    std::string cmd = infer_bin + " " + model_path;
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "Inference process failed with code " << rc << "\n";
        return 1;
    }
    std::cout << "Inference completed successfully." << "\n";
    return 0;
}
