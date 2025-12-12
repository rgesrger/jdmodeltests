#include "../junctiond/httplib.h"
#include "../junctiond/json.hpp"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#include "junctiond.h"

using json = nlohmann::json;

namespace {
struct Config {
    std::string model_path;
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string handler_path;        // distilbert_infer binary (cold path)
    std::string service_path;        // distilbert_service binary (warm path)
    std::string junction_run_path;   // junction_run binary
    int warm_port = 9000;            // port for warm service inside junction
};

std::string default_handler_path(const char* argv0) {
    std::filesystem::path bin_path = std::filesystem::absolute(argv0).parent_path();
    return (bin_path / "distilbert_infer").string();
}

std::string default_service_path(const char* argv0) {
    std::filesystem::path bin_path = std::filesystem::absolute(argv0).parent_path();
    return (bin_path / "distilbert_service").string();
}

std::string default_junction_run_path() {
    const char* home = std::getenv("HOME");
    if (!home) return "/users/nathanan/junction/build/junction/junction_run";
    return (std::filesystem::path(home) / "junction/build/junction/junction_run").string();
}

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
        } else if ((arg == "--handler-path" || arg == "-b") && i + 1 < argc) {
            cfg.handler_path = argv[++i];
        } else if (arg == "--service-path" && i + 1 < argc) {
            cfg.service_path = argv[++i];
        } else if (arg == "--junction-run" && i + 1 < argc) {
            cfg.junction_run_path = argv[++i];
        } else if (arg == "--warm-port" && i + 1 < argc) {
            cfg.warm_port = std::stoi(argv[++i]);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (cfg.model_path.empty()) {
        throw std::runtime_error("--model-path is required");
    }
    return cfg;
}

std::string to_space_separated(const std::vector<int64_t>& vals) {
    std::ostringstream os;
    for (size_t i = 0; i < vals.size(); ++i) {
        if (i) os << ' ';
        os << vals[i];
    }
    return os.str();
}

struct CommandResult {
    int exit_code = -1;
    std::string stdout_output;
    std::string stderr_output;
};

CommandResult exec_and_capture(const std::vector<std::string>& args) {
    if (args.empty()) throw std::runtime_error("No command provided");

    int stdout_pipe[2];
    int stderr_pipe[2];
    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        throw std::runtime_error("Failed to create pipes");
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        throw std::runtime_error("fork() failed");
    }

    if (pid == 0) {
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);

        std::vector<char*> c_args;
        c_args.reserve(args.size() + 1);
        for (const auto& a : args) {
            c_args.push_back(const_cast<char*>(a.c_str()));
        }
        c_args.push_back(nullptr);
        execvp(c_args[0], c_args.data());
        _exit(127);
    }

    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    CommandResult result;
    char buffer[1024];
    ssize_t n;
    while ((n = read(stdout_pipe[0], buffer, sizeof(buffer))) > 0) {
        result.stdout_output.append(buffer, buffer + n);
    }
    while ((n = read(stderr_pipe[0], buffer, sizeof(buffer))) > 0) {
        result.stderr_output.append(buffer, buffer + n);
    }
    close(stdout_pipe[0]);
    close(stderr_pipe[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else {
        result.exit_code = -1;
    }

    return result;
}

std::string write_temp_config(const std::string& name) {
    std::filesystem::path cfg_path = std::filesystem::path("/tmp") / ("junction_" + name + ".config");
    std::ofstream cfg(cfg_path);
    if (!cfg.is_open()) {
        throw std::runtime_error("Failed to open config file: " + cfg_path.string());
    }

    cfg << "host_addr 192.168.127.7\n";
    cfg << "host_netmask 255.255.255.0\n";
    cfg << "host_gateway 192.168.127.1\n";
    cfg << "runtime_kthreads 10\n";
    cfg << "runtime_spinning_kthreads 0\n";
    cfg << "runtime_guaranteed_kthreads 0\n";
    cfg << "runtime_priority lc\n";
    cfg << "runtime_quantum_us 0\n";

    return cfg_path.string();
}

std::atomic<uint64_t> request_counter{0};

// Track a single warm instance name and simple state.
struct WarmState {
    std::string name = "distilbert-warm";
    bool started = false;
};

json run_distilbert_once(const Config& cfg, const std::string& ids_str, const std::string& mask_str) {
    std::string instance = "infer_" + std::to_string(request_counter.fetch_add(1));
    std::string cfg_path = write_temp_config(instance);

    std::vector<std::string> cmd{
        cfg.junction_run_path,
        cfg_path,
        "--",
        cfg.handler_path,
        cfg.model_path,
        ids_str,
        mask_str,
        "--json"
    };

    CommandResult result = exec_and_capture(cmd);

    std::error_code ec;
    std::filesystem::remove(cfg_path, ec);

    if (result.exit_code != 0) {
        throw std::runtime_error(
            "junction_run failed (code " + std::to_string(result.exit_code) + "): " +
            result.stderr_output + result.stdout_output);
    }

    return json::parse(result.stdout_output);
}

json call_warm_service(const Config& cfg, const std::vector<int64_t>& ids, const std::vector<int64_t>& mask) {
    httplib::Client cli("192.168.127.7", cfg.warm_port);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(10, 0);
    cli.set_write_timeout(10, 0);
    json body{{"input_ids", ids}, {"attention_mask", mask}};
    auto resp = cli.Post("/infer", body.dump(), "application/json");
    if (!resp) throw std::runtime_error("warm service unreachable");
    if (resp->status != 200) {
        throw std::runtime_error("warm service error status " + std::to_string(resp->status));
    }
    return json::parse(resp->body);
}
}  // namespace

int main(int argc, char* argv[]) {
    Config cfg;
    try {
        cfg = parse_args(argc, argv);
        if (cfg.handler_path.empty()) cfg.handler_path = default_handler_path(argv[0]);
        if (cfg.service_path.empty()) cfg.service_path = default_service_path(argv[0]);
        cfg.junction_run_path = default_junction_run_path();

        if (!std::filesystem::exists(cfg.handler_path)) {
            throw std::runtime_error("distilbert_infer not found at " + cfg.handler_path);
        }
        if (!std::filesystem::exists(cfg.junction_run_path)) {
            throw std::runtime_error("junction_run not found at " + cfg.junction_run_path);
        }
        if (!std::filesystem::exists(cfg.service_path)) {
            throw std::runtime_error("distilbert_service not found at " + cfg.service_path);
        }
    } catch (const std::exception& e) {
        std::cerr << "Usage: " << argv[0]
                  << " --model-path /path/to/distilbert.onnx [--host 0.0.0.0] [--port 8080]"
                  << " [--handler-path /path/to/distilbert_infer] [--service-path /path/to/distilbert_service]"
                  << " [--junction-run /path/to/junction_run] [--warm-port 9000]\n"
                  << "Error: " << e.what() << "\n";
        return 1;
    }

    try {
        std::cout << "Gateway config: model='" << cfg.model_path
              << "' handler='" << cfg.handler_path
              << "' service='" << cfg.service_path
              << "' junction_run='" << cfg.junction_run_path
              << "' host=" << cfg.host
              << " port=" << cfg.port
              << " warm_port=" << cfg.warm_port << std::endl;

        JunctionD jd;
        WarmState warm;
        std::mutex warm_mtx;

        httplib::Server svr;

        svr.Post("/spawn", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                if (!body.contains("name") || !body.contains("execpath")) {
                    res.status = 400;
                    res.set_content("{\"error\":\"name and execpath required\"}", "application/json");
                    return;
                }

                FunctionData f{};
                f.name = body["name"].get<std::string>();
                f.execpath = body["execpath"].get<std::string>();
                if (body.contains("args")) f.args = body["args"].get<std::string>();
                if (body.contains("cpu")) f.cpu = body["cpu"].get<int>();
                if (body.contains("memoryMB")) f.memoryMB = body["memoryMB"].get<int>();

                if (body.contains("env") && body["env"].is_object()) {
                    for (auto it = body["env"].begin(); it != body["env"].end(); ++it) {
                        f.env[it.key()] = it.value().get<std::string>();
                    }
                }

                bool ok = jd.spawn(f);
                json resp{{"success", ok}};
                res.set_content(resp.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                json err{{"error", e.what()}};
                res.set_content(err.dump(), "application/json");
            }
        });

        svr.Post("/remove", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                if (!body.contains("name")) {
                    res.status = 400;
                    res.set_content("{\"error\":\"name required\"}", "application/json");
                    return;
                }
                std::string name = body["name"].get<std::string>();
                bool ok = jd.remove(name);
                json resp{{"success", ok}};
                res.set_content(resp.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                json err{{"error", e.what()}};
                res.set_content(err.dump(), "application/json");
            }
        });

        svr.Get("/list", [&](const httplib::Request&, httplib::Response& res) {
            try {
                auto list = jd.list();
                json arr = json::array();
                for (const auto& st : list) {
                    arr.push_back({{"name", st.name}, {"running", st.running}, {"pid", st.pid}});
                }
                res.set_content(arr.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                json err{{"error", e.what()}};
                res.set_content(err.dump(), "application/json");
            }
        });

        // Cold path: per-request cold start via junction_run
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

                std::string ids_str = to_space_separated(input_ids);
                std::string mask_str = to_space_separated(attention_mask);

                json resp = run_distilbert_once(cfg, ids_str, mask_str);
                res.set_content(resp.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                json err{{"error", e.what()}};
                res.set_content(err.dump(), "application/json");
            }
        });

        // Warm path: ensure a long-lived junctiond-managed instance is running distilbert_service, then proxy.
        svr.Post("/infer_warm", [&](const httplib::Request& req, httplib::Response& res) {
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

                // First-time warm start: spawn a junctiond-managed service if not already started.
                {
                    std::lock_guard<std::mutex> lk(warm_mtx);
                    if (!warm.started) {
                        FunctionData f{};
                        f.name = warm.name;
                        f.execpath = cfg.service_path;
                        std::ostringstream args;
                        args << "--model-path " << cfg.model_path << " --host 0.0.0.0 --port " << cfg.warm_port;
                        f.args = args.str();
                        f.cpu = 2;
                        f.memoryMB = 512;
                        bool ok = jd.spawn(f);
                        if (!ok) {
                            res.status = 500;
                            res.set_content("{\"error\":\"failed to spawn warm instance\"}", "application/json");
                            return;
                        }
                        warm.started = true;
                    }
                }

                std::vector<int64_t> input_ids;
                std::vector<int64_t> attention_mask;
                input_ids.reserve(ids_j.size());
                attention_mask.reserve(mask_j.size());
                for (size_t i = 0; i < ids_j.size(); ++i) {
                    input_ids.push_back(ids_j.at(i).get<int64_t>());
                    attention_mask.push_back(mask_j.at(i).get<int64_t>());
                }

                json resp = call_warm_service(cfg, input_ids, attention_mask);
                res.set_content(resp.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                json err{{"error", e.what()}};
                res.set_content(err.dump(), "application/json");
            }
        });

        std::cout << "Gateway listening on " << cfg.host << ":" << cfg.port << "\n";
        svr.listen(cfg.host, cfg.port);

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
