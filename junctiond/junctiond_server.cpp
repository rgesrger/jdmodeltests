#include "junctiond.grpc.pb.h"
#include "junctiond.h"

#include <grpcpp/grpcpp.h>
#include <memory>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

// This class adapts your C++ JunctionD *into a gRPC service*.
// Each RPC method simply calls the corresponding C++ method.
//
// Go → gRPC Request → JunctionServiceImpl → JunctionD methods.
class JunctionServiceImpl final : public junctiond::JunctionService::Service {
public:
    // We receive a pointer to your existing JunctionD instance.
    // This keeps all your spawn/remove/list logic unchanged.
    JunctionServiceImpl(JunctionD* jd) : jd_(jd) {}

    // gRPC wrapper for JunctionD::spawn()
    Status Spawn(ServerContext* ctx,
                 const junctiond::FunctionData* req,
                 junctiond::StatusReply* reply) override
    {
        // Convert protobuf → C++ struct
        FunctionData f;
        f.name     = req->name();
        f.execpath   = req->execpath();
        f.args = req ->args();
        f.cpu      = req->cpu();
        f.memoryMB = req->memorymb();

        // Call your real implementation
        bool ok = jd_->spawn(f);

        // Reply to client (Go)
        reply->set_success(ok);
        reply->set_message(ok ? "Spawned" : "Failed to spawn");

        return Status::OK;
    }

    // gRPC wrapper for JunctionD::remove()
    Status Remove(ServerContext* ctx,
                  const junctiond::FunctionName* req,
                  junctiond::StatusReply* reply) override
    {
        bool ok = jd_->remove(req->name());
        reply->set_success(ok);
        reply->set_message(ok ? "Removed" : "Failed to remove");
        return Status::OK;
    }

    // gRPC wrapper for JunctionD::list()
    Status List(ServerContext* ctx,
                const junctiond::Empty*,
                junctiond::FunctionList* reply) override
    {
        // Fetch running processes
        auto list = jd_->list();

        // Convert C++ → protobuf
        for (auto& st : list) {
            auto* f = reply->add_functions();
            f->set_name(st.name);
            f->set_running(st.running);
            f->set_pid(st.pid);
        }
        return Status::OK;
    }

private:
    JunctionD* jd_;   // Your real implementation lives here
};

// This starts the actual gRPC server.
// Think of this as "containerd.sock but for JunctionD".
void RunServer() {
    // gRPC listens over a UNIX socket.
    // This makes it similar to containerd's behavior.
    std::string server_address("unix:/run/junctiond.sock");

    // Create your actual process manager
    JunctionD jd;

    // Create the gRPC layer that wraps the C++ methods
    JunctionServiceImpl service(&jd);

    // Build the server
    ServerBuilder builder;

    // Set Unix socket for RPC
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Register all RPC methods
    builder.RegisterService(&service);

    // Start Server
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "[junctiond] gRPC server listening…" << std::endl;

    // Wait forever (until process killed)
    server->Wait();
}

// Main entry point of junctiond
int main() {
    RunServer();
    return 0;
}
