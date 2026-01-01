#ifndef JUNCTIOND_H
#define JUNCTIOND_H

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <thread>

struct FunctionData {
    std::string name;
    std::string execpath;
    std::string args;
    int cpu;
    int memoryMB;
    std::map<std::string, std::string> env;
};

struct FunctionStatus {
    std::string name;
    bool running;
    pid_t pid;
    
    // Add these two:
    int fd_write; 
    int fd_read;  
};
// Represents a job currently running in the background
struct Job {
    std::string name;
    pid_t pid;
    int fd_write; // stdin of the child
    int fd_read;  // stdout of the child
    std::chrono::steady_clock::time_point startTime;
    bool startupCaptured = false;
    double startupTime = 0.0;
};

struct JobResult {
    std::string name;
    std::string output;
    double startupSeconds; // Time from fork to "READY"
    double totalSeconds;   // Time from fork to exit
};

class JunctionD {
public:
    JunctionD();
    ~JunctionD();

    bool spawn(const FunctionData &func);
    bool remove(const std::string &name);
    JobResult collect(std::string name);
    std::vector<FunctionStatus> list();

private:
    void monitorInstances();
    
    bool generateConfig(const FunctionData &func, std::string &cfgPath); // declare here

    std::map<std::string, FunctionStatus> statusMap;
    std::vector<Job> activeJobs;
    std::mutex mtx;
    std::thread monitorThread;
};


#endif // JUNCTIOND_H
