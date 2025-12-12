CXX = g++
CXXFLAGS = -std=c++17 -pthread -Wall -O2

# 1. Common Files (The Logic)
# junctiond.cpp is included here as it contains the logic needed by test.cpp
COMMON_SRCS = junctiond.cpp
COMMON_OBJS = $(COMMON_SRCS:.cpp=.o)

# 2. Target: Test (test.cpp)
TEST_TARGET = test
TEST_SRCS = test.cpp
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

# 3. Target: Test Inference (test_infer.cpp) — runs distilbert_infer binary
TEST_INFER_TARGET = test_infer
TEST_INFER_SRCS = test_infer.cpp
TEST_INFER_OBJS = $(TEST_INFER_SRCS:.cpp=.o)

# We remove the SERVER_TARGET definitions.

# Default: Build both test executables
all: $(TEST_TARGET) $(TEST_INFER_TARGET)

# --- Build Rules ---

# Build the Test Binary
$(TEST_TARGET): $(TEST_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_OBJS) $(COMMON_OBJS)

$(TEST_INFER_TARGET): $(TEST_INFER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_INFER_OBJS)

# We remove the rule for the server target.

# --- Helper Commands ---

# Convenience commands to build AND run
run: $(TEST_TARGET)
	@echo ">>> Starting Test..."
	./$(TEST_TARGET)

run_infer: $(TEST_INFER_TARGET)
	@echo ">>> Running distilbert_infer wrapper..."
	./$(TEST_INFER_TARGET)

# Compile .cpp files to .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean only the files relevant to the tests
clean:
	rm -f *.o $(TEST_TARGET) $(TEST_INFER_TARGET)
# We remove the cleanup for the 'server' target.