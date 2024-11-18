TARGET = jacobi
SRC = jacobi.cu

.PHONY: all clean build run run_c run_g run_cg

$(TARGET):
	nvcc $(SRC) -O3 -o $(TARGET)

all: clean build

build: $(TARGET)

clean:
	rm -rf $(TARGET)

run_c: build
	./$(TARGET) c

run_g: build
	./$(TARGET) g

run_cg: build
	./$(TARGET) cg

run: 
	@echo "Usage: make run_c, make run_g, or make run_cg"
