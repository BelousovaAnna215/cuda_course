TARGET = jacobi
SRC = jacobi.cu

.PHONY: all clean build run run_c run_g run_cg

$(TARGET):
	nvcc -std=c++11 --extended-lambda $(SRC) -O3 -o $(TARGET)

all: clean build

build: $(TARGET)

clean:
	rm -rf $(TARGET)

run_c: build
	bsub -gpu num=1:mode=exclusive_process -x "./$(TARGET) c >output 2>error"

run_g: build
	bsub -gpu num=1:mode=exclusive_process -x "./$(TARGET) g >output 2>error"

run_cg: build
	bsub -gpu num=1:mode=exclusive_process -x "./$(TARGET) cg >output 2>error"


run: 
	@echo "Usage: make run_c, make run_g, or make run_cg"
