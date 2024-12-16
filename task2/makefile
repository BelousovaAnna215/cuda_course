TARGET = adi
SRC = adi.cu
ARGS =

.PHONY: all clean build run run_c run_g run_cg

$(TARGET):
	nvcc -std=c++11 --extended-lambda $(SRC) -O3 -o $(TARGET)

all: clean build

build: $(TARGET)

clean:
	rm -rf $(TARGET)
	
run_c: build
	bsub -gpu num=1:mode=exclusive_process -x "./$(TARGET) c $(ARGS) >output 2>error"

run_g: build
	bsub -gpu num=1:mode=exclusive_process -x "./$(TARGET) g $(ARGS) >output 2>error"

run_cg: build
	bsub -gpu num=1:mode=exclusive_process -x "./$(TARGET) cg $(ARGS) >output 2>error"

