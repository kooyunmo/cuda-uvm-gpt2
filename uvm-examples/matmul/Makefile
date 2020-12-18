TARGET=main
OBJECTS=util.o mat_mul.o

CXXFLAGS=-O3 -Wall
LDFLAGS=-lm -L/home/kooyunmo/cuda-11.0/lib64 -lcudart

all: $(TARGET)

$(TARGET): $(OBJECTS)

mat_mul.o: mat_mul.cu
	nvcc -c -arch=sm_61 -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	./$(TARGET)

validation:
	./$(TARGET) -v 367 751 601
	./$(TARGET) -v 331 523 457
	./$(TARGET) -v 307 461 89
	./$(TARGET) -v 67 797 653
	./$(TARGET) -v 113 71 3
	./$(TARGET) -v 821 883 241
	./$(TARGET) -v 641 503 257
	./$(TARGET) -v 599 2 239
	./$(TARGET) -v 353 947 13
	./$(TARGET) -v 347 593 229
	./$(TARGET) -v 433 409 443
	./$(TARGET) -v 223 479 491
	./$(TARGET) -v 839 277 677
	./$(TARGET) -v 727 379 53
	./$(TARGET) -v 521 557 541
	./$(TARGET) -v 11 349 463
	./$(TARGET) -v 487 17 293
	./$(TARGET) -v 397 199 337
	./$(TARGET) -v 787 853 619
	./$(TARGET) -v 157 773 739

performance: $(TARGET)
	./$(TARGET) -n 10 8192 8192 8192
