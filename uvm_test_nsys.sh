nsys profile \
  --trace=cublas,cuda,cudnn,nvtx,osrt \
  -o ./nsys/test_small.output \
  -f true \
  python uvm_test.py
