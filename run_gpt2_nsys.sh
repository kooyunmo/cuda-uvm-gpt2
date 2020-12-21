CUDA_VISIBLE_DEVICES=1 nsys profile \
  --trace=cublas,cuda,cudnn,nvtx,osrt \
  -o ./nsys/gpt3_6.7b_mul_stream.output \
  -f true \
  python run_gpt2.py --model gpt3_6.7b --enable-prefetch --warmups 0
