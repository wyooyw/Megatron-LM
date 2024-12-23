nsys profile -w true -t cuda,nvtx -s cpu  \
--capture-range=cudaProfilerApi \
--cudabacktrace=true \
-x true \
-o nsys/test_memory \
python test_memory.py