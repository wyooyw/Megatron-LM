#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_LAUNCH_BLOCKING=1
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6124
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=$1 #<Specify path>
# TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=/cpfs/2926428ee2463e44/user/wangyiou/repos/Megatron-LM/vocab_files/gpt2-vocab.json
MERGE_FILE=/cpfs/2926428ee2463e44/user/wangyiou/repos/Megatron-LM/vocab_files/gpt2-merges.txt
# DATA_PATH=$5 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# MICRO_BS=2
# GLOBAL_BS=16
# GPT_MODEL_ARGS=(
#     --num-layers 4
#     --hidden-size 16384
#     --num-attention-heads 256
#     --seq-length 4096 
#     --max-position-embeddings 4096 
# )

MICRO_BS=8
GLOBAL_BS=64
GPT_MODEL_ARGS=(
    --num-layers 8
    --hidden-size 8192
    --num-attention-heads 128
    --seq-length 8192
    --max-position-embeddings 8192
)

# MICRO_BS=32
# GLOBAL_BS=64
# GPT_MODEL_ARGS=(
#     --num-layers 4
#     --hidden-size 4096
#     --num-attention-heads 32
#     --seq-length 1024
#     --max-position-embeddings 1024
# )

TRAINING_ARGS=(
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --no-bias-gelu-fusion
    --no-bias-swiglu-fusion
    --no-bias-dropout-fusion
    
    --micro-batch-size $MICRO_BS
    --global-batch-size $GLOBAL_BS
    # --rampup-batch-size 16 16 5859375 
    --train-iters 32
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --hidden-dropout 0.0
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $GPUS_PER_NODE
    --sequence-parallel
    # --tp-comm-overlap
)

DATA_ARGS=(
    # --init-method-xavier-uniform
    --mock-data
    # --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    # --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-throughput
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 0
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# PROFILE=(
#     --profile
#     --profile-step-start 5
#     --profile-step-end 10
# )


nsys profile -w true -t cuda,nvtx -s cpu  \
--capture-range=cudaProfilerApi \
--cudabacktrace=true \
-x true \
-o nsys/sp${GPUS_PER_NODE}_wyo_overlap_seqlen8k_mbs8 \
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PROFILE[@]}