#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_LAUNCH_BLOCKING=1
GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6123
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

GPT_MODEL_ARGS=(
    --num-layers 8
    --hidden-size 4096
    --num-attention-heads 32
    --seq-length 1024 
    --max-position-embeddings 1024 
)

TRAINING_ARGS=(
    # --transformer-impl transformer_engine
    --micro-batch-size 4
    --global-batch-size 4
    # --rampup-batch-size 16 16 5859375 
    --train-iters 32
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    # --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2 
    --sequence-parallel
    --tp-comm-overlap
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
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 0
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

PROFILE=(
    --profile
    --profile-step-start 10
    --profile-step-end 20
)

nsys profile -w true -t cuda,nvtx -s cpu  \
--capture-range=cudaProfilerApi \
--cudabacktrace=true \
-x true \
-o nsys/tp2_rs_ag_overlap \
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PROFILE[@]}
