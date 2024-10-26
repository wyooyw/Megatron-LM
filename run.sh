export PYTHONPATH=/cpfs/2926428ee2463e44/user/wangyiou/repos/Megatron-LM:$PYTHONPATH

# bash examples/gpt3/train_gpt3_175b_distributed_pp.sh

# export USE_WYO=1
# export USE_WYO_SCHEDULER=1
# export NCCL_DEBUG=INFO
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# bash examples/gpt3/train_gpt3_175b_distributed.sh

# export USE_WYO=0
# export USE_WYO_SCHEDULER=0
# export EXP_NAME=megatron
# bash examples/gpt3/train_gpt3_175b_distributed_tp.sh

# export USE_WYO=1
# export USE_WYO_SCHEDULER=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export EXP_NAME=wyo_overlap_split512
# bash examples/gpt3/train_gpt3_175b_distributed.sh

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# export EXP_NAME=wyo_overlap_split256
# bash examples/gpt3/train_gpt3_175b_distributed.sh

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export EXP_NAME=wyo_overlap_split128
# bash examples/gpt3/train_gpt3_175b_distributed.sh

# export USE_WYO=1
# export USE_WYO_SCHEDULER=1
# export EXP_NAME=tp_megatron
# bash examples/gpt3/train_gpt3_175b_distributed_tp.sh

# export USE_WYO=1
# export EXP_NAME=tp_wyo
# bash examples/gpt3/train_gpt3_175b_distributed_tp.sh
# export CUDA_LAUNCH_BLOCKING=1
# export USE_WYO=1
# export USE_WYO_SCHEDULER=1
# export EXP_NAME=wyo_tp
# export EXP_NAME=wyo_sp_smart_dmr_rsag
# export EXP_NAME=wyo_smart_dmr_2sort_fa_ln_phfirst
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# bash examples/gpt3/train_gpt3_175b_distributed_tp.sh

# export EXP_NAME=wyo_overlap_smart_maxchannel4
# export NCCL_MAX_NCHANNELS=4
# bash examples/gpt3/train_gpt3_175b_distributed_tp.sh

# export USE_WYO_SCHEDULER=1
# bash examples/gpt3/train_gpt3_175b_distributed.sh


# bash examples/gpt3/train_llama_13b_tp.sh
bash examples/gpt3/train_gpt3_13b_tp.sh
