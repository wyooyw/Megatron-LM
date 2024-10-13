export PYTHONPATH=/cpfs/2926428ee2463e44/user/wangyiou/repos/Megatron-LM:$PYTHONPATH
export USE_WYO=1
export USE_WYO_SCHEDULER=1
bash examples/gpt3/train_gpt3_175b_distributed.sh


# export USE_WYO_SCHEDULER=1
# bash examples/gpt3/train_gpt3_175b_distributed.sh