# 使用 ./gpt_search num_hosts num_devices_per_host
cd ../../../..
# 开启gcc，并设置Megatron目录
source /opt/rh/devtoolset-10/enable
export PYTHONPATH=$PYTHONPATH:/home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/benchmark/megatron/Megatron-LM

# 定义测试变量
nnodes=$1
nproc_per_node=$2
niter=10
repeat_num=2
# best or search
suite='search'

num_gpus=`expr $nnodes \* $nproc_per_node`
# 定义输出文件和配置文件路径
OUTPUT_DATA_FILE_PATH="cyg_test/experiments/results/gpt_megatron-lm_search_${num_gpus}.csv"
GPT_MEGATRON_LOG_FILE_PATH="cyg_test/experiments/gpt/megatron/log/gpt_search_log_${num_gpus}.txt"

# 清空日志文件和数据文件
echo -n > $GPT_MEGATRON_LOG_FILE_PATH
echo -n > $OUTPUT_DATA_FILE_PATH

python cyg_test/experiments/gpt/megatron/write_csv_heads.py --output_name $OUTPUT_DATA_FILE_PATH

python cyg_test/experiments/gpt/megatron/gpt_test.py --nnodes $nnodes --nproc_per_node $nproc_per_node --suite $suite --niter $niter --repeat_num $repeat_num --output_name $OUTPUT_DATA_FILE_PATH  >> $GPT_MEGATRON_LOG_FILE_PATH 2>&1 2>&1
