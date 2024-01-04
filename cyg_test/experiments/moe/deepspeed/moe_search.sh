# 使用 ./moe_search num_hosts num_devices_per_host
cd ../../../..
# 开启gcc，并设置deepspeed目录
source /opt/rh/devtoolset-10/enable
export DEEPSPEED_PATH=/home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/benchmark/deepspeed/DeepSpeed

# 定义测试变量
nnodes=$1
nproc_per_node=$2
niter=10
repeat_num=2
suite='search'

num_gpus=`expr $nnodes \* $nproc_per_node`
# 定义输出文件和配置文件路径
OUTPUT_DATA_FILE_PATH="cyg_test/experiments/results/moe_deepspeed_search_${num_gpus}.csv"
MOE_DEEPSPEED_LOG_FILE_PATH="cyg_test/experiments/moe/deepspeed/log/moe_search_log_${num_gpus}.txt"
DEEPSPEED_CONFIG_FILE_PATH='cyg_test/experiments/moe/deepspeed/ds_zero_stage_2_moe_config.json'
INPUT_DATA_PATH='cyg_test/experiments/moe/deepspeed/data/small-webtext/small-webtext'
VOCAB_FILE_PATH='cyg_test/experiments/moe/deepspeed/data/gpt2-vocab.json'
MERGE_FILE_PATH='cyg_test/experiments/moe/deepspeed/data/gpt2-merges.txt'

# 清空日志文件和数据文件
echo -n > $MOE_DEEPSPEED_LOG_FILE_PATH
echo -n > $OUTPUT_DATA_FILE_PATH

python cyg_test/experiments/moe/deepspeed/write_csv_heads.py --output_name $OUTPUT_DATA_FILE_PATH
for enable_overlapping_flag in "true";
do
    python cyg_test/experiments/moe/deepspeed/moe_test.py --nnodes $nnodes --nproc_per_node $nproc_per_node --suite $suite --niter $niter --repeat_num $repeat_num --enable_overlapping $enable_overlapping_flag --output_name $OUTPUT_DATA_FILE_PATH --config_file $DEEPSPEED_CONFIG_FILE_PATH --data_path $INPUT_DATA_PATH --vocab_file $VOCAB_FILE_PATH --merge_file $MERGE_FILE_PATH  >> $MOE_DEEPSPEED_LOG_FILE_PATH 2>&1 2>&1
    sleep 5
done