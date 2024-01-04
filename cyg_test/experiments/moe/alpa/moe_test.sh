# 使用 ./moe_test num_hosts num_devices_per_host
cd ../../../..

# 定义测试变量
num_hosts=$1
num_devices_per_host=$2
niter=10
repeat_num=2
# best or search
suite='best'

num_gpus=`expr $num_hosts \* $num_devices_per_host`
# 定义输出文件和配置文件路径
OUTPUT_DATA_FILE_PATH="cyg_test/experiments/results/moe_alpa_${num_gpus}.csv"
MOE_ALPA_LOG_FILE_PATH="cyg_test/experiments/moe/alpa/log/moe_test_log_${num_gpus}.txt"

# 清空日志文件和数据文件
echo -n > $MOE_ALPA_LOG_FILE_PATH
echo -n > $OUTPUT_DATA_FILE_PATH

python cyg_test/experiments/fix_patch.py

python cyg_test/experiments/moe/alpa/write_csv_heads.py --output_name $OUTPUT_DATA_FILE_PATH

ray stop --force

for method_name in "alpa" "bpp" "eager" ;  
do  
    for enable_overlapping_flag in "False" "True" ;
    do 
        ray start --head
        export enable_overlapping=$enable_overlapping_flag
        python cyg_test/experiments/moe/alpa/moe_test.py --num_hosts $num_hosts --num_devices_per_host $num_devices_per_host --suite $suite --niter $niter --repeat_num $repeat_num --method_name $method_name --output_name $OUTPUT_DATA_FILE_PATH >> $MOE_ALPA_LOG_FILE_PATH 2>&1 2>&1
        ray stop --force
        sleep 5
    done
done 