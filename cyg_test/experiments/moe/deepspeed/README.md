# Benchmark Deepspeed

## Requirements
1. Install dependencies
```
# torch
pip3 install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install nltk pandas sentencepiece boto3 pybind11 python-config

# torch 1.10.0 and CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install nltk pandas sentencepiece boto3 pybind11 python-config


# Adafactor optimizer
pip3 install torch-optimizer

# pdsh
sudo apt-get update
sudo apt-get install pdsh

# Apex
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
# Comment out the raised RuntimeError in setup.py if you get errors running the following command.
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

2. Install deepspeed and deepspeed examples
```
pip3 install deepspeed==0.5.6
git clone --recursive https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
git checkout v0.5.6
<!-- 如果没拉全 -->
git submodule update

<!-- echo 'export DEEPSPEED_PATH=~/efs/DeepSpeed' >> ~/.bashrc   # use your own path
source ~/.bashrc -->

<!-- 使用时设置 -->
export DEEPSPEED_PATH=/home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/benchmark/deepspeed/DeepSpeed


# Replace source files (use your own path)
cp alpa/benchmark/deepspeed/patch/training.py DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/megatron/training.py
cp alpa/benchmark/deepspeed/patch/gpt2_model.py DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/megatron/model/gpt2_model.py
cp alpa/benchmark/deepspeed/patch/transformer.py DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/megatron/model/transformer.py
```

<!-- 还没做完 -->
3. Download dataset
``` 已经没用
wget deepspeed_dataset.zip  # ask Lianmin to get the file
tar xzf deepspeed_dataset.zip
cd deepspeed_dataset/
ln -s $(pwd) ~/efs/alpa/benchmark/deepspeed/data   # use your own path
```
```
wget https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/arxiv_024de5df-1b7f-447c-8c3a-51407d8d6732.jsonl

mv arxiv_* arxiv.jsonl

cd /home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/benchmark/deepspeed/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3

python tools/preprocess_data.py \
--input /home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/benchmark/deepspeed/data/small-webtext/arxiv.jsonl \
--output-prefix /home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/benchmark/deepspeed/data/small-webtext \
--dataset-impl mmap \
--tokenizer-type GPT2BPETokenizer \
--vocab-file /home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/benchmark/deepspeed/data/gpt2-vocab.json \
--merge-file /home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/benchmark/deepspeed/data/gpt2-merges.txt \
--workers 64 \
--append-eod
```

## Run
### Single Node
```
# GPT
python3 benchmark_gpt2.py --nproc_per_node 8
# MOE
python3 benchmark_moe.py --nproc_per_node 8
```

PYTHONPATH=/home/pc/aibot/cuiyonggan/projects/Pipeline_Experiments/alpa/benchmark/deepspeed/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/ PYTHON_VOCAB_SIZE=32000 deepspeed --num_nodes 1 --master_port 35721 --num_gpus 2 cyg_test/experiments/moe/deepspeed/pretrain_gpt2_moe.py --model-parallel-size 1 --num-layers 8 --hidden-size 768 --num-attention-heads 16 --seq-length 32 --max-position-embeddings 32 --batch-size 32 --train-iters 80 --lr-decay-iters 320000 --data-path cyg_test/experiments/moe/deepspeed/data/small-webtext/small-webtext --vocab-file cyg_test/experiments/moe/deepspeed/data/gpt2-vocab.json --merge-file cyg_test/experiments/moe/deepspeed/data/gpt2-merges.txt --data-impl mmap --split 949,50,1 --distributed-backend nccl --lr 1.5e-4 --lr-decay-style cosine --min-lr 1.0e-5 --weight-decay 1e-2 --clip-grad 1.0 --warmup 0.01 --log-interval 1 --save-interval 10000 --eval-interval 2000 --eval-iters 0 --fp16 --loss-scale 1.0 --scattered-embeddings --split-transformers --deepspeed --deepspeed_config cyg_test/experiments/moe/deepspeed/ds_zero_stage_2_moe_config.json --moe --ep-world-size 2 --num-experts 8 --top-k 2 --min-capacity 4 --noisy-gate-policy None --moe-param-group --output_name cyg_test/experiments/results/gpt_megatron-lm.csv

### Multiple Node
- Modify the [hostfile](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) and setup the ssh connections.
```
python3 benchmark_gpt2.py --nnodes 2 --nproc_per_node 8
```
