
## Getting Started
### Installation
```bash
# Recommend Python 3.10.
conda create -n drpo python=3.10
conda activate drpo
pip install -e ./verl
pip install -e ./deepscaler
pip install wandb
```


### Training

We provide training scripts for both single-node and multi-node setups in `scripts/train/`.

#### Single-Node Training (8 GPUs)
We start with one node for training 1.5B Qwen models with 8k context, with 8 A100-80GB GPUs.
```bash
bash ./scripts/train/run_drpo_1.5b_8k.sh   #### DRPO 
```

#### Multi-Node Training

To train with longer context or larger models, multi-node training is necessary. To achieve this, follow these steps:

1. On the head node:
```bash
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Start Ray head node
ray start --head
```

2. On each worker node:
```bash
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Connect to head node (replace with your head node's address)
ray start --address=[RAY_ADDRESS]
```

3. Finally, on the head node, run the training script, such as:
```bash
bash ./scripts/train/run_drpo_1.5b_8k.sh
```


## Evaluation

Our evaluation scripts automatically runs vLLM to generate 16 samples for each problem. To run our evaluation scripts, run:
```bash
./scripts/eval/eval_model.sh --model [CHECKPOINT_PATH] --datasets [DATASET1] [DATASET2] --output-dir [OUTPUT_DIR]
```



