#!/usr/bin/env bash
# Run by `bash scripts/train_ddp_i2v.sh`

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "/mnt/models/CogVideoX1.5-5B-I2V"
    --model_name "cogvideox1.5-i2v"  
    --model_type "i2v"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/HuggingFace/CogKit/quickstart/outputNEW"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/home/azureuser/cloudfiles/code/Users/dieter.holstein/runs/DataPreparation/CogVideo/Split/Train"
)

# Training Configuration PROVIDED DEFAULT
TRAIN_ARGS=(
    --seed 42  # random seed
    --train_epochs 1  # number of training epochs
    --batch_size 4
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"]
    --learning_rate 5e-5
    # Note:
    #  for CogVideoX series models, number of training frames should be **8N+1**
    #  for CogVideoX1.5 series models, number of training frames should be **16N+1**
    #--train_resolution "81x768x1360"  # (frames x height x width)
    --train_resolution "17x768x768"  # (frames x height x width)  "Durch 16 Teilbar + 1" x "durch 8 teilbar" x "durch 8 Teilbar"
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 32
    --pin_memory true
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpoint_each_epoch true  # save checkpoint after each epoch
    --checkpointing_limit 2   # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false   # ["true", "false"]
    --validation_steps 200  # ignored when checkpoint_each_epoch is true
    --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
