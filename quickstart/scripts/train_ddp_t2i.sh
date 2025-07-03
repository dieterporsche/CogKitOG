#!/usr/bin/env bash
# Run by `bash scripts/train_ddp_i2v.sh`

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogView4-6B"
    --model_name "cogview4-6b"  # candidate: ["cogview4-6b"]
    --model_type "t2i"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/path/to/output"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/path/to/data"
)

# Training Configuration
TRAIN_ARGS=(
    --seed 42  # random seed
    --train_epochs 1  # number of training epochs
    --batch_size 1

    --gradient_accumulation_steps 1

    # Note: For CogView4 series models, height and width should be **32N** (multiple of 32)
    --train_resolution "1024x1024"  # (height x width)

    # When enable_packing is true, training will use the native image resolution
    # (otherwise all images will be resized to train_resolution, which may distort the original aspect ratio).
    #
    # IMPORTANT: When changing enable_packing from true to false (or vice versa),
    # make sure to clear the .cache directories in your data_root/train and data_root/test folders if they exist.
    --enable_packing false

    --mixed_precision "bf16"  # ["no", "fp16"]
    --learning_rate 5e-5

    # enable --low_vram will slow down validation speed and enable quantization during training
    # Note: --low_vram currently does not support multi-GPU training
    --low_vram false
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory true
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 10  # save checkpoint every x steps
    --checkpointing_limit 2   # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true   # ["true", "false"]
    --validation_steps 10  # should be multiple of checkpointing_steps
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
