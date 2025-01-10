module load python/3.8
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2  refl_sdxl.py \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=100 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="checkpoint/refl_sdxl" \
  --grad_scale=0.001 \
  --checkpointing_steps 100 \
  --image_base_dir="data/images/" \
  --mapping_batch_size=128 \
  --save_only_one_ckpt \
  --apply_pre_loss \
  --apply_reward_loss
