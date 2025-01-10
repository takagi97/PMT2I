
NAME=db_prompts_copy4_promptist_format.json
export CUDA_VISIBLE_DEVICES=0
nohup python Step3_LMM_Inference/Lumina/lumina/lumina_next_t2i/utils/cli_db.py \
    --num_gpus 1 \
    --ckpt /path/to/Lumina-Next-T2I \
    --ckpt_lm /path/to/gemma-2b \
    --precision bf16 \
    --config_path Step3_LMM_Inference/Lumina/lumina/lumina_next_t2i/configs/infer/settings.yaml \
    --token "" \
    --json_path Our_Results/Prompts/DrawBench/$NAME \
    --output_path /your/predict/path/lumina.$NAME > lumina.$NAME.log 2>&1 &
