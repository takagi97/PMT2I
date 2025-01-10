export CUDA_VISIBLE_DEVICES='2'

python Step3_LMM_Inference/Lumina/lumina/lumina_next_t2i/utils/cli_db.py \
    --num_gpus 1 \
    --ckpt /path/to/Lumina-Next-T2I \
    --ckpt_lm /path/to/gemma-2b \
    --precision bf16 \
    --config_path Step3_LMM_Inference/Lumina/lumina/lumina_next_t2i/configs/infer/settings.yaml \
    --token "" \
    --json_path Our_Results/Prompts/CompBench/promptist_format/texture_val_id_promptist_format.json \
    --output_path /your/predict/path