
NAME=8.24.30K.en_es_ru_it_zh_fr_de.json
PART=part1
export CUDA_VISIBLE_DEVICES=0
nohup python Step3_LMM_Inference/Lumina/lumina/lumina_next_t2i/utils/cli.py \
    --num_gpus 1 \
    --ckpt /path/to/Lumina-Next-T2I \
    --ckpt_lm /path/to/gemma-2b \
    --precision bf16 \
    --config_path Step3_LMM_Inference/Lumina/lumina/lumina_next_t2i/configs/infer/settings.yaml \
    --token "" \
    --json_path Our_Results/Prompts/MS_COCO/$NAME/$PART.json \
    --output_path /your/predict/path/lumina.$NAME/$PART > lumina.$NAME.$PART.log 2>&1 &

PART=part2
export CUDA_VISIBLE_DEVICES=1
nohup python Step3_LMM_Inference/Lumina/lumina/lumina_next_t2i/utils/cli.py \
    --num_gpus 1 \
    --ckpt /path/to/Lumina-Next-T2I \
    --ckpt_lm /path/to/gemma-2b \
    --precision bf16 \
    --config_path Step3_LMM_Inference/Lumina/lumina/lumina_next_t2i/configs/infer/settings.yaml \
    --token "" \
    --json_path Our_Results/Prompts/MS_COCO/$NAME/$PART.json \
    --output_path /your/predict/path/lumina.$NAME/$PART > lumina.$NAME.$PART.log 2>&1 &

wait