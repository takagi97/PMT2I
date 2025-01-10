
NAME=db_prompts_copy4_promptist_format.json
export CUDA_VISIBLE_DEVICES=6,7
nohup python generate_pict_db_cb.py \
    --input_path Our_Results/Prompts/DrawBench/$NAME \
    --output_path /your/predict/path/ \
    --cuda0 6 \
    --cuda1 7 > /your/predict/path/emu.$NAME.log 2>&1 &
