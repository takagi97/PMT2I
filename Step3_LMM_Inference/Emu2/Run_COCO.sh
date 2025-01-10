export CUDA_VISIBLE_DEVICES=0,1
python generate_pict_COCO.py \
    --input_path Our_Results/Prompts/MS_COCO/8.24.30K.en_es_ru_it_zh_fr_de.split8/part1.json \
    --output_path /your/predict/path/ \
    --cuda0 0 \
    --cuda1 1 \