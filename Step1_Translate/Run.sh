input_path_MS_COCO=../Datasets/MS_COCO/coco_30k.json
MS_COCO_French=/you/want/to/save
input_path_DrawBench=../Datasets/DrawBench/DB_val_data.json
DrawBench_French=/you/want/to/save
input_CompBench=../Datasets/CompBench/color_val.txt
CompBench_French=/you/want/to/save

# MS COCO
mkdir -p $MS_COCO_French
proxychains4 python translate_by_chatgpt_coco.py \
    --file_need_translate_path $input_path_MS_COCO \
    --result_data_path $MS_COCO_French \
    --engine gpt-4o-2024-05-13 \
    --prompt_version 9 \
    --lang_para_flores English_#_eng_Latn_###_French_#_fra_Latn \
    --language French

# DrawBench
mkdir -p $DB_French
proxychains4 python translate_by_chatgpt_db_cb.py \
    --file_need_translate_path $input_path_DrawBench \
    --result_data_path $DrawBench_French \
    --engine gpt-4o-2024-05-13 \
    --prompt_version 33 \
    --lang_para_flores English_#_eng_Latn_###_French_#_fra_Latn \
    --language French

# CompBench
proxychains4 python translate_by_chatgpt_db_cb.py \
    --file_need_translate_path $input_CompBench \
    --result_data_path $CompBench_French \
    --engine gpt-4o-2024-05-13 \
    --prompt_version 9 \
    --lang_para_flores English_#_eng_Latn_###_French_#_fra_Latn \
    --language French