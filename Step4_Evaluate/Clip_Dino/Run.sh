export CUDA_VISIBLE_DEVICES=1
# l1,l2,clip-i,dino,clip-t

# MS COCO
python MB_eval.py \
--metric clip-i,dino,clip-t \
--generated_path /your/predict/path \
--caption_path Utils/coco_30k_img_id2caption.json

# DrawBench
python MB_eval.py \
--metric clip-t \
--generated_path /your/predict/path \
--caption_path Utils/db_prompts_copy4_image_name2caption.json

# CompBench
python MB_eval.py \
--metric clip-t \
--generated_path /your/predict/path \
--caption_path Utils/cb_id2caption/color_output.json