# Only CompBench
# The evaluation results will be saved in out_dir.
python Step4_Evaluate/BlipScore/BLIPvqa_eval/Blipvqa_eval.py \
  --image_folder /your/generatedImage/path \
  --json_path Our_Results/Prompts/CompBench/baseline/color_val_id.json \
  --out_dir /your/output/path \
  --np_num 8