import json

def convert_jsonl_to_json(input_file, output_file):
    result_dict = {}

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            
            id_value = data.get("id")
            caption_value = data.get("caption")
            
            if id_value is not None and caption_value is not None:
                key = f"{id_value}.jpg"
                result_dict[key] = caption_value

    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(result_dict, out_file, ensure_ascii=False, indent=4)

input_file = 'Our_Results/Prompts/DrawBench/db_prompts_copy4.json'
output_file = 'Utils/db_prompts_copy4_image_name2caption.json'

convert_jsonl_to_json(input_file, output_file)
