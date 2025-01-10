import json
import sys

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            original_id = data.get("id")
            if original_id is not None:
                for i in range(1, 5):
                    new_data = data.copy()
                    new_data["id"] = int(f"{original_id}0{i}")
                    outfile.write(json.dumps(new_data) + "\n")

if __name__ == "__main__":
    name = "en_de_fr_es_zh_it_ru"
    input_file = f"Our_Results/Prompts/DrawBench/{name}.json"
    output_file = f"Our_Results/Prompts/DrawBench/{name}_copy4.json"
    print(output_file)
    process_jsonl(input_file, output_file)
