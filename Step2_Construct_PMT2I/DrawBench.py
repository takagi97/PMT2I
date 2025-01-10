import os
import json

input_files = [
    "Our_Results/Translations/DrawBench/Spanish_result.json",
    "Our_Results/Translations/DrawBench/Russian_result.json",
    "Our_Results/Translations/DrawBench/Italian_result.json",
    "Our_Results/Translations/DrawBench/Chinese_result.json",
    "Our_Results/Translations/DrawBench/French_result.json",
    "Our_Results/Translations/DrawBench/German_result.json"
]

output_file = "Our_Results/Prompts/DrawBench/en_de_fr_es_zh_it_ru.json"

data_list = []
id_counter = 1

for filepath in input_files:
    with open(filepath, "r", encoding="utf-8") as f:
        json_data = f.readlines()

        for line in json_data:
            item = json.loads(line)
            language = filepath.split("/")[-1].split("_")[0].capitalize()

            found = False
            for existing_item in data_list:
                if existing_item['Category'] == item['Category'] and existing_item['caption'].startswith(f"English: {item['caption']}"):
                    existing_item['caption'] += f"\n{language}: {item[language]}"
                    found = True
                    break

            if not found:
                new_item = {
                    "Category": item['Category'],
                    "caption": f"English: {item['caption']}\n{language}: {item[language]}",
                    "id": id_counter
                }
                data_list.append(new_item)
                id_counter += 1

with open(output_file, "w", encoding="utf-8") as f:
    for entry in data_list:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

print(f"Output:{output_file}")