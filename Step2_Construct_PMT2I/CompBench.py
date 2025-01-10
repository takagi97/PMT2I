import os
import json


input_files = [
    "Our_Results/Translations/CompBench/color/Spanish_result.json",
    "Our_Results/Translations/CompBench/color/Russian_result.json",
    "Our_Results/Translations/CompBench/color/Italian_result.json",
    "Our_Results/Translations/CompBench/color/Chinese_result.json",
    "Our_Results/Translations/CompBench/color/French_result.json",
    "Our_Results/Translations/CompBench/color/German_result.json"
]

output_file = "Our_Results/Prompts/CompBench/en_es_ru_it_zh_fr_de/color_PMI-6_es_ru_it_zh_fr_de.json"

data_list = []
id_counter = 1

for filepath in input_files:
    with open(filepath, "r", encoding="utf-8") as f:
        json_data = f.readlines()
        language = os.path.basename(filepath).split("_")[0].capitalize()

        for line in json_data:
            item = json.loads(line)
            found = False

            for existing_item in data_list:
                if existing_item['caption'].startswith(f"English: {item['caption']}"):
                    existing_item['caption'] += f"\n{language}: {item[language]}"
                    found = True
                    break

            if not found:
                new_item = {
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
