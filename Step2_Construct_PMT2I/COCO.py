import os
import json

import json

dir_name = "Our_Results/Translations/MS_COCO/"
file_name_list = ["French_result.json",  "Italian_result.json",  "Russian_result.json",  "Chinese_result.json","Spanish_result.json","German_result.json"]


def get_all_langague(file_name)->list:
    result = []
    with open(file_name,'r') as file:
        for str_content in file:
            content = json.loads(str_content)
            result.append(content)
    return result


Englist_result = []
with open(dir_name + file_name_list[0],'r') as English_file:
    for str_content in English_file:
        content = json.loads(str_content)
        Englist_result.append(content.get("caption"))

Chinese_result = get_all_langague(dir_name + file_name_list[3])
French_result = get_all_langague(dir_name + file_name_list[0])
German_result = get_all_langague(dir_name + file_name_list[5])
Italian_result = get_all_langague(dir_name + file_name_list[1])
Russian_result = get_all_langague(dir_name + file_name_list[2])
Spanish_result = get_all_langague(dir_name + file_name_list[4])


writer = open(dir_name + 'Our_Results/Prompts/MS_COCO/8.24.30K.en_es_ru_it_zh_fr_de.json','w')
count = 0
for index,zip_content in enumerate(zip(Chinese_result,French_result,Italian_result,Russian_result,Spanish_result,German_result)):
    multi_prompt = ""
    other = zip_content[0]
    Chinese_prompt = f"Chinese: {zip_content[0].get('Chinese')}" + "\n"
    French_prompt = f"French: {zip_content[1].get('French')}" + "\n"
    German_prompt = f"German: {zip_content[5].get('German')}" + "\n"
    Italian_prompt = f"Italian: {zip_content[2].get('Italian')}" + "\n"
    Russian_prompt = f"Russian: {zip_content[3].get('Russian')}" + "\n"
    Spanish_prompt = f"Spanish: {zip_content[4].get('Spanish')}" + "\n"
    English_prompt = f"English: {Englist_result[index]}" + "\n"

    Chinese_id = zip_content[0].get('id')
    French_id = zip_content[1].get('id')
    German_id = zip_content[5].get('id')
    Italian_id = zip_content[2].get('id')
    Russian_id = zip_content[3].get('id')
    Spanish_id = zip_content[4].get('id')
    if Chinese_id == French_id == German_id == Italian_id == Russian_id == Spanish_id:
        del other["caption"]
        del other["Chinese"]
        # EN_Prompt
        # multi_prompt = English_prompt
        # PMI-6
        multi_prompt = multi_prompt + English_prompt + Spanish_prompt + Russian_prompt + Italian_prompt + Chinese_prompt + French_prompt + German_prompt
        other["multi_prompt"] = multi_prompt.strip()
        writer.write(json.dumps(other,ensure_ascii=False) + '\n')
    else:
        print("Fail!")
writer.close()