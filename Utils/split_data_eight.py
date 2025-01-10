import os
import json

# This optional method splits the COCO data into 4/8 pieces and runs each piece on a GPU.


source_path = "Our_Results/Prompts/MS_COCO/8.24.30K.en_es_ru_it_zh_fr_de.json"
dir_path = "Our_Results/Prompts/MS_COCO/8.24.30K.en_es_ru_it_zh_fr_de.split8/"
os.makedirs(dir_path, exist_ok=True)

def write_data(data, path):
    writer = open(path, 'a')
    for i in data:
        writer.write(json.dumps(i, ensure_ascii=False) + "\n")

is_exit = os.path.exists(source_path)
print(is_exit)
with open(source_path, 'r') as file:
    data = []
    for i in file:
        data.append(json.loads(i))
    length = len(data)
    print(length)
    
file_count = 0
start_index = 0
tgt_list = ["part1.json", "part2.json", "part3.json", "part4.json", "part5.json", "part6.json", "part7.json", "part8.json"]
# tgt_list = ["part1.json", "part2.json", "part3.json", "part4.json"]
one_file_length = int(length / len(tgt_list))
print(one_file_length)

while file_count < len(tgt_list):
    file_data = data[start_index:start_index + one_file_length]
    write_data(file_data, dir_path + tgt_list[file_count])
    file_count += 1
    start_index += one_file_length
