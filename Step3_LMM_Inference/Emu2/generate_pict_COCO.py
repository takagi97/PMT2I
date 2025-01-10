# -*- coding: utf-8 -*-
import argparse
import json
from generate_fun import generate,load_resources,PATH

def arges():
    parser = argparse.ArgumentParser(description='Emu2 inference')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--cuda0', type=str)
    parser.add_argument('--cuda1', type=str)
    return parser.parse_args()

def main(source_path,tgt_path,cuda0,cuda1):
    cuda_ = f"{cuda0},{cuda1}"
    pipe = load_resources(PATH,cuda_)
    with open(source_path, 'r') as file:
        for str_content in file:
            content = json.loads(str_content)
            image_id = str(content.get("image_id"))
            suffix = "COCO_val2014_" + "0" * (12 - len(image_id)) + image_id
            prefix = str(content.get("id"))
            file_name = prefix + "-" + suffix + ".jpg"
            final_out_name = tgt_path + file_name
            prompt = content.get("caption")
            # print(path)
            # print(f"prompt:{prompt}")
            print(f"final_out_name:{final_out_name}")
            # print(pipe)
            generate(prompt, final_out_name, pipe)

if __name__ == "__main__":
    arg = arges()
    input_path = arg.input_path
    output_path = arg.output_path
    cuda0 = arg.cuda0
    cuda1 = arg.cuda1
    import os
    os.makedirs(output_path, exist_ok=True)
    main(input_path,output_path,cuda0=cuda0,cuda1=cuda1)

    
