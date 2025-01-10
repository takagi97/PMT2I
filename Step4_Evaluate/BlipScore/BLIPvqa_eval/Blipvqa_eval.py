# - *- coding: utf-8 -*-
import argparse
import os

import torch

from tqdm import tqdm, trange

import json
from tqdm.auto import tqdm
import sys
import spacy

from BLIP.train_vqa_func import VQA_main


import json
import os
import spacy

def Create_annotation_for_BLIP(json_path, image_folder, outpath, np_index=None):

    nlp = spacy.load("en_core_web_sm")


    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    annotations = []
    cnt = 0


    for item in data:
        image_dict = {}
        image_id = item['id']
        caption = item['caption']


        image_suffix = f".{image_id}.jpg"
        image_name = None
        for image_name in os.listdir(image_folder):
            if image_suffix in image_name:
                image_name = image_name
                break
        image_path = os.path.join(image_folder, image_name)


        if os.path.exists(image_path):
            image_dict['image'] = image_path
            image_dict['question_id'] = cnt
            cnt += 1


            doc = nlp(caption)
            noun_phrases = []


            for chunk in doc.noun_chunks:
                if chunk.text not in ['top', 'the side', 'the left', 'the right']:
                    noun_phrases.append(chunk.text)


            if np_index is not None and len(noun_phrases) > np_index:
                q_tmp = noun_phrases[np_index]
                image_dict['question'] = f'{q_tmp}?'
            else:
                image_dict['question'] = ''

            image_dict['dataset'] = "color"
            annotations.append(image_dict)
        else:
            print(f"Image {image_name} not found in {image_folder}.")


    print('Number of Processed Images:', len(annotations))


    json_file = json.dumps(annotations, indent=4)
    with open(os.path.join(outpath, 'vqa_test.json'), 'w') as f:
        f.write(json_file)




def parse_args():
    parser = argparse.ArgumentParser(description="BLIP vqa evaluation.")
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        required=True,
        help="Path to image folder",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        required=True,
        help="Path to json file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        required=True,
        help="Path to output BLIP vqa score",
    )
    parser.add_argument(
        "--np_num",
        type=int,
        default=8,
        help="Noun phrase number, can be greater or equal to the actual noun phrase number",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    np_index = args.np_num  # how many noun phrases
    image_folder = args.image_folder
    json_path = args.json_path
    answer = []
    # only 'color' category needs to add 5
    sample_num = len(os.listdir(image_folder)) + 5
    reward = torch.zeros((sample_num, np_index)).to(device='cuda')
    out_dir = args.out_dir

    order = "_blip"  # rename file
    for i in tqdm(range(np_index)):
        print(f"start VQA{i + 1}/{np_index}!")
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}", exist_ok=True)
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}/VQA/", exist_ok=True)
        Create_annotation_for_BLIP(
            json_path,
            image_folder,
            f"{out_dir}/annotation{i + 1}{order}",
            np_index=i,
        )
        answer_tmp = VQA_main(f"{out_dir}/annotation{i + 1}{order}/",
                              f"{out_dir}/annotation{i + 1}{order}/VQA/")
        answer.append(answer_tmp)

        with open(f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
            r = json.load(file)
        with open(f"{out_dir}/annotation{i + 1}{order}/vqa_test.json", "r") as file:
            r_tmp = json.load(file)
        for k in range(len(r)):
            if (r_tmp[k]['question'] != ''):
                reward[k][i] = float(r[k]["answer"])
            else:
                reward[k][i] = 1
        print(f"end VQA{i + 1}/{np_index}!")
    reward_final = reward[:, 0]
    for i in range(1, np_index):
        reward_final *= reward[:, i]

    # output final json
    with open(f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
        r = json.load(file)
    reward_after = 0
    for k in range(len(r)):
        r[k]["answer"] = '{:.4f}'.format(reward_final[k].item())
        reward_after += float(r[k]["answer"])
    os.makedirs(f"{out_dir}/annotation{order}", exist_ok=True)
    with open(f"{out_dir}/annotation{order}/vqa_result.json", "w") as file:
        json.dump(r, file)

    # calculate avg of BLIP-VQA as BLIP-VQA score
    print("BLIP-VQA score:", reward_after / len(r), '!\n')
    with open(f"{out_dir}/annotation{order}/blip_vqa_score.txt", "w") as file:
        file.write("BLIP-VQA score:" + str(reward_after / len(r)))


if __name__ == "__main__":
    main()
