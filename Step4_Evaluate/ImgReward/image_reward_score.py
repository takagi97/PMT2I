import json
import os
import argparse
import ImageReward as reward
import torch
from tqdm import tqdm


def args_():
    parser = argparse.ArgumentParser(description="image reward score")
    parser.add_argument("--json_file", required=True, help="the json file of captions", type=str)
    parser.add_argument("--folder_path", required=True, help="the real path of image", type=str)
    parser.add_argument("--dataset_type", required=True, help="we used dataset", type=str)
    args = parser.parse_args()
    return args


def load_model():
    model = reward.load("ImageReward-v1.0")
    return model




def _change_folder(folder_path):

    def _getfilelist(folder_path,prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(name)
            # not in order
        # folder.sort(key=lambda x:x.split("-")[1])
        return folder
    folder = _getfilelist(folder_path)
    return folder

def _cheak(json_path, folder_content,type="COCO"):
    flag = True
    # json_content = []
    if type=="COCO":
        folder_content.sort(key=lambda x:x.split("-")[1])
        with open(json_path, 'r') as f:
            json_content = json.load(f).get("annotations")
        json_content.sort(key=lambda x: x['image_id'])
        ids = [i.split('-')[0] for i in folder_content]
        json_content_new = []
        print(len(json_content))
        for j in json_content:
            id = str(j.get("id"))
            if id in ids:
                json_content_new.append(j)

        for json_file ,folder_file in tqdm(zip(json_content_new,folder_content)):
            json_pict_name = str(json_file.get("image_id")) + ".jpg"
            folder_pict_name = folder_file.split("_")[-1].lstrip("0")
            if json_pict_name == folder_pict_name:
                pass
            else:
                print(f"json_pict_name:{json_pict_name}")
                print(f"folder_pict_name:{folder_pict_name}")
                flag = False
                break
    elif type == "DB":
        folder_content.sort(key=lambda x:int(x.split(".")[0]))
        json_content = []
        with open(json_path,'r') as f:
            for content_string in f:
                content = json.loads(content_string)
                json_content.append(content)
        json_content.sort(key=lambda x: x['id'])
        print(folder_content[0])
        ids = [i.split('.')[0] for i in folder_content]
        json_content_new = []
        print(len(json_content))
        for j in json_content:
            id = str(j.get("id"))
            if id in ids:
                json_content_new.append(j)

        json_pict_name = [str(i.get("id"))+".jpg" for i in json_content_new]
        for i,j in tqdm(zip(folder_content,json_pict_name)):
            if i == j:
                pass
            else:
                print(i)
                print(j)
                flag = False
                break
            

    return flag,json_content,folder_content


def main():
    args = args_()
    json_path = args.json_file
    folder_path = args.folder_path
    dataset_type = args.dataset_type
    model = load_model()
    folder_score = 0
    count = 0
    folder_content = _change_folder(folder_path)
    flag , json_content, folder_content = _cheak(json_path, folder_content,dataset_type)
    if not flag:
        raise "pairs error"
    for prompt, folder_img in tqdm(zip(json_content, folder_content)):
        images = [folder_path + "/" + folder_img]
        with torch.no_grad():
            rewards = model.inference_rank(prompt.get("caption"), images)[1]
            folder_score += rewards
            count += 1
    print("folder score:", folder_score / count)

if __name__ == '__main__':
    main()

