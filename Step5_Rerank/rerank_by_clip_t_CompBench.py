import os
import torch
from PIL import Image
from tqdm import tqdm
import clip
import json
from scipy import spatial


def calculate_clip_t_score(model, transform, image_path, text_features, device):
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).detach().cpu().float()
        return image_features

    generated_features = encode(Image.open(image_path).convert('RGB'), model, transform)
    gen_clip_t = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                             text_features.view(text_features.shape[1]))

    return gen_clip_t


def select_best_images(input_dirs, output_dir, caption_dict):
    device = "cuda:1"
    model, transform = clip.load("ViT-B/32", device)

    rerank_statistics = {os.path.basename(os.path.dirname(os.path.dirname(input_dir))): 0 for input_dir in input_dirs}
    # Get list of images
    image_files = sorted(os.listdir(input_dirs[0]))

    for image_file in tqdm(image_files, desc="Processing images"):
        best_score = float('-inf')
        best_image_path = None
        best_input_dir = None
        gt_caption = caption_dict[image_file]
        text_features = clip.tokenize(gt_caption, truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_features).detach().cpu().float()

        for input_dir in input_dirs:
            image_path = os.path.join(input_dir, image_file)
            score = calculate_clip_t_score(model, transform, image_path, text_features, device)

            if score > best_score:
                best_score = score
                best_image_path = image_path
                best_input_dir = input_dir

        if best_image_path and best_input_dir:
            prefix = os.path.basename(os.path.dirname(os.path.dirname(best_input_dir)))
            rerank_statistics[prefix] += 1
            new_image_name = f"{prefix}.{image_file}"
            output_image_path = os.path.join(output_dir, new_image_name)
            Image.open(best_image_path).save(output_image_path)

    for key in rerank_statistics:
        print(rerank_statistics[key], end=" ")


if __name__ == "__main__":
    input_dirs = [
        "/your/predict/path1",
        "/your/predict/path2",
        "/your/predict/path3",
        "/your/predict/path4",
        "/your/predict/path5",
        "/your/predict/path6"
    ]
    output_dir = "/your/rerank/path"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.makedirs(output_dir, exist_ok=True)
    with open("Utils/cb_id2caption/color_output.json", 'r') as f:
        caption_dict = json.load(f)

    select_best_images(input_dirs, output_dir, caption_dict)
