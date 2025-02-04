import unicodedata
import base64
import time
import json
import openai
import os
from tqdm import tqdm
import sys

openai.api_key = ""

def create_completion(image_path,prompt):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    base64_image = encode_image(image_path)
    err_count = 0
    completion = None
    while(True):
        try:
            completion = openai.ChatCompletion.create(
                messages=[
                {"role": "user", "content":[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": f'''
                                            You are responsible for judging the faithfulness of images generated by a computer program to the caption
                                            used to generate them . You will be presented with an image and given the caption that was used to produce
                                            the image . The captions you are judging are designed to stress - test image generation programs , and may
                                            include things such as :
                                            1. Scrambled or mis - spelled words ( the image generator should an image associated with the probably meaning )
                                            2. Color assignment ( the image generator should
                                            apply the correct color to the correct object )
                                            3. Counting ( the correct number of objects should be present )
                                            4. Abnormal associations , for example ’ elephant under a sea ’ , where the image should depict what is
                                            requested .
                                            5. Descriptions of objects , the image generator should draw the most commonly associated object .
                                            6. Rare single words , where the image generator should create an image somewhat associable with the
                                            specified image .
                                            7. Images with text in them , where the image generator should create an image with the specified text in it .
                                            You need to make a decision as to whether or not the image is correct , given the caption . You will first
                                            think out loud about your eventual conclusion , enumerating reasons why the image does or does not
                                            match the given caption . After thinking out loud , you should output either ’ Correct ’ or ’ Incorrect ’ depending
                                            on whether you think the image is faithful to the caption .
                                            A few rules :
                                            1. Do not nitpick . If the caption requests an object and the object is generally depicted correctly , then
                                            you should answer ’ Correct ’.
                                            2. Ignore other objects in the image that are not explicitly mentionedby the caption ; it is fine for these to
                                            be shown .
                                            3. It is also OK if the object being depicted is slightlydeformed , as long as a human would recognize it and
                                            it does not violate the caption .
                                            4. Your response must always end with either ’ incorrect ’ or ’ correct ’
                                            5. ’ Incorrect ’ should be reserved for instances where a specific aspect of the caption is not followed correctly ,
                                            such as a wrong object , color or count .
                                            6. You must keep your thinking out loud short , less than 50 words .
                                            {prompt}
                                '''
                            }
                        ]}
            ],
                model="gpt-4o",
            )
            break
        except:
            err_count += 1
        if err_count > 20:
            completion = None
            break
    return completion

def generate_res(completion):
    if completion is not None:
        message = completion.choices[0].message
        content = unicodedata.normalize('NFKC', message.content)
    else:
        content = "WRONG LINE!!!"
    return content

if __name__ == "__main__":
    count = 0
    eval_file_name = sys.argv[1]
    json_file = "../Our_Results/Prompts/DrawBench/en_de_copy4.json"
    gen_file = f"/your/predict/path"
    json_contents = []
    gen_contents = os.listdir(gen_file)
    gen_contents.sort(key=lambda x:int(x.split(".")[0]))
    with open(json_file,'r') as file:
        for content_string in file:
            content = json.loads(content_string)
            json_contents.append(content)
    
    json_ids = [str(i.get("id")) + ".jpg" for i in json_contents]
    json_ids_new = []
    for old in json_ids:
        if old in gen_contents:
            json_ids_new.append(old)
    for i, j in zip(json_ids_new,gen_contents):
        print(i,j)
        if i != j:
            raise "id error" 

    output_file = f"you/want/to/save/path"
    already_count = 0
    if os.path.exists(os.path.join(output_file)):
        with open(os.path.join(output_file), "r") as fin:
            for line in fin:
                already_count += 1
        print(f"Loaded {already_count} pieces of machine-generated data")

    writer = open(output_file,'a')
    for num in tqdm(range(len(json_ids_new))):
        if already_count > count:
            count += 1
            continue
        print(num)
        prompt = json_contents[num].get("caption")
        image_path = gen_file + "/" + gen_contents[num]
        completion = create_completion(image_path,prompt)
        content = generate_res(completion)
        writer.write(json.dumps(content)+"\n")
        print(content)
        count += 1
        print(count)
        