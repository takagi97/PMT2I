import os
import re
import json
import random
import tqdm
import argparse
from request_chatgpt import make_requests
import tiktoken
import time

key = "sk-your_openai_key"

'''
code prompt fewshot (5) on WMT22 with demonstrations from WMT21
'''

prompt_v1 = '''Tell me only in ###target_language###. ###shot_n_text###[###source_language###]=[###input###][###target_language###]='''

prompt_v2 = '''###shot_n_text###\nGiven the ###source_language### sentence: ###input###\nthe ###target_language### translation of the sentence is:'''

prompt_v3 = '''###shot_n_text###\nTranslate this into 1. ###target_language###:\n###input###\n1.'''

prompt_v7 = '''###shot_n_text###Translate into ###target_language###.\\n[###source_language###]=[###input###][###target_language###]=[''' # same to MLP v6

prompt_v9 = '''###shot_n_text###\nTranslate into ###target_language###.\n###source_language###: ###input###\n'''

def encode_prompt(source_language, target_language, shot_n, input, prompt_version=1):
    """Encode multiple prompt sentences into a single string."""
    shot_n_text = ""
    if prompt_version == 1:
        prompt = prompt_v1
        for shot in shot_n:
            shot_n_text += f'''[{source_language}]=[{shot["src"]}][{target_language}]=[{shot["tgt"]}]'''
    elif prompt_version == 2:
        prompt = prompt_v2
        for shot in shot_n:
            shot_n_text += f'''Given the ###source_language### sentence: {shot["src"]}\nthe ###target_language### translation of the sentence is: {shot["tgt"]}'''
    elif prompt_version == 3:
        prompt = prompt_v3
        for shot in shot_n:
            shot_n_text += f'''Translate this into 1. ###target_language###:\n{shot["src"]}\n1. {shot["tgt"]}'''
    elif prompt_version == 7:
        prompt = prompt_v7
        for shot in shot_n:
            shot_n_text += '''Translate into ###target_language###.\\n'''
            shot_n_text += f'''[{source_language}]=[{shot["src"]}]\\n[{target_language}]=[{shot["tgt"]}]\\n'''
            shot_n_text += '''\\n'''
    elif prompt_version == 9:
        prompt = prompt_v9
        for index , shot in enumerate(shot_n):
            shot_string = f'''Translate into ###target_language###.\n{source_language}: {shot["src"]}\n{target_language}: {shot["tgt"]}'''
            if index == len(shot_n) - 1:
                shot_n_text += shot_string
            else:
                shot_string_ = shot_string + "\n"
                shot_n_text += shot_string_
    else:
        print(f"There are only 2 versions of the prompt")
        exit(0)

    prompt = prompt.replace("###shot_n_text###", shot_n_text)
    prompt = prompt.replace("###source_language###", source_language)
    prompt = prompt.replace("###target_language###", target_language)
    prompt = prompt.replace("###input###", input.strip())
    prompt = prompt + f"{target_language}: "

    return {"role": "user", "content": prompt}


def num_tokens_of_string(string: str, encoding_name="cl100k_base"):
    """Returns the number of tokens in a text string."""
    tokenizer = tiktoken.get_encoding(encoding_name)
    return len(tokenizer.encode(string))


def post_process_response(response):
    response = response.replace("\n", "").strip("=").strip("[").strip("]")
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_data_path",
        type=str,
        required=True,
        help="The directory where the result is stored.",
    )
    parser.add_argument(
        "--file_need_translate_path",
        type=str,
        required=True,
        help="The path to the file needing translation.",
    )
    parser.add_argument(
        "--lang_para_flores",
        type=str,
        required=True,
        default="German_#_deu_Latn_###_English_#_eng_Latn",
        help="src_lang1_#_src_file1_##_src_lang2_#_src_file2_###_tgt_lang1_#_tgt_file1",
    )
    parser.add_argument(
        "--shot_n",
        type=int,
        default=1,
        help="the number of demonstrations",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-4",
        help="The engine to use."
    )
    parser.add_argument(
        "--prompt_version",
        type=int,
        default=7,
        help="Which version of prompt you would like to use",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=3500,
        help="max number of the input prompt",
    )
    parser.add_argument(
        "--decoding_temperature",
        type=float,
        default=0,
        help="temperature"
    )
    parser.add_argument(
        "--TopP",
        type=float,
        default=1,
        help="TopP"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Fre",
        help="language"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()    

    lang_para_flores = args.lang_para_flores
    src_lang_flores = lang_para_flores.split("_###_")[0].split("_#_")[0]
    src_file_flores = lang_para_flores.split("_###_")[0].split("_#_")[1]
    tgt_lang_flores = lang_para_flores.split("_###_")[1].split("_#_")[0]
    tgt_file_flores = lang_para_flores.split("_###_")[1].split("_#_")[1]

    src_sents = []
    
    with open(args.file_need_translate_path,'r') as file_obj:
        Result_Obj = json.load(file_obj)
    src_sents =  Result_Obj.get("annotations")

    language = args.language
    if language == 'French':
        shots = [{'src': 'Ring also settled a lawsuit with competing security company, the ADT Corporation.', 
                'tgt': "Ring a également réglé un procès avec une entreprise de sécurité concurrente, l'ADT Corporation."}]
    elif language == "Russian":
        shots = [{'src': 'Television reports show white smoke coming from the plant.', 
                'tgt': 'В репортажах по телевидению сообщается  о белом дыме, идущем от завода.'}]
    elif language == "Spanish":
        shots = [{'src': 'Efforts to search for the crash site are being met by bad weather and harsh terrain.',
                'tgt': 'Los esfuerzos para hallar el lugar del accidente deben lidiar con el mal tiempo y el terreno escarpado.'},
                {"src":"The find also grants insight into the evolution of feathers in birds.",
                 "tgt":"el hallazgo permite comprender mejor la evolución de las plumas de las aves."},
                {"src":"The smaller the Rossby number, the less active the star with respect to magnetic reversals",
                 "tgt":"A menor número de Rossby, menor es la actividad de la estrella en relación con sus inversiones magnéticas."},
                {"src":"Over four million people went to Rome to attend the funeral.",
                 "tgt":"Más de cuatro millones de individuos se concentraron en Roma para presenciar el funeral."},
                {"src":"Television reports show white smoke coming from the plant.",
                 "tgt":"En las coberturas televisivas puede verse humo blanco emanando de la planta."},
                ]
    elif language == "German":
        shots = [{'src': 'French electoral law rather strictly codifies the proceedings.', 
                'tgt': 'Das französische Wahlgesetz legt den Ablauf ziemlich streng fest.'},
                {'src':'The result of plotting analysis will be posted to a public website.',
                 'tgt':'Das Ergebnis der Plotting-Analyse wird auf einer öffentlichen Website publiziert.'},
                {'src':'The first cases of the disease this season were reported in late July.',
                 'tgt':'Die ersten Fälle der Krankheit dieser Saison wurden Ende Juli gemeldet.'},
                {'src':'The truck driver, who is aged 64, was not injured in the crash.',
                 'tgt':'Der LKW-Fahrer, der 64 Jahre alt ist, wurde bei dem Unfall nicht verletzt.'},
                {'src':'Over four million people went to Rome to attend the funeral.',
                 'tgt':'Es kamen mehr als vier Millionen Menschen nach Rom, um an der Beerdigung teilzunehmen.'},
]
    elif language == "Italian":
        shots = [{'src': 'International sanctions have meant that new aircraft cannot be purchased.', 
                'tgt': 'Le sanzioni internazionali hanno significato non poter acquistare nuovi velivoli.'}]
    else:
        shots = [{'src': "One bomb exploded outside the governor general's office.", 
                'tgt': '一枚炸弹在总督办公室外爆炸。'}]
    
    # load the LM-generated sentences
    os.makedirs(args.result_data_path, exist_ok=True)
    machine_data = []
    if os.path.exists(os.path.join(args.result_data_path, "result.json")):
        with open(os.path.join(args.result_data_path, "result.json"), "r") as fin:
            for line in fin:
                line = line.strip()
                machine_data.append(line)
        print(f"Loaded {len(machine_data)} pieces of machine-generated data")
    

    # now let's generate new sentences!
    progress_bar = tqdm.tqdm(total=len(src_sents))
    if machine_data:
        progress_bar.update(len(machine_data))
    
    fout_prompt_writer = open(f'{args.result_data_path}/{tgt_lang_flores}_prompt.json','a')
    fout_writer = open(f'{args.result_data_path}/{tgt_lang_flores}_result.json','a')
    print(f'{args.result_data_path}/{tgt_lang_flores}_prompt.json')
    while len(machine_data) < len(src_sents):
        time.sleep(0.5)
        inner_dict = src_sents[len(machine_data)]
        prompt = encode_prompt(src_lang_flores, tgt_lang_flores, shots, inner_dict.get("caption"), prompt_version=args.prompt_version)
        font_prompt_inner_dict = inner_dict.copy()
        font_prompt_inner_dict[tgt_lang_flores] = prompt["content"].replace("\n", "\\n")
        fout_prompt_writer.write(json.dumps(font_prompt_inner_dict) + '\n')
        if num_tokens_of_string(prompt["content"]) <= args.max_input_tokens:
            result = make_requests(
                key=key,
                engine=args.engine,
                prompts=[prompt],
                max_tokens=(4000 - args.max_input_tokens),
                temperature=args.decoding_temperature,
                top_p=args.TopP,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n"]
                )[0]["response"]["choices"][0]["message"]["content"]
        else:
            result = "###over length!!!"
        # result = post_process_response(result)
        # result = re.sub(r'^\d+', '',result)[2:]
        font_inner_dict = inner_dict.copy()
        font_inner_dict[tgt_lang_flores] = result
        fout_writer.write(json.dumps(font_inner_dict) + '\n')
        machine_data.append("result")
        progress_bar.update(1)
    fout_prompt_writer.close()
    fout_writer.close()
    print("finish!")
