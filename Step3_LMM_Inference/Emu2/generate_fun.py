import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from diffusers import DiffusionPipeline
import torch
PATH = "/path/to/Emu2-Gen"

def load_resources(path, gpu_ids):
    # 设置可见的 GPU 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            f"{path}/multimodal_encoder",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

    # 推断设备映射
    device_map = infer_auto_device_map(model, max_memory={0: '30GiB', 1: '80GiB'}, no_split_module_classes=['Block', 'LlamaDecoderLayer'])
    device_map["model.decoder.lm.lm_head"] = 0

    model = load_checkpoint_and_dispatch(
        model,
        f'{path}/multimodal_encoder',
        device_map=device_map
    ).eval()

    with init_empty_weights():
        pipe = DiffusionPipeline.from_pretrained(
            path,
            custom_pipeline="pipeline_emu2_gen",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="bf16",
            multimodal_encoder=model,
            tokenizer=tokenizer
        )
    
    pipe.safety_checker.to("cuda:0")
    pipe.unet.to("cuda:0")
    pipe.vae.to("cuda:0")

    return pipe

def generate(prompt, name_pict, pipe):
    ret = pipe(prompt)
    ret.image.save(name_pict)
    return True

# if __name__ == "__main__":
#     path = "/localnvme/application/sc_new/myy_multimodal/muyongyu/emu_demo/test.png"
#     pipe = load_resources(PATH, "0,4")  # 传入你想要使用的 GPU 编号
#     prompt = "English: A bed with four square wooden posts with balls on top and a striped Christmas stocking lying on it, with a decorated Christmas tree beside the bed, a corner wall single shelf, and four framed photos on wall in a room.\nSpanish: Una cama con cuatro postes cuadrados de madera con bolas en la parte superior y una media navideña a rayas sobre ella, con un árbol de Navidad decorado al lado de la cama, un estante de esquina en la pared y cuatro fotos enmarcadas en la pared de una habitación.\nRussian: Кровать с четырьмя квадратными деревянными столбиками с шарами на вершине и полосатым рождественским носком, лежащим на ней, рядом украшенная ёлка, настенная угловая полка и четыре обрамленных фотографии на стене в комнате.\nItalian:  letto con quattro montanti quadrati di legno con sfere sulla cima e una calza di Natale a righe sopra, con un albero di Natale decorato accanto al letto, una mensola angolare singola e quattro foto incorniciate sulla parete in una stanza.\nGerman: n Bett mit vier quadratischen Holzpfosten mit Kugeln darauf und einem gestreiften Weihnachtsstrumpf, der darauf liegt, mit einem geschmückten Weihnachtsbaum neben dem Bett, einem Eckregal an der Wand und vier gerahmten Fotos an der Wand in einem Zimmer.\nFrench:  lit avec quatre poteaux carrés en bois avec des boules sur le dessus et une chaussette de Noël rayée posée dessus, avec un arbre de Noël décoré à côté du lit, une étagère d'angle murale simple, et quatre photos encadrées sur le mur dans une pièce.\nChinese: 一张床有四根方形木柱，顶部有球，床上放着一只条纹圣诞袜，床旁边摆放着一棵装饰好的圣诞树，房间一角有一个单独的墙架，墙上挂着四幅装裱好的照片。"
    
#     generate(prompt, path, pipe)