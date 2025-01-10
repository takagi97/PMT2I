# Boosting Text-To-Image Generation via Multilingual Prompting in Large Multimodal Models

This repository contains the datasets, code, and results for our ICASSP 2025 [paper]().

<details>
  <summary>Here is our abstract.</summary>
  Previous work on augmenting large multimodal models (LMMs) for text-to-image (T2I) generation has focused on enriching the input space of in-context learning (ICL). This includes providing a few demonstrations and optimizing image descriptions to be more detailed and logical. However, as demand for more complex and flexible image descriptions grows, enhancing comprehension of input text within the ICL paradigm remains a critical yet underexplored area. In this work, we extend this line of research by constructing parallel multilingual prompts aimed at harnessing the multilingual capabilities of LMMs. More specifically, we translate the input text into several languages and provide the models with both the original text and the translations. Experiments on two LMMs across 3 benchmarks show that our method, PMT2I, achieves superior performance in general, compositional, and fine-grained assessments, especially in human preference alignment. Additionally, with its advantage of generating more diverse images, PMT2I significantly outperforms baseline prompts when incorporated with reranking methods.
</details>

## Overview

We release the code, data, and some results of our work in this repository, which is structured as follows:

### Datasets
Contains the image descriptions of the three benchmarks used in our experiments, including [CompBench](https://github.com/Karine-Huang/T2I-CompBench), [DrawBench](https://docs.google.com/spreadsheets/d/1y7nAbmR4FREi6npB1u-Bo3GFdwdOPYJc617rBOxIRHY/edit#gid=0), and [MS_COCO](https://cocodataset.org/#download). Note that only MS_COCO includes ground truth images, while the other datasets contain only text inputs. You can download the full MS_COCO (text and ground truth images) by clicking [here](http://images.cocodataset.org/zips/val2014.zip).

### Our Results
Stores the outcomes of our experiments. It includes prompts and translations for each dataset. The image descriptions of these benchmarks are in English. And we translated these English texts into six languages. For the MS-COCO dataset, we employed GPT-4o to translate into German and Spanish, DeepL for French and Italian translations, and NiuTrans for Russian and Chinese. For other benchmarks, GPT-4o was utilized to translate into all six languages. **It's worth to note that these translations also serve as parallel multlingual T2I benchmarks, and can be used directly :)**

### Steps to Reproduce Results
Each step in the repository has its own `Run.sh` script that details how to execute the programs. You can start each step independently, but it is recommended to follow the sequential order:
- **Step1_Translate:** Scripts for translating the English image descritions into multiple languages.
- **Step2_Construct_PMT2:** Scripts for constructing prompts using the translated data.
- **Step3_LMM_Inference:** Inference scripts for generating images from the constructed prompts, including Lumina and Emu2.
- **Step4_Evaluate:** Evaluation scripts for assessing the quality of generated images, including BliScore, Clip_Dino, GPT4o-Judgement, and ImgReward.
- **Step5_Rerank:** Scripts for reranking images based on model evaluations.

## Requirements

To translate the image descriptions and request GPTo to evaluate via OpenAI's API, you will need Python 3.8+, and the following libraries:
- openai
- tqdm
- tiktoken

Installation commands:
```bash
pip install openai tqdm datetime tiktoken
```

For other environments used to conduct inference and evaluation, please the requirement files in Requirements.

## Citation

If you find this repository or our results help your research, please consider to cite our paper:
```bibtex
Boosting Text-To-Image Generation via Multilingual Prompting in Large Multimodal Models
```

## Acknowledgments

Thanks to everyone who contributed to this project, and special thanks to the ICASSP reviewers and our funding agencies.

## Contact

For any inquiries, don't hesitate 🤗, please open an issue or contact lixiaoyumu9 [at] gmail [dot] com.
