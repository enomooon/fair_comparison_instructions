from prompt_template import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import os
import argparse
import random
from collections import Counter
import datasets

random.seed(42)


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer


def load_data_ls(lang_dataset):
    if lang_dataset == 'Chinese':
        with open('./../datasets/Chinese-LS/dataset/eno_valid.json') as f_exam,\
             open('./../datasets/Chinese-LS/dataset/eno_test.json') as f_test:
            dataset_exam = json.load(f_exam)
            dataset_test = json.load(f_test)
    else:
        dataset_exam = datasets.Dataset.from_csv(f"https://huggingface.co/datasets/MLSP2024/MLSP2024/resolve/main/{lang_dataset}/multils_trial_{lang_dataset.lower()}_ls_labels.tsv", sep='\t')
        dataset_test = datasets.Dataset.from_csv(f"https://huggingface.co/datasets/MLSP2024/MLSP2024/resolve/main/{lang_dataset}/multils_test_{lang_dataset.lower()}_ls_labels.tsv", sep='\t')
        # dataset_exam = load_dataset("MLSP2024/MLSP2024",f'{lang_dataset.lower()}_ls_labels', split="trial")
        # dataset_test = load_dataset("MLSP2024/MLSP2024",f'{lang_dataset.lower()}_ls_labels', split="test")
    return dataset_test, dataset_exam


def get_exam_ans(exam):
    sentence = exam['context']
    word = exam['target']
    answers = [exam[f'substitution_{num}'] for num in range(1, 31) if exam[f'substitution_{num}'] is not None]
    answers_freq = dict(Counter(answers))
    for now_exam, now_freq in answers_freq.items():
        if now_exam != word:
            return now_exam
    return word


def get_exam_ans_zh(exam):
    sentence = exam['context']
    word = exam['target']
    answers = exam['substitutions'].strip().split()
    for now_exam in answers:
        if now_exam != word:
            return now_exam
    return word


def generation(model_name, model, tokenizer, prompt):
    if 'Mistral-Nemo' in model_name:
        tokenized_input = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
    else:
        tokenized_input = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **tokenized_input,
            max_new_tokens=30,
            do_sample=True,
            top_p=0.9,
            temperature=0.6
        )[0]

    output = tokenizer.decode(output, skip_special_tokens=False)
    output = output.replace(tokenizer.decode(
        tokenized_input['input_ids'][0], skip_special_tokens=False), '')
    return output


def normalize_pred(pred, eos_token, eos_token_2):
    pred = pred.strip().split('\n')[0]
    pred = pred.split(eos_token)[0].strip()
    if eos_token_2 is not None:
        pred = pred.split(eos_token_2)[0].strip()
    if len(pred) > 0:
        if pred[-1] == '.' or pred[-1] == '。':
            pred = pred[:-1]
        pred = pred.strip()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_dataset', '-ld', 
                        required=True, help='language of dataset', choices=['de', 'es', 'fr', 'ja', 'zh'])
    parser.add_argument('--lang_prompt', '-lp',
                        required=True, help='language of instruction', choices=['en', 'tgt', 'tgt-translate'])
    parser.add_argument('--num_shot', '-ns', 
                        default='zero', help='num of examples', choices=['zero', 'few'])
    parser.add_argument('--model_name_ease', '-mne', 
                        required=True, help='short model name like qwen2-i', choices=['llama3-i', 'qwen2-i', 'mistraln-i', 'llama3-b', 'qwen2-b', 'mistraln-b'])
    parser.add_argument('--use_chat_template', '-uct', 
                        help='when using chat_template, please flag this', action='store_true')

    args = parser.parse_args()
    lang_dataset = args.lang_dataset
    lang_prompt = args.lang_prompt
    num_shot = args.num_shot
    model_name_ease = args.model_name_ease
    use_chat_template = args.use_chat_template

    lang_short2long = {'de':'German', 
                        'es':'Spanish', 
                        'fr':'French', 
                        'ja':'Japanese',
                        'zh':'Chinese'}
    lang_dataset_long = lang_short2long[lang_dataset]
    print('lang_dataset is', lang_dataset_long)
    
    if model_name_ease == 'llama3-b':
        model_name = 'meta-llama/Meta-Llama-3-8B'
        eos_token = '<|end_of_text|>'
        eos_token_2 = '<|eot_id|>'
    elif model_name_ease == 'llama3-i':
        model_name = 'lightblue/suzume-llama-3-8B-multilingual'
        eos_token = '<|end_of_text|>'
        eos_token_2 = '<|eot_id|>'
    elif model_name_ease == 'qwen2-b':
        model_name = 'Qwen/Qwen2-7B'
        eos_token = '<|endoftext|>'
        eos_token_2 = '<|im_end|>'
    elif model_name_ease == 'qwen2-i':
        model_name = 'Qwen/Qwen2-7B-Instruct'
        eos_token = '<|endoftext|>'
        eos_token_2 = '<|im_end|>'
    elif model_name_ease == 'mistraln-b':
        model_name = 'mistralai/Mistral-Nemo-Base-2407'
        eos_token = '</s>'
        eos_token_2 = None
    elif model_name_ease == 'mistraln-i':
        model_name = 'mistralai/Mistral-Nemo-Instruct-2407'
        eos_token = '</s>'
        eos_token_2 = None

    print(model_name)
    model, tokenizer = load_model(model_name)
    dataset_test, dataset_trial = load_data_ls(lang_dataset_long)

    
    cnt_cor = 0  # count of correct predictions
    datas_new = list()

    prompt_funcs = {
        'en': prompt_ls_en,
        'de': prompt_ls_de,
        'es': prompt_ls_es,
        'fr': prompt_ls_fr,
        'ja': prompt_ls_ja,
        'zh': prompt_ls_zh,
        'de_translate': prompt_ls_de_translate,
        'es_translate': prompt_ls_es_translate,
        'fr_translate': prompt_ls_fr_translate,
        'ja_translate': prompt_ls_ja_translate,
        'zh_translate': prompt_ls_zh_translate,
        'en_exam': prompt_ls_en_with_example,
        'de_exam': prompt_ls_de_with_example,
        'es_exam': prompt_ls_es_with_example,
        'fr_exam': prompt_ls_fr_with_example,
        'ja_exam': prompt_ls_ja_with_example,
        'zh_exam': prompt_ls_zh_with_example,
        'de_translate_exam': prompt_ls_de_translate_with_example,
        'es_translate_exam': prompt_ls_es_translate_with_example,
        'fr_translate_exam': prompt_ls_fr_translate_with_example,
        'ja_translate_exam': prompt_ls_ja_translate_with_example,
        'zh_translate_exam': prompt_ls_zh_translate_with_example
    }

    example_funcs = {
        'en': example_ls_en, 
        'de': example_ls_de, 
        'es': example_ls_es, 
        'fr': example_ls_fr,
        'ja': example_ls_ja,
        'zh': example_ls_zh,
        'de_translate': example_ls_de_translate, 
        'es_translate': example_ls_es_translate, 
        'fr_translate': example_ls_fr_translate,
        'ja_translate': example_ls_ja_translate, 
        'zh_translate': example_ls_zh_translate, 
    }

    for line in tqdm(dataset_test):
        sentence = line['context'].strip()
        word = line['target'].strip()
        if lang_dataset_long == 'Chinese':
            answers = line["substitutions"].strip().split()
        else:
            answers = [line[f'substitution_{num}'] for num in range(1, 31) if line[f'substitution_{num}'] is not None]
        answers_freq = dict(Counter(answers))

        data_new = {'sentence': sentence,'word': word, 'answers': answers_freq}

        if lang_prompt == 'en':
            if num_shot == 'zero':
                prompt = prompt_funcs[lang_prompt](sentence, word, lang_dataset_long)
            elif num_shot == 'few':
                # randomly select examples
                exams_ids = random.sample(range(len(dataset_trial)), 4)
                exams = [dataset_trial[idx] for idx in exams_ids]
                if lang_dataset_long == 'Chinese':
                    exams_ans = [get_exam_ans_zh(exam) for exam in exams]
                else:
                    exams_ans = [get_exam_ans(exam) for exam in exams]
                exams_format = [example_funcs[lang_prompt](
                    exams[i]['context'], exams[i]['target'], exams_ans[i]) for i in range(len(exams))]
                exams_concat = '\n'.join(exams_format)
                prompt = prompt_funcs[f'{lang_prompt}_exam'](sentence, word, exams_concat, lang_dataset_long)

        elif lang_prompt == 'tgt':
            if num_shot == 'zero':
                prompt = prompt_funcs[lang_dataset](sentence, word)
            elif num_shot == 'few':
                exams_ids = random.sample(range(len(dataset_trial)), 4)
                exams = [dataset_trial[idx] for idx in exams_ids]
                if lang_dataset_long == 'Chinese':
                    exams_ans = [get_exam_ans_zh(exam) for exam in exams]
                else:
                    exams_ans = [get_exam_ans(exam) for exam in exams]
                exams_format = [example_funcs[lang_dataset](
                    exams[i]['context'], exams[i]['target'], exams_ans[i]) for i in range(len(exams))]
                exams_concat = '\n'.join(exams_format)
                prompt = prompt_funcs[f'{lang_dataset}_exam'](sentence, word, exams_concat)

        else:
            if num_shot == 'zero':
                prompt = prompt_funcs[f'{lang_dataset}_translate'](sentence, word)
            elif num_shot == 'few':
                exams_ids = random.sample(range(len(dataset_trial)), 4)
                exams = [dataset_trial[idx] for idx in exams_ids]
                if lang_dataset_long == 'Chinese':
                    exams_ans = [get_exam_ans_zh(exam) for exam in exams]
                else:
                    exams_ans = [get_exam_ans(exam) for exam in exams]
                exams_format = [example_funcs[f'{lang_dataset}_translate'](
                    exams[i]['context'], exams[i]['target'], exams_ans[i]) for i in range(len(exams))]
                exams_concat = '\n'.join(exams_format)
                prompt = prompt_funcs[f'{lang_dataset}_translate_exam'](sentence, word, exams_concat)


        prompt = prompt.replace('</s>', tokenizer.eos_token)
        if use_chat_template:
            messages = [{'role':'user', 'content':prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(prompt)

        output = generation(model_name, model, tokenizer, prompt)
        output_normalize = normalize_pred(output, eos_token, eos_token_2)
        print(output_normalize)

        data_new['pred'] = output
        data_new['pred_normalize'] = output_normalize
        datas_new.append(data_new)
        if output_normalize in answers_freq.keys():
            cnt_cor += 1

    print('cnt_cor: ', cnt_cor)
    f1 = cnt_cor / len(dataset_test)

    pred_dir = f'./../pred/ls/{lang_dataset}/{num_shot}'
    result_dir = f'./../result/ls/{lang_dataset}/{num_shot}'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(f'{pred_dir}/{model_name_ease}_{lang_prompt}.json', 'w') as f_pred:
        json.dump(datas_new, f_pred, ensure_ascii=False)

    with open(f'{result_dir}/{model_name_ease}_{lang_prompt}.txt', 'a') as f_result:
        f_result.write('cnt_cor: ' + str(cnt_cor) + '\n')
        f_result.write('f1: ' + str(f1) + '\n')


if __name__ == '__main__':
    main()
