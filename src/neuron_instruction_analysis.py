from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from prompt_template import *
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
import random
from tqdm import tqdm
import argparse
from datasets import load_dataset
from neuron_activation import get_out_llm, get_over_zero_vec
import os
from infer_rc import load_model

random.seed(42)

def load_lang_neuron_info(path):
    ''' load infomation of lang specific neuron'''
    active_neurons = torch.load(path)
    active_neurons_en = active_neurons[0]
    active_neurons_fr = active_neurons[1]
    active_neurons_es = active_neurons[2]
    active_neurons_de = active_neurons[3]
    active_neurons_zh = active_neurons[4]
    active_neurons_ja = active_neurons[5]
    return active_neurons_en, active_neurons_fr, active_neurons_es, active_neurons_de, active_neurons_zh, active_neurons_ja 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_ease', '-mne', 
                        required=True, help='short model name like qwen2-i', choices=['llama3-i', 'qwen2-i', 'mistraln-i', 'llama3-b', 'qwen2-b', 'mistraln-b'])
    parser.add_argument('--lang_dataset', '-ld', required=True, choices=['de', 'es', 'fr', 'ja', 'zh'])

    args = parser.parse_args()
    model_name_ease = args.model_name_ease
    lang_dataset = args.lang_dataset
    label_nl = ['bad', 'good']

    num_instance = 100
    save_dir = f'./../neuron/rc/en'

    dataset_tar = load_dataset(f"SetFit/amazon_reviews_multi_{lang_dataset}", split="test")
    dataset_tar_label_0 = dataset_tar.filter(lambda x: x["label"]==0) # ラベルが0のデータ
    dataset_tar_label_1 = dataset_tar.filter(lambda x: x["label"]==1)
    dataset_tar_label_3 = dataset_tar.filter(lambda x: x["label"]==3)
    dataset_tar_label_4 = dataset_tar.filter(lambda x: x["label"]==4)
    random_ids = random.sample(range(1000), k=25)

    if model_name_ease == 'llama3-b':
        model_name = 'meta-llama/Meta-Llama-3-8B'
    elif model_name_ease == 'llama3-i':
        model_name = 'lightblue/suzume-llama-3-8B-multilingual'
    elif model_name_ease == 'qwen2-b':
        model_name = 'Qwen/Qwen2-7B'
    elif model_name_ease == 'qwen2-i':
        model_name = 'Qwen/Qwen2-7B-Instruct'
    elif model_name_ease == 'mistraln-b':
        model_name = 'mistralai/Mistral-Nemo-Base-2407'
    elif model_name_ease == 'mistraln-i':
        model_name = 'mistralai/Mistral-Nemo-Instruct-2407'

    model, tokenizer = load_model(model_name)

    active_neurons_en, active_neurons_fr, active_neurons_es, active_neurons_de, active_neurons_zh, active_neurons_ja = load_lang_neuron_info(f'./../neuron/activation/activation_mask/{model_name_ease}')

    layers = [str(i+1) for i in range(len(active_neurons_en))]
    num_layers = model.config.num_hidden_layers

    function_map_for_prompt = {'fr_rc':prompt_rc_fr, 
                            'es_rc':prompt_rc_es, 
                            'de_rc':prompt_rc_de, 
                            'zh_rc':prompt_rc_zh, 
                            'ja_rc':prompt_rc_ja}

    # Number of neurons in a paticular state like both active
    cnts_same_active, cnts_same_nonactive, cnts_only_en_active, cnts_only_tar_active = [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))] # 層ごと

    # Number of language-specific neurons to be activated
    cnts_en_lang_en, cnts_fr_lang_en, cnts_es_lang_en, cnts_de_lang_en, cnts_zh_lang_en, cnts_ja_lang_en = [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))]
    cnts_en_lang_tar, cnts_fr_lang_tar, cnts_es_lang_tar, cnts_de_lang_tar, cnts_zh_lang_tar, cnts_ja_lang_tar = [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))], [0 for _ in range(len(layers))]

    # sampling instance
    prompts_en, prompts_tar = [], []
    for random_idx in random_ids:
        sentence_tar_0 = dataset_tar_label_0[random_idx]['text'].strip()
        sentence_tar_1 = dataset_tar_label_1[random_idx]['text'].strip()
        sentence_tar_3 = dataset_tar_label_3[random_idx]['text'].strip()
        sentence_tar_4 = dataset_tar_label_4[random_idx]['text'].strip()
        sentences_tar = [sentence_tar_0, sentence_tar_1, sentence_tar_3, sentence_tar_4]

        for sentence_tar in sentences_tar:
            input_text_en = prompt_rc_en(sentence_tar, label_nl)
            input_text_tar = function_map_for_prompt[f'{lang_dataset}_rc'](sentence_tar, label_nl)
            prompts_en.append(input_text_en)
            prompts_tar.append(input_text_tar)

    # camparing neuron states
    for i in tqdm(range(num_instance)):
        input_ids_en = tokenizer(prompts_en[i], return_tensors="pt")['input_ids'].to(model.device)
        input_ids_tar = tokenizer(prompts_tar[i], return_tensors="pt")['input_ids'].to(model.device)

        output_en = get_out_llm(model, input_ids_en, num_layers)
        output_tar = get_out_llm(model, input_ids_tar, num_layers)

        for layer in range(num_layers):
            output_en_now = output_en[layer].to('cpu').detach().numpy().copy()
            output_tar_now = output_tar[layer].to('cpu').detach().numpy().copy()
            binary_vec_en = get_over_zero_vec(output_en_now[0][-1])
            binary_vec_tar = get_over_zero_vec(output_tar_now[0][-1])
            for i in range(len(binary_vec_en)):
                if binary_vec_en[i] == 1: 
                    if binary_vec_tar[i] == 1:
                        cnts_same_active[layer] += 1
                    elif binary_vec_tar[i] == 0:
                        cnts_only_en_active[layer] += 1

                    if i in active_neurons_en[layer]:
                        cnts_en_lang_en[layer] += 1
                    if i in active_neurons_ja[layer]:
                        cnts_ja_lang_en[layer] += 1
                    if i in active_neurons_de[layer]:
                        cnts_de_lang_en[layer] += 1
                    if i in active_neurons_es[layer]:
                        cnts_es_lang_en[layer] += 1
                    if i in active_neurons_fr[layer]:
                        cnts_fr_lang_en[layer] += 1
                    if i in active_neurons_zh[layer]:
                        cnts_zh_lang_en[layer] += 1
                    
                if binary_vec_tar[i] == 1:
                    if binary_vec_en[i] == 0:
                        cnts_only_tar_active[layer] += 1

                    if i in active_neurons_en[layer]:
                        cnts_en_lang_tar[layer] += 1
                    if i in active_neurons_ja[layer]:
                        cnts_ja_lang_tar[layer] += 1
                    if i in active_neurons_de[layer]:
                        cnts_de_lang_tar[layer] += 1
                    if i in active_neurons_es[layer]:
                        cnts_es_lang_tar[layer] += 1
                    if i in active_neurons_fr[layer]:
                        cnts_fr_lang_tar[layer] += 1
                    if i in active_neurons_zh[layer]:
                        cnts_zh_lang_tar[layer] += 1

                if binary_vec_en[i] == 0 and binary_vec_tar[i] == 0:
                    cnts_same_nonactive[layer] += 1


    cnt_en_lang_en_all, cnt_fr_lang_en_all, cnt_es_lang_en_all, cnt_de_lang_en_all, cnt_zh_lang_en_all, cnt_ja_lang_en_all = 0, 0, 0, 0, 0, 0
    cnt_en_lang_tar_all, cnt_fr_lang_tar_all, cnt_es_lang_tar_all, cnt_de_lang_tar_all, cnt_zh_lang_tar_all, cnt_ja_lang_tar_all = 0, 0, 0, 0, 0, 0
    cnt_en_active_neurons, cnt_fr_active_neurons, cnt_es_active_neurons, cnt_de_active_neurons, cnt_zh_active_neurons, cnt_ja_active_neurons = 0, 0, 0, 0, 0, 0
    # Total for all layers
    for layer in range(num_layers):
        cnt_en_lang_en_all += cnts_en_lang_en[layer]
        cnt_fr_lang_en_all += cnts_fr_lang_en[layer]
        cnt_es_lang_en_all += cnts_es_lang_en[layer]
        cnt_de_lang_en_all += cnts_de_lang_en[layer]
        cnt_zh_lang_en_all += cnts_zh_lang_en[layer]
        cnt_ja_lang_en_all += cnts_ja_lang_en[layer]

        cnt_en_lang_tar_all += cnts_en_lang_tar[layer]
        cnt_fr_lang_tar_all += cnts_fr_lang_tar[layer]
        cnt_es_lang_tar_all += cnts_es_lang_tar[layer]
        cnt_de_lang_tar_all += cnts_de_lang_tar[layer]
        cnt_zh_lang_tar_all += cnts_zh_lang_tar[layer]
        cnt_ja_lang_tar_all += cnts_ja_lang_tar[layer]

        cnt_en_active_neurons += len(active_neurons_en[layer])
        cnt_fr_active_neurons += len(active_neurons_fr[layer])
        cnt_es_active_neurons += len(active_neurons_es[layer])
        cnt_de_active_neurons += len(active_neurons_de[layer])
        cnt_zh_active_neurons += len(active_neurons_zh[layer])
        cnt_ja_active_neurons += len(active_neurons_ja[layer])

        cnts_same_active[layer] = round(cnts_same_active[layer] / num_instance)
        cnts_same_nonactive[layer] = round(cnts_same_nonactive[layer] / num_instance)
        cnts_only_en_active[layer] = round(cnts_only_en_active[layer] / num_instance)
        cnts_only_tar_active[layer] = round(cnts_only_tar_active[layer] / num_instance)

    per_en_lang_en_all = (cnt_en_lang_en_all / num_instance) / cnt_en_active_neurons
    per_fr_lang_en_all = (cnt_fr_lang_en_all / num_instance) / cnt_fr_active_neurons
    per_es_lang_en_all = (cnt_es_lang_en_all / num_instance) / cnt_es_active_neurons
    per_de_lang_en_all = (cnt_de_lang_en_all / num_instance) / cnt_de_active_neurons
    per_zh_lang_en_all = (cnt_zh_lang_en_all / num_instance) / cnt_zh_active_neurons
    per_ja_lang_en_all = (cnt_ja_lang_en_all / num_instance) / cnt_ja_active_neurons

    per_en_lang_tar_all = (cnt_en_lang_tar_all / num_instance) / cnt_en_active_neurons
    per_fr_lang_tar_all = (cnt_fr_lang_tar_all / num_instance) / cnt_fr_active_neurons
    per_es_lang_tar_all = (cnt_es_lang_tar_all / num_instance) / cnt_es_active_neurons
    per_de_lang_tar_all = (cnt_de_lang_tar_all / num_instance) / cnt_de_active_neurons
    per_zh_lang_tar_all = (cnt_zh_lang_tar_all / num_instance) / cnt_zh_active_neurons
    per_ja_lang_tar_all = (cnt_ja_lang_tar_all / num_instance) / cnt_ja_active_neurons

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    txt_path = f'{save_dir}/distribute_percent_lang_neuron_instruct_{model_name_ease}_{lang_dataset}.txt'
    with open(txt_path, 'a') as f:
        f.write(f'rc_{model_name_ease}_{lang_dataset}\n')
        f.write('Divide the two columns on the right, Average number of language neurons activated in each instance, Number of language neurons')
        f.write('English Instruction\n')
        f.write(f'en: {per_en_lang_en_all}, {cnt_en_lang_en_all / num_instance}, {cnt_en_active_neurons}\n')
        f.write(f'fr: {per_fr_lang_en_all}, {cnt_fr_lang_en_all / num_instance}, {cnt_fr_active_neurons}\n')
        f.write(f'es: {per_es_lang_en_all}, {cnt_es_lang_en_all / num_instance}, {cnt_es_active_neurons}\n')
        f.write(f'de: {per_de_lang_en_all}, {cnt_de_lang_en_all / num_instance}, {cnt_de_active_neurons}\n')
        f.write(f'zh: {per_zh_lang_en_all}, {cnt_zh_lang_en_all / num_instance}, {cnt_zh_active_neurons}\n')
        f.write(f'ja: {per_ja_lang_en_all}, {cnt_ja_lang_en_all / num_instance}, {cnt_ja_active_neurons}\n\n')
        f.write('Target_language Instruction\n')
        f.write(f'en: {per_en_lang_tar_all}, {cnt_en_lang_tar_all / num_instance}, {cnt_en_active_neurons}\n')
        f.write(f'fr: {per_fr_lang_tar_all}, {cnt_fr_lang_tar_all / num_instance}, {cnt_fr_active_neurons}\n')
        f.write(f'es: {per_es_lang_tar_all}, {cnt_es_lang_tar_all / num_instance}, {cnt_es_active_neurons}\n')
        f.write(f'de: {per_de_lang_tar_all}, {cnt_de_lang_tar_all / num_instance}, {cnt_de_active_neurons}\n')
        f.write(f'zh: {per_zh_lang_tar_all}, {cnt_zh_lang_tar_all / num_instance}, {cnt_zh_active_neurons}\n')
        f.write(f'ja: {per_ja_lang_tar_all}, {cnt_ja_lang_tar_all / num_instance}, {cnt_ja_active_neurons}\n\n')


    fig, ax = plt.subplots(figsize=(30, 10))
    ax.plot(layers, cnts_same_active, marker='o', label='both_act', linewidth=6, markersize=17)
    ax.plot(layers, cnts_same_nonactive, marker='^', label='both_nonact', linewidth=6, markersize=17)
    ax.plot(layers, cnts_only_en_active, marker='s', label='only_en_act', linewidth=6, markersize=17)
    ax.plot(layers, cnts_only_tar_active, marker='*', label='only_tgt_act', linewidth=6, markersize=17)
    ax.set_xlabel('Layer', fontsize=30)
    ax.set_ylabel('Num of neurons', fontsize=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.legend(fontsize=30)
    ax.set_ylim(bottom=0)
    fig.suptitle(f'rc_{model_name_ease}_{lang_dataset}', fontsize=25)
    plt.savefig(f'{save_dir}/active_neurons_{model_name_ease}_{lang_dataset}.png')

if __name__=='__main__':
    main()
