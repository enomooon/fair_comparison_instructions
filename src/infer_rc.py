# for review classification
from prompt_template import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import os
import argparse
from sklearn.metrics import f1_score
import random
import datasets

random.seed(42)


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    return model, tokenizer


def load_data_rc(lang_dataset):
    if lang_dataset == 'id':
        dataset_test = datasets.Dataset.from_csv(['./../datasets/Prdect-ID/eno_test.csv'])
        dataset_exam = datasets.Dataset.from_csv(['./../datasets/Prdect-ID/eno_trial.csv'])
    elif lang_dataset == 'ko':
        dataset_test = datasets.Dataset.from_csv(['./../datasets/nsmc/eno_test.csv'])
        dataset_exam = datasets.Dataset.from_csv(['./../datasets/nsmc/eno_train.csv'])
    else:
        dataset_test = load_dataset(f"SetFit/amazon_reviews_multi_{lang_dataset}", split="test")
        dataset_exam = load_dataset(f"SetFit/amazon_reviews_multi_{lang_dataset}", split="train")
    return dataset_test, dataset_exam


def likelihood(model_name, tokenizer, model, batch, label_nl):
    batch = batch[0]
    if 'Mistral-Nemo' in model_name:
        encoded_batch = tokenizer(batch, return_tensors="pt", return_token_type_ids=False).to(model.device)
    else:
        encoded_batch = tokenizer(batch, return_tensors="pt").to(model.device)
    outputs = model(**encoded_batch)
    logits = outputs.logits
    logits = torch.nn.functional.log_softmax(logits, dim=-1)[0]

    # To avoid the influence of token length bias,
    # we judge by the probability of the first token portion of each label.
    if 'Mistral-Nemo' in model_name:
        encoded_bad = tokenizer(label_nl[0], add_special_tokens=False, return_token_type_ids=False)['input_ids']
        encoded_good = tokenizer(label_nl[1], add_special_tokens=False, return_token_type_ids=False)['input_ids']
    else:
        encoded_bad = tokenizer(label_nl[0], add_special_tokens=False)['input_ids']
        encoded_good = tokenizer(label_nl[1], add_special_tokens=False)['input_ids']
    if encoded_bad[0] == encoded_good[0]:
        raise Exception('first tokens are same')
    logit_bad = logits[-1][encoded_bad[0]]
    logit_good = logits[-1][encoded_good[0]]
    return logit_bad, logit_good



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_dataset', '-ld', 
                        required=True, help='language of dataset', choices=['de', 'es', 'fr', 'id', 'ja', 'ko', 'zh'])
    parser.add_argument('--lang_prompt', '-lp',
                        required=True, help='language of instruction', choices=['en', 'tgt', 'tgt-translate'])
    parser.add_argument('--lang_label', '-ll', 
                        required=True, help='language of classification labels', choices=['en', 'tgt'])
    parser.add_argument('--num_shot', '-ns', 
                        default='zero', help='num of examples', choices=['zero', 'few'])
    parser.add_argument('--model_name_ease', '-mne', 
                        required=True, help='short model name like qwen2-i', choices=['llama3-i', 'qwen2-i', 'mistraln-i', 'llama3-b', 'qwen2-b', 'mistraln-b'])
    parser.add_argument('--use_chat_template', '-uct', 
                        help='when using chat_template, please flag this', action='store_true')

    args = parser.parse_args()
    lang_dataset = args.lang_dataset
    lang_prompt = args.lang_prompt
    lang_label = args.lang_label
    num_shot = args.num_shot
    model_name_ease = args.model_name_ease
    use_chat_template = args.use_chat_template
    
    if lang_label == 'en':
        label_good, label_bad = 'good', 'bad'
    else:
        labels_each_lang = {'de':('gut', 'schlecht'), 
                            'es':('bueno', 'malo'), 
                            'fr':('bon', 'mauvais'), 
                            'id':('baik', 'buruk'), 
                            'ja':('良い', '悪い'),
                            'ko':('좋음', '나쁨'),
                            'zh':('好', '差')}
        label_good, label_bad = labels_each_lang[lang_dataset]

    print('lang_dataset is', lang_dataset)
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

    print(model_name)
    model, tokenizer = load_model(model_name)
    dataset, dataset_exam = load_data_rc(lang_dataset)

    cnt_correct_high_prob = 0  # count of correct predictions
    cnt_wrong_high_prob = 0  # count of wrong predictions
    cnt_good = 0  # count of output good
    cnt_bad = 0  # count of output bad

    datas_new = list()
    answers = list()
    preds = list()

    label_nl = [label_bad, label_good]
    print('label_nl is', label_nl)

    # for few-shot examples
    dataset_label_0 = dataset_exam.filter(lambda x: x["label"] == 0)
    dataset_label_1 = dataset_exam.filter(lambda x: x["label"] == 1)
    dataset_label_3 = dataset_exam.filter(lambda x: x["label"] == 3)
    dataset_label_4 = dataset_exam.filter(lambda x: x["label"] == 4)

    prompt_funcs = {
        'en': prompt_rc_en,
        'de': prompt_rc_de,
        'es': prompt_rc_es,
        'fr': prompt_rc_fr,
        'id': prompt_rc_id,
        'ja': prompt_rc_ja,
        'ko': prompt_rc_ko,
        'zh': prompt_rc_zh,
        'de_translate': prompt_rc_de_translate,
        'es_translate': prompt_rc_es_translate,
        'fr_translate': prompt_rc_fr_translate,
        'id_translate': prompt_rc_id_translate,
        'ja_translate': prompt_rc_ja_translate,
        'ko_translate': prompt_rc_ko_translate,
        'zh_translate': prompt_rc_zh_translate,
        'en_exam': prompt_rc_en_with_example,
        'de_exam': prompt_rc_de_with_example,
        'es_exam': prompt_rc_es_with_example,
        'fr_exam': prompt_rc_fr_with_example,
        'id_exam': prompt_rc_id_with_example,
        'ja_exam': prompt_rc_ja_with_example,
        'ko_exam': prompt_rc_ko_with_example,
        'zh_exam': prompt_rc_zh_with_example,
        'de_translate_exam': prompt_rc_de_translate_with_example,
        'es_translate_exam': prompt_rc_es_translate_with_example,
        'fr_translate_exam': prompt_rc_fr_translate_with_example,
        'id_translate_exam': prompt_rc_id_translate_with_example,
        'ja_translate_exam': prompt_rc_ja_translate_with_example,
        'ko_translate_exam': prompt_rc_ko_translate_with_example,
        'zh_translate_exam': prompt_rc_zh_translate_with_example
    }
    
    example_funcs = {
        'en': example_rc_en, 
        'de': example_rc_de, 
        'es': example_rc_es, 
        'fr': example_rc_fr, 
        'id': example_rc_id, 
        'ja': example_rc_ja, 
        'ko': example_rc_ko, 
        'zh': example_rc_zh,
        'de_translate': example_rc_de_translate, 
        'es_translate': example_rc_es_translate, 
        'fr_translate': example_rc_fr_translate, 
        'id_translate': example_rc_id_translate, 
        'ja_translate': example_rc_ja_translate, 
        'ko_translate': example_rc_ko_translate, 
        'zh_translate': example_rc_zh_translate, 
    }

    for line in tqdm(dataset):
        sentence = line['text'].strip()
        label_5step = line['label']
        if label_5step in [0, 1]:
            label_2step = label_nl[0]
        elif label_5step in [3, 4]:
            label_2step = label_nl[1]
        else:
            continue

        if lang_prompt == 'en':
            if num_shot == 'zero':
                prompt = prompt_funcs[lang_prompt](sentence, label_nl)
            elif num_shot == 'few':
                # randomly select examples
                exam0 = example_funcs[lang_prompt](dataset_label_0[random.randrange(
                    len(dataset_label_0))]['text'].strip(), label_nl[0])
                exam1 = example_funcs[lang_prompt](dataset_label_1[random.randrange(
                    len(dataset_label_1))]['text'].strip(), label_nl[0])
                exam3 = example_funcs[lang_prompt](dataset_label_3[random.randrange(
                    len(dataset_label_3))]['text'].strip(), label_nl[1])
                exam4 = example_funcs[lang_prompt](dataset_label_4[random.randrange(
                    len(dataset_label_4))]['text'].strip(), label_nl[1])
                exams = '\n'.join(random.sample([exam0, exam1, exam3, exam4], 4))
                prompt = prompt_funcs[f'{lang_prompt}_exam'](sentence, label_nl, exams)

        elif lang_prompt == 'tgt':
            if num_shot == 'zero':
                prompt = prompt_funcs[lang_dataset](sentence, label_nl)
            elif num_shot == 'few':
                exam0 = example_funcs[lang_dataset](dataset_label_0[random.randrange(
                    len(dataset_label_0))]['text'].strip(), label_nl[0])
                exam1 = example_funcs[lang_dataset](dataset_label_1[random.randrange(
                    len(dataset_label_1))]['text'].strip(), label_nl[0])
                exam3 = example_funcs[lang_dataset](dataset_label_3[random.randrange(
                    len(dataset_label_3))]['text'].strip(), label_nl[1])
                exam4 = example_funcs[lang_dataset](dataset_label_4[random.randrange(
                    len(dataset_label_4))]['text'].strip(), label_nl[1])
                exams = '\n'.join(random.sample([exam0, exam1, exam3, exam4], 4))
                prompt = prompt_funcs[f'{lang_dataset}_exam'](sentence, label_nl, exams)

        else:
            if num_shot == 'zero':
                prompt = prompt_funcs[f'{lang_dataset}_translate'](sentence, label_nl)
            elif num_shot == 'few':
                exam0 = example_funcs[f'{lang_dataset}_translate'](dataset_label_0[random.randrange(
                    len(dataset_label_0))]['text'].strip(), label_nl[0])
                exam1 = example_funcs[f'{lang_dataset}_translate'](dataset_label_1[random.randrange(
                    len(dataset_label_1))]['text'].strip(), label_nl[0])
                exam3 = example_funcs[f'{lang_dataset}_translate'](dataset_label_3[random.randrange(
                    len(dataset_label_3))]['text'].strip(), label_nl[1])
                exam4 = example_funcs[f'{lang_dataset}_translate'](dataset_label_4[random.randrange(
                    len(dataset_label_4))]['text'].strip(), label_nl[1])
                exams = '\n'.join(random.sample([exam0, exam1, exam3, exam4], 4))
                prompt = prompt_funcs[f'{lang_dataset}_translate_exam'](sentence, label_nl, exams)


        prompt = prompt.replace('</s>', tokenizer.eos_token)
        if use_chat_template:
            messages = [{'role':'user', 'content':prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(prompt)

        prob_bad, prob_good = likelihood(model_name, tokenizer, model, [f'{prompt}'], label_nl)
        # print(label_nl[1], label_nl[0])
        # print(prob_good, prob_bad)

        if prob_good > prob_bad:
            line['high_prob'] = label_nl[1]
            cnt_good += 1
        else:
            line['high_prob'] = label_nl[0]
            cnt_bad += 1
        print('high_prob', line['high_prob'])

        line['label_2step'] = label_2step

        if line['high_prob'] == label_2step:
            cnt_correct_high_prob += 1
        else:
            cnt_wrong_high_prob += 1

        datas_new.append(line)
        answers.append(line['label_2step'])
        preds.append(line['high_prob'])

    f1 = f1_score(answers, preds, average='macro')

    print('cnt_correct_high_prob', cnt_correct_high_prob)
    print('cnt_wrong_high_prob', cnt_wrong_high_prob)
    print('f1', f1)

    pred_dir = f'./../pred/rc/{lang_dataset}/{num_shot}'
    result_dir = f'./../result/rc/{lang_dataset}/{num_shot}'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(f'{pred_dir}/{model_name_ease}_{lang_prompt}_{label_nl[0]}-{label_nl[1]}.json', 'w') as f_pred:
        json.dump(datas_new, f_pred, ensure_ascii=False)

    with open(f'{result_dir}/{model_name_ease}_{lang_prompt}_{label_nl[0]}-{label_nl[1]}.txt', 'w') as f_result:
        f_result.write('cnt_correct_high_prob: ' +
                       str(cnt_correct_high_prob) + '\n')
        f_result.write('cnt_wrong_high_prob: ' +
                       str(cnt_wrong_high_prob) + '\n')
        f_result.write('f1_score: ' + str(f1) + '\n')
        f_result.write('cnt_good: ' + str(cnt_good) + '\n')
        f_result.write('cnt_bad: ' + str(cnt_bad) + '\n')


if __name__ == '__main__':
    main()
