# Please run check_same_pred_inst.py before running this file.
import re
import json
from ftlangdetect import detect
from collections import defaultdict
import argparse


def check_pred_gen(pred, anss):
    if pred in anss:
        return True
    else:
        return False


def tokenize_spacy(tokenizer, sent):
    doc = tokenizer(sent)
    return [tok.text for tok in doc]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_dataset', '-ld', required=True,
                        help='language of dataset', choices=['de', 'es', 'fr', 'ja', 'zh', 'id', 'ko'])
    parser.add_argument('--task', '-t', 
                        required=True, help='task', choices=['mrc', 'ls'])
    parser.add_argument('--num_shot', '-ns', 
                        default='zero', help='num of examples', choices=['zero', 'few'])
    parser.add_argument('--model_name_ease', '-mne', 
                        required=True, help='short model name like qwen2-i', choices=['llama3-i', 'qwen2-i', 'mistraln-i'])

    args = parser.parse_args()
    lang_dataset = args.lang_dataset
    num_shot = args.num_shot
    task = args.task
    model_name_ease = args.model_name_ease

    lang_short2long = {'de':'German', 
                            'es':'Spanish', 
                            'fr':'French', 
                            'id': 'Indonesian', 
                            'ja':'Japanese',
                            'zh':'Chinese',
                            'ko': 'Korean'}
    lang_dataset_long = lang_short2long[lang_dataset]
    print('lang_dataset is', lang_dataset_long)

    low_score = 0.5

    if task == 'mrc':
        results_lang_en = defaultdict(lambda: 0)
        results_lang_tgt = defaultdict(lambda: 0)
        cnt_not_ref_en = 0
        cnt_not_ref_tgt = 0

        with open(f'./../pred_diff/check_same_pred_inst{task}_{lang_dataset}_{num_shot}_{model_name_ease}_all.json', 'r') as f_pred_all:
            data_all = json.load(f_pred_all)
            for i in range(len(data_all)):
                try:
                    passage_text = data_all[i]['passage_text']
                except:
                    passage_text = data_all[i]['context']
                pred_en = data_all[i]['pred_en_clear']
                pred_tgt = data_all[i]['pred_tgt_clear']

                result_lang_en = detect(text=pred_en, low_memory=False)
                result_lang_tgt = detect(text=pred_tgt, low_memory=False)

                if pred_en not in passage_text:
                    cnt_not_ref_en += 1
                    # When the pred_text is not found in the reference text, identify the language of the generated text.
                    # If it is found, determine that the language is the same as that of the dataset.
                    if result_lang_en['score'] < low_score:
                        results_lang_en['low_score'] += 1
                    else:
                        results_lang_en[result_lang_en['lang']] += 1

                if pred_tgt not in passage_text:
                    cnt_not_ref_tgt += 1
                    if result_lang_tgt['score'] <= low_score:
                        results_lang_tgt['low_score'] += 1
                    else:
                        results_lang_tgt[result_lang_tgt['lang']] += 1

        print(f'all : {len(data_all)}')
        print('-----en instruction-----')
        print(f'per of pred not present in ref: {cnt_not_ref_en / len(data_all)}')
        print(f'per of pred not tgt and low: {(cnt_not_ref_en - results_lang_en[lang_dataset] - results_lang_en["low_score"]) / len(data_all)}')
        print(f'distribution of lang : {sorted(dict(results_lang_en).items(), key=lambda x:x[1], reverse=True)}')
        
        print('-----tgt instruction-----')
        print(f'per of pred not present in ref: {cnt_not_ref_tgt / len(data_all)}')
        print(f'per of pred not tgt and low: {(cnt_not_ref_tgt - results_lang_tgt[lang_dataset] - results_lang_tgt["low_score"]) / len(data_all)}')
        print(f'distribution of lang : {sorted(dict(results_lang_tgt).items(), key=lambda x:x[1], reverse=True)}')


    if task == 'ls':
        if lang_dataset == 'ja':
            threshold_long = 7
        else:
            threshold_long = 5
        import spacy
        lang2tokenizer_name = {'ja': 'ja_core_news_sm', 'zh': 'zh_core_web_sm',
                            'de': 'de_core_news_sm', 'es': 'es_core_news_sm', 'fr': 'fr_core_news_sm'}
        tokenizer = spacy.load(lang2tokenizer_name[lang_dataset])
        results_lang_en = defaultdict(lambda: 0)
        results_lang_tgt = defaultdict(lambda: 0)
        cnt_not_word_en = 0
        cnt_not_word_tgt = 0
        cnt_wrong_en = 0
        cnt_wrong_tgt = 0

        with open(f'./../pred_diff/check_same_pred_inst{task}_{lang_dataset}_{num_shot}_{model_name_ease}_all.json', 'r') as f_pred_all:
            data_all = json.load(f_pred_all)
            for i in range(len(data_all)):
                sentence = data_all[i]['sentence']
                word = data_all[i]['word']
                answers = list(data_all[i]['answers'].keys())
                pred_en = data_all[i]['pred_en_clear']
                pred_tgt = data_all[i]['pred_tgt_clear']

                result_lang_en = detect(text=pred_en, low_memory=False)
                result_lang_tgt = detect(text=pred_tgt, low_memory=False)

                pred_en_tokens = tokenize_spacy(tokenizer, pred_en)
                pred_tgt_tokens = tokenize_spacy(tokenizer, pred_tgt)

                if len(pred_en_tokens) > threshold_long:
                    cnt_not_word_en += 1

                if len(pred_tgt_tokens) > threshold_long:
                    cnt_not_word_tgt += 1

                if pred_en not in answers:
                    cnt_wrong_en += 1
                    # When the pred_text is not correct, identify the language of the generated text.
                    # If it is correct, determine that the language is the same as that of the dataset.
                    if result_lang_en['score'] < low_score:
                        results_lang_en['low_score'] += 1
                    else:
                        results_lang_en[result_lang_en['lang']] += 1

                if pred_tgt not in answers:
                    cnt_wrong_tgt += 1
                    if result_lang_tgt['score'] <= low_score:
                        results_lang_tgt['low_score'] += 1
                    else:
                        results_lang_tgt[result_lang_tgt['lang']] += 1

        print(f'all : {len(data_all)}')
        print('-----en instruction-----')
        print(f'num of pred not word: {cnt_not_word_en / len(data_all)}')
        print(f'num of pred not tgt and symbol: {(cnt_wrong_en - results_lang_en[lang_dataset] - results_lang_en["low_score"]) / len(data_all)}')
        print(f'distribution of lang : {sorted(dict(results_lang_en).items(), key=lambda x:x[1], reverse=True)}')
        print('-----tgt instruction-----')
        print(f'num of pred not word: {cnt_not_word_tgt / len(data_all)}')
        print(f'num of pred not tgt and symbol: {(cnt_wrong_tgt - results_lang_tgt[lang_dataset] - results_lang_tgt["low_score"]) / len(data_all)}')
        print(f'distribution of lang : {sorted(dict(results_lang_tgt).items(), key=lambda x:x[1], reverse=True)}')


if __name__=='__main__':
    main()
