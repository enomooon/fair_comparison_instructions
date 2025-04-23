import json
import argparse
import os

def check_pred_rc(pred, ans):
    if pred == ans:
        return True
    else:
        return False


def check_pred_ls_mrc(pred, anss):
    if pred in anss:
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_dataset', '-ld', 
                        required=True, help='language of dataset', choices=['de', 'es', 'fr', 'ja', 'zh', 'id', 'ko'])
    parser.add_argument('--lang_label_rc', '-ll', 
                        default='en', help='(for rc)language of classification labels', choices=['en', 'tgt'])
    parser.add_argument('--task', '-t', 
                        required=True, help='task', choices=['mrc', 'ls', 'rc'])
    parser.add_argument('--num_shot', '-ns', 
                        default='zero', help='num of examples', choices=['zero', 'few'])
    parser.add_argument('--model_name_ease', '-mne', 
                        required=True, help='short model name like qwen2-i', choices=['llama3-i', 'qwen2-i', 'mistraln-i'])


    args = parser.parse_args()
    lang_dataset = args.lang_dataset
    lang_label = args.lang_label_rc
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
    label_nl = [label_bad, label_good]

    output_list = list()

    if task == 'rc':
        path_folder = f'./../pred/{task}/{lang_dataset}/{num_shot}'
        file_en = f'{path_folder}/{model_name_ease}_en_{label_nl[0]}-{label_nl[1]}.json'
        file_tgt = f'{path_folder}/{model_name_ease}_tgt_{lang_dataset}_{label_nl[0]}-{label_nl[1]}.json'
        with open(file_en) as fe, \
                open(file_tgt) as ft:
            data_en = json.load(fe)
            data_tgt = json.load(ft)

        data_cor_en = list()
        data_cor_tgt = list()
        data_cor_both = list()
        data_cor_neit = list()

        for i in range(len(data_en)):
            if data_en[i]['text'] != data_tgt[i]['text']:
                raise Exception('INSTANCES ARE NOT MATCH')
            is_cor_en = check_pred_rc(data_en[i]['high_prob'], data_en[i]['label_2step'])
            is_cor_tgt = check_pred_rc(data_tgt[i]['high_prob'], data_tgt[i]['label_2step'])

            data_en[i]['high_prob_tgt'] = data_tgt[i]['high_prob']
            if is_cor_en and is_cor_tgt:
                data_cor_both.append(data_en[i])
            elif not (is_cor_en or is_cor_tgt):
                data_cor_neit.append(data_en[i])
            elif is_cor_en:
                data_cor_en.append(data_en[i])
            elif is_cor_tgt:
                data_cor_tgt.append(data_en[i])

        print(f'model is {model_name_ease}')
        print(f'correct en only {len(data_cor_en)}')
        print(f'correct tgt only {len(data_cor_tgt)}')
        print(f'correct both {len(data_cor_both)}')
        print(f'correct neither {len(data_cor_neit)}')

    elif task == 'mrc':
        cnt_same_ans = 0
        lang_dataset_long = lang_dataset_long.lower()
        path_folder = f'./../pred/{task}/{lang_dataset}/{num_shot}'
        file_en = f'{path_folder}/{model_name_ease}_en.json'
        file_tgt = f'{path_folder}/{model_name_ease}_tgt.json'
        with open(file_en) as fe, \
             open(file_tgt) as ft:
            data_en = json.load(fe)
            data_tgt = json.load(ft)

        data_cor_en = list()
        data_cor_tgt = list()
        data_cor_both = list()
        data_cor_neit = list()
        data_all = list()

        for i in range(len(data_en)):
            if 'question_text' in data_en[i].keys():
                if data_en[i]['question_text'] != data_tgt[i]['question_text']:
                    raise Exception('INSTANCES ARE NOT MATCH')
            elif 'question' in data_en[i].keys():
                if data_en[i]['question'] != data_tgt[i]['question']:
                    raise Exception('INSTANCES ARE NOT MATCH')

            pred_en_clear = data_en[i]['pred_normalize']
            pred_tgt_clear = data_tgt[i]['pred_normalize']
            if pred_en_clear == pred_tgt_clear:
                cnt_same_ans += 1

            if lang_dataset in ['zh', 'fr']:
                is_cor_en = check_pred_ls_mrc(pred_en_clear, data_en[i]['answers'])
                is_cor_tgt = check_pred_ls_mrc(pred_tgt_clear, data_tgt[i]['answers'])
            else:
                is_cor_en = check_pred_ls_mrc(pred_en_clear, data_en[i]['answers']['text'])
                is_cor_tgt = check_pred_ls_mrc(pred_tgt_clear, data_tgt[i]['answers']['text'])

            data_en[i]['pred_en_clear'] = pred_en_clear
            data_en[i]['pred_tgt_clear'] = pred_tgt_clear

            data_all.append(data_en[i])
            if is_cor_en and is_cor_tgt:
                data_cor_both.append(data_en[i])
            elif not (is_cor_en or is_cor_tgt):
                data_cor_neit.append(data_en[i])
            elif is_cor_en:
                data_cor_en.append(data_en[i])
            elif is_cor_tgt:
                data_cor_tgt.append(data_en[i])

        print(f'model is {model_name_ease}')
        print(f'correct en only {len(data_cor_en)}')
        print(f'correct tgt only {len(data_cor_tgt)}')
        print(f'correct both {len(data_cor_both)}')
        print(f'correct correct neither {len(data_cor_neit)}')
        print(f'percentage of same ans {cnt_same_ans / len(data_en)}')
        print(cnt_same_ans / len(data_en))
        output_list.append(f'{(cnt_same_ans / len(data_en))*100:.2f}')

        if not os.path.exists('./../pred_diff'):
            os.makedirs('./../pred_diff')
        with open(f'./../pred_diff/check_same_pred_inst{task}_{lang_dataset}_{num_shot}_{model_name_ease}_all.json', 'w') as f_pred_all:
            json.dump(data_all, f_pred_all, ensure_ascii=False)

    elif task == 'ls':
        cnt_same_ans = 0
        path_folder = f'./../pred/{task}/{lang_dataset}/{num_shot}'
        file_en = f'{path_folder}/{model_name_ease}_en.json'
        file_tgt = f'{path_folder}/{model_name_ease}_tgt.json'
        with open(file_en) as fe, \
             open(file_tgt) as ft:
            data_en = json.load(fe)
            data_tgt = json.load(ft)

        data_cor_en = list()
        data_cor_tgt = list()
        data_cor_both = list()
        data_cor_neit = list()
        data_all = list()

        for i in range(len(data_en)):
            if data_en[i]['sentence'] != data_tgt[i]['sentence'] or data_en[i]['word'] != data_tgt[i]['word']:
                raise Exception('INSTANCES ARE NOT MATCH')

            pred_en_clear = data_en[i]['pred_normalize']
            pred_tgt_clear = data_tgt[i]['pred_normalize']
            if pred_en_clear == pred_tgt_clear:
                cnt_same_ans += 1

            is_cor_en = check_pred_ls_mrc(pred_en_clear, list(data_en[i]['answers'].keys()))
            is_cor_tgt = check_pred_ls_mrc(pred_tgt_clear, list(data_tgt[i]['answers'].keys()))

            data_en[i]['pred_en_clear'] = pred_en_clear
            data_en[i]['pred_tgt_clear'] = pred_tgt_clear

            data_all.append(data_en[i])
            if is_cor_en and is_cor_tgt:
                data_cor_both.append(data_en[i])
            elif not (is_cor_en or is_cor_tgt):
                data_cor_neit.append(data_en[i])
            elif is_cor_en:
                data_cor_en.append(data_en[i])
            elif is_cor_tgt:
                data_cor_tgt.append(data_en[i])

        print(f'model is {model_name_ease}')
        print(f'correct en only {len(data_cor_en)}')
        print(f'correct tgt only {len(data_cor_tgt)}')
        print(f'correct both {len(data_cor_both)}')
        print(f'correct neither {len(data_cor_neit)}')
        print(f'percentage of same ans {cnt_same_ans / len(data_en)}')
        output_list.append(f'{(cnt_same_ans / len(data_en))*100:.2f}')

        if not os.path.exists('./../pred_diff'):
            os.makedirs('./../pred_diff')
        with open(f'./../pred_diff/check_same_pred_inst{task}_{lang_dataset}_{num_shot}_{model_name_ease}_all.json', 'w') as f_pred_all:
            json.dump(data_all, f_pred_all, ensure_ascii=False)


if __name__=='__main__':
    main()
