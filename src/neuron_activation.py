# Before executing this code, please place the data from Kojima et al. (https://github.com/kojima-takeshi188/lang_neuron/tree/main/assets/Language/sense) under ./../datasets/ 
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from baukit import TraceDict
import json
from tqdm import tqdm
import argparse
import os

def get_out_llm(model, prompt, num_layers): 

    model.eval()
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(num_layers)]
    
    with torch.no_grad():
        with TraceDict(model, MLP_act) as ret:
            output = model(prompt, output_hidden_states = True)
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]
        return MLP_act_value
        
def get_over_zero_vec(mlp_act):
    mlp_act[mlp_act > 0]
    binary_act = (mlp_act > 0).astype(int)

    binary_act = torch.tensor(binary_act)
    return binary_act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_dataset', '-ld', 
                        required=True, help='language of dataset', choices=['en', 'de', 'es', 'fr', 'ja', 'zh'])
    parser.add_argument('--model_name_ease', '-mne', 
                        required=True, help='short model name like qwen2-i', choices=['llama3-i', 'qwen2-i', 'mistraln-i', 'llama3-b', 'qwen2-b', 'mistraln-b'])

    args = parser.parse_args()
    model_name_ease = args.model_name_ease
    lang = args.lang_dataset

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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    num_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32)
    all_token_num = 0

    with open(f'./../datasets/neuron/assets/Language/sense/{lang}.json') as f:
        datas = json.load(f)
        datas = datas['sentences']['positive']

    for line in tqdm(datas):
        input_ids = tokenizer(line, return_tensors="pt")['input_ids'].to(model.device)
        output = get_out_llm(model, input_ids, num_layers)
        prompt_length = len(output[0][0])
        all_token_num += prompt_length
        for layer in range(num_layers):
            output_now = output[layer].to('cpu').detach().numpy().copy()
            for i in range(prompt_length):
                binary_act = get_over_zero_vec(output_now[0][i])
                over_zero[layer] += binary_act


    print('all_token_num', all_token_num)
    print('over_zero', over_zero)

    output = dict(n=all_token_num, over_zero=over_zero.to('cpu'))

    save_dir = f'./../neuron/activation'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(output, f'./../neuron/activation/activation.{lang}.train.{model_name_ease}')

if __name__=='__main__':
    main()
