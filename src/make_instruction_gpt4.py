import openai
import argparse
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

def gpt_generation(
    prompt: str,
    api_key: str = "OPENAI_API_KEY",
    model_name: str = "gpt-4o-2024-05-13",
    max_tokens: int = 500,
    temperature: float = 0.6,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0
):

    openai.api_key = api_key
    masseges = [{"role": "user", "content": prompt}]

    res = openai.ChatCompletion.create(
        model=model_name,
        messages=masseges,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    return res


def step2_generate_inst(task, lang):
    if lang == 'English':
        lang1 = 'Chinese'
        print(f'lang1 is {lang1}')
    else:
        lang1 = lang

    if task == 'LS':
        prompt = f'''
I will perform a lexical simplification task using an LLM. I want to use {lang} for instructions to the LLM.
Please create instructions based on each of the following requirements.
Treat the requirements as guidelines rather than as text to be translated.

- Please inform the LLM that I will provide a sentence and a word included in the sentence.
- Please instruct the LLM to generate one simpler {lang1} synonym for the word.
- Please instruct the LLM to generate nothing but the synonym.
'''

    elif task == 'MRC':
        prompt = f'''
I will perform a question answering task using an LLM. I want to use {lang} for instructions to the LLM.
Please create instructions based on each of the following requirements.
Treat the requirements as guidelines rather than as text to be translated.

- Please inform the LLM that I will provide a question and a reference sentence.
- Please instruct the LLM to extract the answer to the question from the reference sentence.
- Please instruct the LLM to generate nothing but the answer.
'''

    elif task == 'RC':
        prompt = f'''
I will perform a customer review classification task using an LLM. I want to use {lang} for instructions to the LLM.
Please create instructions based on each of the following requirements.
Treat the requirements as guidelines rather than as text to be translated.

- Please inform the LLM that I will provide a review.
- Please request the LLM to rate a given review based on the following criteria.
- Please inform the LLM that the criterion is to choose "good" if the review indicates a high evaluation and "bad" if it indicates a low evaluation.
'''

    print(prompt)

    res = gpt_generation(prompt=prompt, api_key=OPENAI_API_KEY)
    print(res['choices'][0]['message']['content'])



def step3_verify_diff(lang, inst_en, inst_tgt):
    '''
    Verifying whether the English and target-language instructions convey the same content with GPT-4. 
    '''
    prompt = f'''
Determine whether the following two instructions, one in English and one in {lang}, convey exactly the same content.

If there is any difference in content, such as meaning, intent, or nuance, respond only with “different”.  
Only if the two instructions convey exactly the same content, respond only with “same”.

English
{inst_en}

{lang}
{inst_tgt}
'''

    print(prompt)

    res = gpt_generation(prompt=prompt, api_key=OPENAI_API_KEY)
    print(res['choices'][0]['message']['content'])


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--step', '-s', required=True, help='which step',
                        choices=['step2', 'step3'])
    parser.add_argument('--lang', '-l', required=True, help='(For step2)language you want to generate. (For step3)language to compare with EN-inst',
                        choices=['en', 'de', 'es', 'fr', 'id', 'ja', 'ko', 'zh'])
    parser.add_argument('--task_step2', '-t', help='(For step2) which task', choices=['LS', 'MRC', 'RC'], default='LS')
    parser.add_argument('--inst_en_step3', '-ie', help='(For step3) English instructions with \n')
    parser.add_argument('--inst_tgt_step3', '-it', help='(For step3) Target-language instructions with \n')

    args = parser.parse_args()
    step = args.step
    lang = args.lang
    task = args.task_step2
    inst_en = args.inst_en_step3
    inst_tgt = args.inst_tgt_step3

    lang_short2long = {'en':'English',
                        'de':'German', 
                        'es':'Spanish', 
                        'fr':'French', 
                        'id':'Indonesian',
                        'ja':'Japanese',
                        'ko':'Korean',
                        'zh':'Chinese'}
    lang_long = lang_short2long[lang]

    if step == 'step2':
        step2_generate_inst(task, lang_long)
    elif step == 'step3':
        step3_verify_diff(lang_long, inst_en, inst_tgt)
