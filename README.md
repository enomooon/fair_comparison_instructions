# fair_comparison_instructions
Code for "[A Fair Comparison without Translationese: English vs. Target-language Instructions for Multilingual LLMs](https://aclanthology.org/2025.naacl-short.55/)" and "[多言語大規模言語モデルにおける英語指示文と対象言語指示文の公平な比較](https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/P7-12.pdf)"

In addition to the content of the former paper, the latter Japanese paper analyzes the relationship between instruction languages and activate neurons in LLMs.

## Environment

python version: 3.11.8

```
pip install -r requirements.txt
```

## Download and pre-process datasets

```
cd datasets
sh download_ls.sh
sh download_mrc.sh
sh download_rc.sh
```
For French MRC, you need to request access to FQuAD through this [application form](https://fquad.illuin.tech/).
After download it, please create FQuAD folder and place train.json and valid.json under FQuAD folder like ./datasets/FQuAD/valid.json

## Fair Instruction Construction (Section 3.1)
The instructions for each language created in this study are available in `prompt_template.py`.
If you would like to create additional instructions, you can do so using the method below and incorporating revisions by native speakers as described in Step 4.

#### Example for Step 2 in Section 3.1
```
python make_instruction_gpt4.py 
    --step step2 
    --lang ja
    --task_step2 MRC
```

#### Example for Step 3 in Section 3.1
Please specify the output from Step 2 for `--inst_en_step3` and `--inst_tgt_step3` 
```
python make_instruction_gpt4.py 
    --step step3 
    --lang ja 
    --inst_en_step3 'I will provide a sentence and a word included in the sentence. \nPlease generate a simpler Japanese synonym for the word. \nGenerate nothing but the synonym.' 
    --inst_tgt_step3 '私は文とその中に含まれる単語を提供します。 \n提供された単語に対して、より簡単な日本語の同義語を一つ生成してください。 \n同義語以外は何も生成しないでください。' 
```

## Inference (Section 4)

#### Example for Lexical Simplification Tasks
```
cd src
python infer_ls.py 
    --lang_dataset ja 
    --lang_prompt tgt 
    --model_name_ease qwen2-b
```

#### Example for Machine Reading Comprehension Tasks
```
cd src
python infer_mrc.py 
    --lang_dataset ja 
    --lang_prompt tgt 
    --model_name_ease qwen2-b
```

#### Example for Review Classification Tasks
```
cd src
python infer_mrc.py 
    --lang_dataset ja 
    --lang_prompt tgt 
    --lang_label en
    --model_name_ease qwen2-b
```


## Anlysis of Model Outputs (Section 5)

#### Agreement of Pred Between Instruction Languages (Section 5.1)
Count the relationship between predictions under English and target-language instructions (both correct, only correct with English instructions, only correct with target-language instructions, both incorrect).
```
cd src
check_same_pred_inst.py
    --lang_dataset ja
    --task mrc
    --model_name_ease qwen2-i
```

#### Detecting Pred Language and Measuring Instruction-following Ability (Section 5.1 and 5.2)
```
cd src
python detect_lang_and_follow.py
    --lang_dataset ja
    --task mrc
    --model_name_ease qwen2-i
```

## Anlysis of Model Neurons

#### Detecting of Language Specific Neurons
We identify language-specific neurons using LAPE ([Tang et al., 2024](https://aclanthology.org/2024.acl-long.309/)).
For this purpose, we prepare the language-specific text files required by LAPE using this [dataset](https://github.com/kojima-takeshi188/lang_neuron/tree/main/assets/Language/sense).
Please place the dataset under the datasets folder like `datasets/neuron/assets/Language/sense/ja.json`.

```
cd src
python neuron_activation.py
    --lang_dataset ja
    --model_name_ease qwen2-i

python neuron_identify.py
    --model_name_ease qwen2-i
```

#### Relationship Between Instruction Languages and Active Neurons

For each layer, the degree of overlap in activated neurons is visualized when processing the same instance with English and target-language instructions.
The percentage of language-specific neurons (per language) that are activated is also calculated for each instruction type.
```
cd src
python neuron_instruction_analysis.py
    --lang_dataset ja
    --model_name_ease qwen2-i
```
