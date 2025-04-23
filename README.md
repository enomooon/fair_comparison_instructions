# fair_comparison_instructions
Code for "A Fair Comparison without Translationese: English vs. Target-language Instructions for Multilingual LLMs" and "[多言語大規模言語モデルにおける英語指示文と対象言語指示文の公平な比較](https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/P7-12.pdf)"

In addition to the content of the former paper, the latter Japanese paper analyzes the relationship between the instruction language and the activated neurons.

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
sh download_ls.sh
```
For French MRC, you need to request access to FQuAD through this [application form](https://fquad.illuin.tech/).
After download it, please create FQuAD folder and place train.json and valid.json under FQuAD folder like ./datasets/FQuAD/valid.json

## Fair Instruction Construction (Section 3.1)
The instructions for each language created in this study are available in `prompt_template.py`.
If you would like to create additional instructions, you can do so using the method described below.

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
    --inst_en_step3 'I will provide a review.\nPlease rate the given review based on the following criteria.\nChoose "1" if the review indicates a high evaluation and "0" if it indicates a low evaluation.' 
    --inst_tgt_step3 '私はこれからレビューを提供します。\nレビューを以下の基準に基づいて評価してください。\n高い評価を示すレビューの場合は「1」を、低い評価を示すレビューの場合は「0」を選んでください。'
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
For this purpose, we use the [dataset](https://github.com/kojima-takeshi188/lang_neuron/tree/main/assets/Language/sense) created by [Kojima et al.](https://aclanthology.org/2024.naacl-long.384/).
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
