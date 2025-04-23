import csv
import json
from sklearn.model_selection import train_test_split

folder_path = './Chinese-LS/dataset/'
file_path = f'{folder_path}annotation_data.csv'

with open(file_path) as f:
    reader = csv.reader(f, delimiter='\t')
    conts = [row for row in reader]

    

conts_valid, conts_test = train_test_split(conts, random_state=42, test_size=0.9)
conts_valid_new, conts_test_new = list(), list()

for cont in conts_valid:
    sentence = cont[0].strip()
    word = cont[1].strip()
    answers = cont[4].strip()

    cont_new = {'context':sentence, 'target': word, 'substitutions': answers}
    conts_valid_new.append(cont_new)


for cont in conts_test:
    sentence = cont[0].strip()
    word = cont[1].strip()
    answers = cont[4].strip()

    cont_new = {'context':sentence, 'target': word, 'substitutions': answers}
    conts_test_new.append(cont_new)

with open(f'{folder_path}eno_valid.json', 'w') as f_valid, \
     open(f'{folder_path}eno_test.json', 'w') as f_test:
    json.dump(conts_valid_new, f_valid, ensure_ascii=False)
    json.dump(conts_test_new, f_test, ensure_ascii=False)
    