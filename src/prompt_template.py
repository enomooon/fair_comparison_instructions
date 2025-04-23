# --------------------------------------------RC---------------------------------------------------
def prompt_rc_en(sentence, label_nl):
    prompt = f'''
I will provide a review.
Please rate the given review based on the following criteria.
Choose `{label_nl[1]}` if the review indicates a high evaluation and `{label_nl[0]}` if it indicates a low evaluation.

Review: {sentence}

Rating: '''
    return prompt

def prompt_rc_en_with_example(sentence, label_nl, examples):
    prompt = f'''
I will provide a review.
Please rate the given review based on the following criteria.
Choose `{label_nl[1]}` if the review indicates a high evaluation and `{label_nl[0]}` if it indicates a low evaluation.
{examples}

Review: {sentence}

Rating: '''
    return prompt

def example_rc_en(sentence, label):
    example = f'''
Review: {sentence}

Rating: {label}</s>'''
    return example


def prompt_rc_ja(sentence, label_nl):
    prompt = f'''
これからレビューの文を与えます。
そのレビューを以下の基準に基づいて評価してください。
そのレビューが高い評価を示す場合は`{label_nl[1]}`を、低い評価を示す場合は`{label_nl[0]}`を選んでください。

レビュー: {sentence}

評価: '''
    return prompt


def prompt_rc_ja_with_example(sentence, label_nl, examples):
    prompt = f'''
これからレビューの文を与えます。
そのレビューを以下の基準に基づいて評価してください。
そのレビューが高い評価を示す場合は`{label_nl[1]}`を、低い評価を示す場合は`{label_nl[0]}`を選んでください。
{examples}

レビュー: {sentence}

評価: '''
    return prompt

def example_rc_ja(sentence, label):
    example = f'''
レビュー: {sentence}

評価: {label}</s>'''
    return example


def prompt_rc_zh(sentence, label_nl):
    prompt = f'''
我将提供一条评论。
请根据以下标准对给定的评论进行评分。
如果评论表示高度评价，请选择`{label_nl[1]}`；如果评论表示不好的评价，请选择`{label_nl[0]}`。

评论: {sentence}

评分: '''
    return prompt

def prompt_rc_zh_with_example(sentence, label_nl, examples):
    prompt = f'''
我将提供一条评论。
请根据以下标准对给定的评论进行评分。
如果评论表示高度评价，请选择`{label_nl[1]}`；如果评论表示不好的评价，请选择`{label_nl[0]}`。
{examples}

评论: {sentence}

评分: '''
    return prompt

def example_rc_zh(sentence, label):
    example = f'''
评论: {sentence}

评分: {label}</s>'''
    return example


def prompt_rc_id(sentence, label_nl):
    prompt = f'''
Saya akan memberikan sebuah ulasan.
Tolong nilai ulasan yang diberikan berdasarkan kriteria berikut.
Pilih `{label_nl[1]}` jika ulasan menunjukkan evaluasi tinggi dan `{label_nl[0]}` jika menunjukkan evaluasi rendah.

Ulasan: {sentence}

Nilai: '''
    return prompt

def prompt_rc_id_with_example(sentence, label_nl, examples):
    prompt = f'''
Saya akan memberikan sebuah ulasan.
Tolong nilai ulasan yang diberikan berdasarkan kriteria berikut.
Pilih `{label_nl[1]}` jika ulasan menunjukkan evaluasi tinggi dan `{label_nl[0]}` jika menunjukkan evaluasi rendah.
{examples}

Ulasan: {sentence}

Nilai: '''
    return prompt

def example_rc_id(sentence, label):
    example = f'''
Ulasan: {sentence}

Nilai: {label}</s>'''
    return example



def prompt_rc_ko(sentence, label_nl):
    prompt = f'''
지금부터 리뷰를 입력합니다.
주어진 리뷰를 다음 기준에 따라 평가해 주세요.
리뷰가 높은 평가를 나타내는 경우 `{label_nl[1]}`을, 낮은 평가를 나타내는 경우 `{label_nl[0]}`을 선택해 주세요.

리뷰: {sentence}

평가: '''
    return prompt


def prompt_rc_ko_with_example(sentence, label_nl, examples):
    prompt = f'''
지금부터 리뷰를 입력합니다.
주어진 리뷰를 다음 기준에 따라 평가해 주세요.
리뷰가 높은 평가를 나타내는 경우 `{label_nl[1]}`을, 낮은 평가를 나타내는 경우 `{label_nl[0]}`을 선택해 주세요.
{examples}

리뷰: {sentence}

평가: '''
    return prompt

def example_rc_ko(sentence, label):
    example = f'''
리뷰: {sentence}

평가: {label}</s>'''
    return example


def prompt_rc_de(sentence, label_nl):
    prompt = f'''
Ich gebe Ihnen eine Rezension.
Bitte bewerten Sie die Rezension anhand der folgenden Kriterien.
Wählen Sie `{label_nl[1]}`, wenn die Rezension eine positive Bewertung darstellt, und `{label_nl[0]}`, wenn sie eine negative Bewertung darstellt.

Rezension: {sentence}

Bewertung: '''
    return prompt

def prompt_rc_de_with_example(sentence, label_nl, examples):
    prompt = f'''
Ich gebe Ihnen eine Rezension.
Bitte bewerten Sie die Rezension anhand der folgenden Kriterien.
Wählen Sie `{label_nl[1]}`, wenn die Rezension eine positive Bewertung darstellt, und `{label_nl[0]}`, wenn sie eine negative Bewertung darstellt.
{examples}

Rezension: {sentence}

Bewertung: '''
    return prompt

def example_rc_de(sentence, label):
    example = f'''
Rezension: {sentence}

Bewertung: {label}</s>'''
    return example



def prompt_rc_es(sentence, label_nl):
    prompt = f'''
Voy a proporcionarte una reseña.
Por favor, califícala proporcionadamente según los siguientes criterios.
Elige `{label_nl[1]}` si la reseña muestra una alta valoracion y `{label_nl[0]}` si es una baja valoración.

Reseña: {sentence}

Calificación: '''
    return prompt

def prompt_rc_es_with_example(sentence, label_nl, examples):
    prompt = f'''
Voy a proporcionarte una reseña.
Por favor, califícala proporcionadamente según los siguientes criterios.
Elige `{label_nl[1]}` si la reseña muestra una alta valoracion y `{label_nl[0]}` si es una baja valoración.
{examples}

Reseña: {sentence}

Calificación: '''
    return prompt

def example_rc_es(sentence, label):
    example = f'''
Reseña: {sentence}

Calificación: {label}</s>'''
    return example


def prompt_rc_fr(sentence, label_nl):
    prompt = f'''
Je vais fournir une critique.
Merci d'évaluer la critique en fonction des critères suivants.
Choisissez `{label_nl[1]}` si la critique est positive et `{label_nl[0]}` si elle est négative.

Critique: {sentence}

Évaluation: '''
    return prompt

def prompt_rc_fr_with_example(sentence, label_nl, examples):
    prompt = f'''
Je vais vous donner une critique.
Merci d'évaluer la critique en fonction des critères suivants.
Choisissez `{label_nl[1]}` si la critique est positive et `{label_nl[0]}` si elle est négative.
{examples}

Critique: {sentence}

Évaluation: '''
    return prompt

def example_rc_fr(sentence, label):
    example = f'''
Critique: {sentence}

Notation: {label}</s>'''
    return example




def prompt_rc_ja_translate(sentence, label_nl):
    prompt = f'''
レビューをさせていただきます。
以下の基準に基づいてレビューを評価してください。
レビューで高い評価が示された場合は`{label_nl[1]}`を選択し、低い評価が示された場合は`{label_nl[0]}`を選択します。

レビュー: {sentence}

評価: '''
    return prompt


def prompt_rc_ja_translate_with_example(sentence, label_nl, examples):
    prompt = f'''
レビューをさせていただきます。
以下の基準に基づいてレビューを評価してください。
レビューで高い評価が示された場合は`{label_nl[1]}`を選択し、低い評価が示された場合は`{label_nl[0]}`を選択します。
{examples}

レビュー: {sentence}

評価: '''
    return prompt

def example_rc_ja_translate(sentence, label):
    example = f'''
レビュー: {sentence}

評価: {label}</s>'''
    return example



def prompt_rc_zh_translate(sentence, label_nl):
    prompt = f'''
我会提供一个评论。
请根据以下标准对给定的评论进行评分。
如果评论表示评价高，则选择`{label_nl[1]}`，如果评价低，则选择`{label_nl[0]}`。

评论: {sentence}

评分: '''
    return prompt


def prompt_rc_zh_translate_with_example(sentence, label_nl, examples):
    prompt = f'''
我会提供一个评论。
请根据以下标准对给定的评论进行评分。
如果评论表示评价高，则选择`{label_nl[1]}`，如果评价低，则选择`{label_nl[0]}`。
{examples}

评论: {sentence}

评分: '''
    return prompt

def example_rc_zh_translate(sentence, label):
    example = f'''
评论: {sentence}

评分: {label}</s>'''
    return example



def prompt_rc_id_translate(sentence, label_nl):
    prompt = f'''
Saya akan memberikan ulasan.
Silakan menilai ulasan yang diberikan berdasarkan kriteria berikut.
Pilih `{label_nl[1]}` jika ulasan menunjukkan penilaian tinggi dan `{label_nl[0]}` jika ulasan menunjukkan penilaian rendah.

Ulasan: {sentence}

Peringkat: '''
    return prompt

def prompt_rc_id_translate_with_example(sentence, label_nl, examples):
    prompt = f'''
Saya akan memberikan ulasan.
Silakan menilai ulasan yang diberikan berdasarkan kriteria berikut.
Pilih `{label_nl[1]}` jika ulasan menunjukkan penilaian tinggi dan `{label_nl[0]}` jika ulasan menunjukkan penilaian rendah.
{examples}

Ulasan: {sentence}

Peringkat: '''
    return prompt

def example_rc_id_translate(sentence, label):
    example = f'''
Ulasan: {sentence}

Peringkat: {label}</s>'''
    return example



def prompt_rc_ko_translate(sentence, label_nl):
    prompt = f'''
리뷰를 제공하겠습니다.
다음 기준에 따라 주어진 리뷰를 평가하십시오.
리뷰가 높은 평가를 나타내면 `{label_nl[1]}`을 선택하고 낮은 평가를 나타내면 `{label_nl[0]}`을 선택합니다.

리뷰: {sentence}

평가: '''
    return prompt


def prompt_rc_ko_translate_with_example(sentence, label_nl, examples):
    prompt = f'''
리뷰를 제공하겠습니다.
다음 기준에 따라 주어진 리뷰를 평가하십시오.
리뷰가 높은 평가를 나타내면 `{label_nl[1]}`을 선택하고 낮은 평가를 나타내면 `{label_nl[0]}`을 선택합니다.
{examples}

리뷰: {sentence}

평가: '''
    return prompt

def example_rc_ko_translate(sentence, label):
    example = f'''
리뷰: {sentence}

평가: {label}</s>'''
    return example


def prompt_rc_de_translate(sentence, label_nl):
    prompt = f'''
Ich werde eine Rezension abgeben.
Bitte bewerten Sie die abgegebene Bewertung anhand der folgenden Kriterien.
Wählen Sie `{label_nl[1]}`, wenn die Bewertung eine hohe Bewertung anzeigt, und `{label_nl[0]}`, wenn sie auf eine niedrige Bewertung hinweist.

Rezension: {sentence}

Bewertung: '''
    return prompt

def prompt_rc_de_translate_with_example(sentence, label_nl, examples):
    prompt = f'''
Ich werde eine Rezension abgeben.
Bitte bewerten Sie die abgegebene Bewertung anhand der folgenden Kriterien.
Wählen Sie `{label_nl[1]}`, wenn die Bewertung eine hohe Bewertung anzeigt, und `{label_nl[0]}`, wenn sie auf eine niedrige Bewertung hinweist.
{examples}

Rezension: {sentence}

Bewertung: '''
    return prompt

def example_rc_de_translate(sentence, label):
    example = f'''
Rezension: {sentence}

Bewertung: {label}</s>'''
    return example



def prompt_rc_es_translate(sentence, label_nl):
    prompt = f'''
Voy a hacer una reseña.
Por favor, califique la reseña dada en función de los siguientes criterios.
Elija `{label_nl[1]}` si la reseña indica una evaluación alta y `{label_nl[0]}` si indica una evaluación baja.

Reseña: {sentence}

Valoración: '''
    return prompt

def prompt_rc_es_translate_with_example(sentence, label_nl, examples):
    prompt = f'''
Voy a hacer una reseña.
Por favor, califique la reseña dada en función de los siguientes criterios.
Elija `{label_nl[1]}` si la reseña indica una evaluación alta y `{label_nl[0]}` si indica una evaluación baja.
{examples}

Reseña: {sentence}

Valoración: '''
    return prompt

def example_rc_es_translate(sentence, label):
    example = f'''
Reseña: {sentence}

Valoración: {label}</s>'''
    return example



def prompt_rc_fr_translate(sentence, label_nl):
    prompt = f'''
Je vais donner un avis.
Veuillez noter l’avis en question en fonction des critères suivants.
Choisissez `{label_nl[1]}` si l’avis indique une évaluation élevée et `{label_nl[0]}` s’il indique une évaluation faible.

Critique: {sentence}

Notation: '''
    return prompt

def prompt_rc_fr_translate_with_example(sentence, label_nl, examples):
    prompt = f'''
Je vais donner un avis.
Veuillez noter l’avis en question en fonction des critères suivants.
Choisissez `{label_nl[1]}` si l’avis indique une évaluation élevée et `{label_nl[0]}` s’il indique une évaluation faible.
{examples}

Critique: {sentence}

Notation: '''
    return prompt

def example_rc_fr_translate(sentence, label):
    example = f'''
Critique: {sentence}

Notation: {label}</s>'''
    return example

# --------------------------------------------LS---------------------------------------------------
def prompt_ls_en(sentence, word, lang):
    prompt = f'''
I will provide a sentence and a word included in the sentence.
Please generate a simpler {lang} synonym for the word.
Generate nothing but the synonym.

Sentence: {sentence}

Word: {word}

Synonym: '''
    return prompt

def prompt_ls_en_with_example(sentence, word, examples, lang):
    prompt = f'''
I will provide a sentence and a word included in the sentence.
Please generate a simpler {lang} synonym for the word.
Generate nothing but the synonym.
{examples}

Sentence: {sentence}

Word: {word}

Synonym: '''
    return prompt

def example_ls_en(sentence, word, ans):
    example = f'''
Sentence: {sentence}

Word: {word}

Synonym: {ans}</s>'''
    return example


def prompt_ls_ja(sentence, word):
    prompt = f'''
これから文とその文に含まれる単語を与えます。
与えられた単語に対して、より簡単な日本語の同義語を一つ生成してください。
同義語以外は何も生成しないでください。

文: {sentence}

単語: {word}

同義語: '''
    return prompt

def prompt_ls_ja_with_example(sentence, word, examples):
    prompt = f'''
これから文とその文に含まれる単語を与えます。
与えられた単語に対して、より簡単な日本語の同義語を一つ生成してください。
同義語以外は何も生成しないでください。
{examples}

文: {sentence}

単語: {word}

同義語: '''
    return prompt

def example_ls_ja(sentence, word, ans):
    example = f'''
文: {sentence}

単語: {word}

同義語: {ans}</s>'''
    return example



def prompt_ls_zh(sentence, word):
    prompt = f'''
我会给出一个句子并指定其中的一个词。
请生成一个该词的更简单的中文同义词。
只需生成同义词，不要生成其他内容。

句子: {sentence}

词: {word}

同义词: '''
    return prompt

def prompt_ls_zh_with_example(sentence, word, examples):
    prompt = f'''
我会给出一个句子并指定其中的一个词。
请生成一个该词的更简单的中文同义词。
只需生成同义词，不要生成其他内容。
{examples}

句子: {sentence}

词: {word}

同义词: '''
    return prompt

def example_ls_zh(sentence, word, ans):
    example = f'''
句子: {sentence}

词: {word}

同义词: {ans}</s>'''
    return example



def prompt_ls_de(sentence, word):
    prompt = f'''
Ich gebe Ihnen jetzt einen Satz und ein darin enthaltenes Wort.
Bitte generiere ein einfacheres deutsches Synonym für das Wort.
Generiere nur das Synonym und nichts anderes.

Satz: {sentence}

Wort: {word}

Synonym: '''
    return prompt

def prompt_ls_de_with_example(sentence, word, examples):
    prompt = f'''
Ich gebe Ihnen jetzt einen Satz und ein darin enthaltenes Wort.
Bitte generiere ein einfacheres deutsches Synonym für das Wort.
Generiere nur das Synonym und nichts anderes.
{examples}

Satz: {sentence}

Wort: {word}

Synonym: '''
    return prompt

def example_ls_de(sentence, word, ans):
    example = f'''
Satz: {sentence}

Wort: {word}

Synonym: {ans}</s>'''
    return example



def prompt_ls_es(sentence, word):
    prompt = f'''
Te proporcionaré una oración y una palabra de ella.
Genere un sinónimo en español más sencillo para esta palabra.
Genere solamente el sinónimo.

Oración: {sentence}

Palabra: {word}

Sinónimo: '''
    return prompt

def prompt_ls_es_with_example(sentence, word, examples):
    prompt = f'''
Te proporcionaré una oración y una palabra de ella.
Genere un sinónimo en español más sencillo para esta palabra.
Genere solamente el sinónimo.
{examples}

Oración: {sentence}

Palabra: {word}

Sinónimo: '''
    return prompt

def example_ls_es(sentence, word, ans):
    example = f'''
Oración: {sentence}

Palabra: {word}

Sinónimo: {ans}</s>'''
    return example



def prompt_ls_fr(sentence, word):
    prompt = f'''
Je vais vous donner une phrase et un mot tiré la phrase.
Veuillez générer un synonyme en français plus simple pour le mot tiré.
Ne générez que le synonyme.

Phrase: {sentence}

Mot: {word}

Synonyme: '''
    return prompt

def prompt_ls_fr_with_example(sentence, word, examples):
    prompt = f'''
Je vais vous donner une phrase et un mot tiré la phrase.
Veuillez générer un synonyme en français plus simple pour le mot tiré.
Ne générez que le synonyme.
{examples}

Phrase: {sentence}

Mot: {word}

Synonyme: '''
    return prompt

def example_ls_fr(sentence, word, ans):
    example = f'''
Phrase: {sentence}

Mot: {word}

synonyme: {ans}</s>'''
    return example




def prompt_ls_ja_translate(sentence, word):
    prompt = f'''
文章と文に含まれる単語を提供します。
より簡単な日本語の同義語を生成してください。
同義語以外は何も生成しません。

文: {sentence}

言葉: {word}

同義語: '''
    return prompt

def prompt_ls_ja_translate_with_example(sentence, word, examples):
    prompt = f'''
文章と文に含まれる単語を提供します。
より簡単な日本語の同義語を生成してください。
同義語以外は何も生成しません。
{examples}

文: {sentence}

言葉: {word}

同義語: '''
    return prompt

def example_ls_ja_translate(sentence, word, ans):
    example = f'''
文: {sentence}

言葉: {word}

同義語: {ans}</s>'''
    return example



def prompt_ls_zh_translate(sentence, word):
    prompt = f'''
我将提供一个句子和句子中包含的一个词。
请为该词生成一个更简单的中文同义词。
只生成同义词。

句: {sentence}

词: {word}

同义词: '''
    return prompt

def prompt_ls_zh_translate_with_example(sentence, word, examples):
    prompt = f'''
我将提供一个句子和句子中包含的一个词。
请为该词生成一个更简单的中文同义词。
只生成同义词。
{examples}

句: {sentence}

词: {word}

同义词: '''
    return prompt

def example_ls_zh_translate(sentence, word, ans):
    example = f'''
句: {sentence}

词: {word}

同义词: {ans}</s>'''
    return example



def prompt_ls_de_translate(sentence, word):
    prompt = f'''
Ich werde einen Satz und ein Wort angeben, das in dem Satz enthalten ist.
Bitte generieren Sie ein einfacheres deutsches Synonym für das Wort.
Generieren Sie nichts als das Synonym.

Satz: {sentence}

Wort: {word}

Synonym: '''
    return prompt

def prompt_ls_de_translate_with_example(sentence, word, examples):
    prompt = f'''
Ich werde einen Satz und ein Wort angeben, das in dem Satz enthalten ist.
Bitte generieren Sie ein einfacheres deutsches Synonym für das Wort.
Generieren Sie nichts als das Synonym.
{examples}

Satz: {sentence}

Wort: {word}

Synonym: '''
    return prompt

def example_ls_de_translate(sentence, word, ans):
    example = f'''
Satz: {sentence}

Wort: {word}

Synonym: {ans}</s>'''
    return example



def prompt_ls_es_translate(sentence, word):
    prompt = f'''
Proporcionaré una oración y una palabra incluida en la oración.
Por favor, genere un sinónimo más simple en español para la palabra.
No genere nada más que el sinónimo.

Frase: {sentence}

Palabra: {word}

Sinónimo: '''
    return prompt

def prompt_ls_es_translate_with_example(sentence, word, examples):
    prompt = f'''
Proporcionaré una oración y una palabra incluida en la oración.
Por favor, genere un sinónimo más simple en español para la palabra.
No genere nada más que el sinónimo.
{examples}

Frase: {sentence}

Palabra: {word}

Sinónimo: '''
    return prompt

def example_ls_es_translate(sentence, word, ans):
    example = f'''
Frase: {sentence}

Palabra: {word}

Sinónimo: {ans}</s>'''
    return example



def prompt_ls_fr_translate(sentence, word):
    prompt = f'''
Je vais fournir une phrase et un mot inclus dans la phrase.
Veuillez générer un synonyme français plus simple pour le mot.
Ne générez rien d’autre que le synonyme.

Phrase: {sentence}

Mot: {word}

Synonyme: '''
    return prompt

def prompt_ls_fr_translate_with_example(sentence, word, examples):
    prompt = f'''
Je vais fournir une phrase et un mot inclus dans la phrase.
Veuillez générer un synonyme français plus simple pour le mot.
Ne générez rien d’autre que le synonyme.
{examples}

Phrase: {sentence}

Mot: {word}

Synonyme: '''
    return prompt

def example_ls_fr_translate(sentence, word, ans):
    example = f'''
Phrase: {sentence}

Mot: {word}

Synonyme: {ans}</s>'''
    return example

# --------------------------------------------MRC---------------------------------------------------

def prompt_mrc_en(question, passage):
    prompt = f'''
I will provide a question and a reference sentence.
Please extract the answer to the question from the reference sentence.
Generate nothing but the answer.

Question: {question}

Reference: {passage}

Answer: '''
    return prompt

def prompt_mrc_en_with_example(question, passage, examples):
    prompt = f'''
I will provide a question and a reference sentence.
Please extract the answer to the question from the reference sentence.
Generate nothing but the answer.
{examples}

Question: {question}

Reference: {passage}

Answer: '''
    return prompt

def example_mrc_en(question, passage, ans):
    example = f'''
Question: {question}

Reference: {passage}

Answer: {ans}</s>'''
    return example



def prompt_mrc_ja(question, passage):
    prompt = f'''
これから質問と参照文を与えます。
質問に対する答えを参照文から抽出してください。
答え以外は生成しないでください。

質問: {question}

参照文: {passage}

答え: '''
    return prompt

def prompt_mrc_ja_with_example(question, passage, examples):
    prompt = f'''
これから質問と参照文を与えます。
質問に対する答えを参照文から抽出してください。
答え以外は生成しないでください。
{examples}

質問: {question}

参照文: {passage}

答え: '''
    return prompt

def example_mrc_ja(question, passage, ans):
    example = f'''
質問: {question}

参照文: {passage}

答え: {ans}</s>'''
    return example



def prompt_mrc_zh(question, passage):
    prompt = f'''
我会提供一个问题和一段参考。
请根据这段参考，提取答案，回答问题。
请只生成答案。

问题: {question}

参考: {passage}

答案: '''
    return prompt

def prompt_mrc_zh_with_example(question, passage, examples):
    prompt = f'''
我会提供一个问题和一段参考。
请根据这段参考，提取答案，回答问题。
请只生成答案。
{examples}

问题: {question}

参考: {passage}

答案: '''
    return prompt

def example_mrc_zh(question, passage, ans):
    example = f'''
问题: {question}

参考: {passage}

答案: {ans}</s>'''
    return example




def prompt_mrc_id(question, passage):
    prompt = f'''
Saya akan memberikan sebuah pertanyaan dan sebuah kalimat referensi.
Silakan ekstrak jawaban untuk pertanyaan tersebut dari kalimat referensi.
Hasilkan hanya jawaban tanpa tambahan informasi lain.

Pertanyaan: {question}

Referensi: {passage}

Jawaban: '''
    return prompt

def prompt_mrc_id_with_example(question, passage, examples):
    prompt = f'''
Saya akan memberikan sebuah pertanyaan dan sebuah kalimat referensi.
Silakan ekstrak jawaban untuk pertanyaan tersebut dari kalimat referensi.
Hasilkan hanya jawaban tanpa tambahan informasi lain.
{examples}

Pertanyaan: {question}

Referensi: {passage}

Jawaban: '''
    return prompt

def example_mrc_id(question, passage, ans):
    example = f'''
Pertanyaan: {question}

Referensi: {passage}

Jawaban: {ans}</s>'''
    return example



def prompt_mrc_ko(question, passage):
    prompt = f'''
지금부터 질문과 참고 문서를 입력합니다.
질문에 대한 답변을 참고 문서에서 추출해 주세요.
답변에 해당되는 부분만 생성해 주세요.

질문: {question}

참고 문서: {passage}

답변: '''
    return prompt

def prompt_mrc_ko_with_example(question, passage, examples):
    prompt = f'''
지금부터 질문과 참고 문서를 입력합니다.
질문에 대한 답변을 참고 문서에서 추출해 주세요.
답변에 해당되는 부분만 생성해 주세요.
{examples}

질문: {question}

참고 문서: {passage}

답변: '''
    return prompt

def example_mrc_ko(question, passage, ans):
    example = f'''
질문: {question}

참고 문서: {passage}

답변: {ans}</s>'''
    return example




def prompt_mrc_de(question, passage):
    prompt = f'''
Ich gebe Ihnen jetzt eine Frage und einen Referenzsatz.
Extrahiere die Antwort auf die Frage aus dem Referenzsatz.
Generiere nichts außer der Antwort.

Frage: {question}

Referenzsatz: {passage}

Antwort: '''
    return prompt

def prompt_mrc_de_with_example(question, passage, examples):
    prompt = f'''
Ich gebe Ihnen jetzt eine Frage und einen Referenzsatz.
Extrahiere die Antwort auf die Frage aus dem Referenzsatz.
Generiere nichts außer der Antwort.
{examples}

Frage: {question}

Referenzsatz: {passage}

Antwort: '''
    return prompt

def example_mrc_de(question, passage, ans):
    example = f'''
Frage: {question}

Referenzsatz: {passage}

Antwort: {ans}</s>'''
    return example



def prompt_mrc_es(question, passage):
    prompt = f'''
Te proporcionaré una pregunta y una oración de referencia.
Extraiga la respuesta a la pregunta de la oración de referencia.
Genere únicamente la respuesta.

Pregunta: {question}

Referencia: {passage}

Respuesta: '''
    return prompt

def prompt_mrc_es_with_example(question, passage, examples):
    prompt = f'''
Te proporcionaré una pregunta y una oración de referencia.
Extraiga la respuesta a la pregunta de la oración de referencia.
Genere únicamente la respuesta.
{examples}

Pregunta: {question}

Referencia: {passage}

Respuesta: '''
    return prompt

def example_mrc_es(question, passage, ans):
    example = f'''
Pregunta: {question}

Referencia: {passage}

Respuesta: {ans}</s>'''
    return example



def prompt_mrc_fr(question, passage):
    prompt = f'''
Je vais donner une question et une phrase de référence.
Veuillez extraire la réponse à la question à partir de la phrase de référence.
Ne générez rien d'autre que la réponse.

Question: {question}

Référence: {passage}

Réponse: '''
    return prompt

def prompt_mrc_fr_with_example(question, passage, examples):
    prompt = f'''
Je vais donner une question et une phrase de référence.
Veuillez extraire la réponse à la question à partir de la phrase de référence.
Ne générez rien d'autre que la réponse.
{examples}

Question: {question}

Référence: {passage}

Réponse: '''
    return prompt

def example_mrc_fr(question, passage, ans):
    example = f'''
Question: {question}

Référence: {passage}

Réponse: {ans}</s>'''
    return example



def prompt_mrc_ja_translate(question, passage):
    prompt = f'''
質問と参考文を提供します。
参考文から質問に対する回答を抽出してください。
答えだけを生み出します。

質問: {question}

参考: {passage}

答え: '''
    return prompt

def prompt_mrc_ja_translate_with_example(question, passage, examples):
    prompt = f'''
質問と参考文を提供します。
参考文から質問に対する回答を抽出してください。
答えだけを生み出します。
{examples}

質問: {question}

参考: {passage}

答え: '''
    return prompt

def example_mrc_ja_translate(question, passage, ans):
    example = f'''
質問: {question}

参考: {passage}

答え: {ans}</s>'''
    return example



def prompt_mrc_zh_translate(question, passage):
    prompt = f'''
我将提供一个问题和一个参考句子。
请从参考句子中提取问题的答案。
只生成答案。

问题: {question}

参考: {passage}

答: '''
    return prompt

def prompt_mrc_zh_translate_with_example(question, passage, examples):
    prompt = f'''
我将提供一个问题和一个参考句子。
请从参考句子中提取问题的答案。
只生成答案。
{examples}

问题: {question}

参考: {passage}

答: '''
    return prompt

def example_mrc_zh_translate(question, passage, ans):
    example = f'''
问题: {question}

参考: {passage}

答: {ans}</s>'''
    return example




def prompt_mrc_id_translate(question, passage):
    prompt = f'''
Saya akan memberikan pertanyaan dan kalimat referensi.
Silakan ekstrak jawaban atas pertanyaan dari kalimat referensi.
Hasilkan apa pun selain jawabannya.

Pertanyaan: {question}

Referensi: {passage}

Jawaban: '''
    return prompt

def prompt_mrc_id_translate_with_example(question, passage, examples):
    prompt = f'''
Saya akan memberikan pertanyaan dan kalimat referensi.
Silakan ekstrak jawaban atas pertanyaan dari kalimat referensi.
Hasilkan apa pun selain jawabannya.
{examples}

Pertanyaan: {question}

Referensi: {passage}

Jawaban: '''
    return prompt

def example_mrc_id_translate(question, passage, ans):
    example = f'''
Pertanyaan: {question}

Referensi: {passage}

Jawaban: {ans}</s>'''
    return example



def prompt_mrc_ko_translate(question, passage):
    prompt = f'''
질문과 참조 문장을 제공하겠습니다.
참조 문장에서 질문에 대한 답변을 추출하십시오.
답 외에는 아무것도 생성하지 않습니다.

질문: {question}

참조: {passage}

답변: '''
    return prompt

def prompt_mrc_ko_translate_with_example(question, passage, examples):
    prompt = f'''
질문과 참조 문장을 제공하겠습니다.
참조 문장에서 질문에 대한 답변을 추출하십시오.
답 외에는 아무것도 생성하지 않습니다.
{examples}

질문: {question}

참조: {passage}

답변: '''
    return prompt

def example_mrc_ko_translate(question, passage, ans):
    example = f'''
질문: {question}

참조: {passage}

답변: {ans}</s>'''
    return example



def prompt_mrc_de_translate(question, passage):
    prompt = f'''
Ich werde eine Frage und einen Referenzsatz zur Verfügung stellen.
Bitte entnehmen Sie die Antwort auf die Frage dem Referenzsatz.
Generieren Sie nichts als die Antwort.

Frage: {question}

Referenz: {passage}

Antwort: '''
    return prompt

def prompt_mrc_de_translate_with_example(question, passage, examples):
    prompt = f'''
Ich werde eine Frage und einen Referenzsatz zur Verfügung stellen.
Bitte entnehmen Sie die Antwort auf die Frage dem Referenzsatz.
Generieren Sie nichts als die Antwort.
{examples}

Frage: {question}

Referenz: {passage}

Antwort: '''
    return prompt

def example_mrc_de_translate(question, passage, ans):
    example = f'''
Frage: {question}

Referenz: {passage}

Antwort: {ans}</s>'''
    return example




def prompt_mrc_es_translate(question, passage):
    prompt = f'''
Proporcionaré una pregunta y una oración de referencia.
Extraiga la respuesta a la pregunta de la frase de referencia.
No generes nada más que la respuesta.

Pregunta: {question}

Referencia: {passage}

Respuesta: '''
    return prompt

def prompt_mrc_es_translate_with_example(question, passage, examples):
    prompt = f'''
Proporcionaré una pregunta y una oración de referencia.
Extraiga la respuesta a la pregunta de la frase de referencia.
No generes nada más que la respuesta.
{examples}

Pregunta: {question}

Referencia: {passage}

Respuesta: '''
    return prompt

def example_mrc_es_translate(question, passage, ans):
    example = f'''
Pregunta: {question}

Referencia: {passage}

Respuesta: {ans}</s>'''
    return example



def prompt_mrc_fr_translate(question, passage):
    prompt = f'''
Je vais fournir une question et une phrase de référence.
Veuillez extraire la réponse à la question de la phrase de référence.
Ne générez rien d’autre que la réponse.

Question: {question}

Référence: {passage}

Réponse: '''
    return prompt

def prompt_mrc_fr_translate_with_example(question, passage, examples):
    prompt = f'''
Je vais fournir une question et une phrase de référence.
Veuillez extraire la réponse à la question de la phrase de référence.
Ne générez rien d’autre que la réponse.
{examples}

Question: {question}

Référence: {passage}

Réponse: '''
    return prompt

def example_mrc_fr_translate(question, passage, ans):
    example = f'''
Question: {question}

Référence: {passage}

Réponse: {ans}</s>'''
    return example

