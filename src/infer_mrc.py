from prompt_template import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import os
import argparse
import random

random.seed(42)


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer


def load_data_mrc(lang_dataset_long):
    if lang_dataset_long in ['japanese', 'indonesian', 'korean']:
        dataset = load_dataset("khalidalt/tydiqa-goldp", lang_dataset_long, split="validation")

    elif lang_dataset_long == 'spanish':
        dataset = load_dataset('PlanTL-GOB-ES/SQAC', split='test', revision='refs/pr/4')
        
    elif lang_dataset_long == 'german':
        dataset = load_dataset('deepset/germanquad', split='test')
    
    elif lang_dataset_long == 'french':
        with open('./../datasets/FQuAD/valid.json') as f:
            data = json.load(f)
        dataset = data['data']
    
    elif lang_dataset_long == 'chinese':
        with open('./../datasets/DRCD/DRCD_test.json') as f:
            data = json.load(f)
        dataset = data['data']
        
    return dataset

def get_exams(lang_dataset_long):
    exams = []
    if lang_dataset_long == 'japanese':
        exams.append({'question': '森上亜希子の出身はどこ', 'reference': '森上 亜希子（もりがみ あきこ、1980年1月12日 - ）は、大阪市出身の元女子プロテニス選手。身長167cm、体重56kg、右利き。フォアハンド・ストローク、バックハンド・ストロークとも両手打ち。シングルス自己最高ランキングは41位（2005年8月15日付）。ミキハウス所属。', 'answer': '大阪市'})
        exams.append({'question': 'イギリス王室属国のマン島の面積は？', 'reference': 'マン島（マンとう、Manx: Ellan Vannin、またはMannin、English: Isle of Man または Mann）は、グレートブリテン島とアイルランド島に囲まれたアイリッシュ海の中央に位置する島。面積は約572km2。主都はダグラス。人口は80,058人（2006年時点）。', 'answer': '572km2'})
        exams.append({'question': 'ルイス・キャロルはいつ生まれた？', 'reference': 'ルイス・キャロル（Lewis Carroll ˈluːɪs ˈkæɹəł, 1832年1月27日 - 1898年1月14日）は、イギリスの数学者、論理学者、写真家、作家、詩人である。', 'answer': '1832年1月27日'})
        exams.append({'question': '『ブリタニア列王史』の著者は誰？', 'reference': '『ブリタニア列王史』（ブリタニアれつおうし、ブリタニア列王伝、Historia Regum Britanniae）は、1136年頃にジェフリー・オブ・モンマスがラテン語で書いたブリテン（グレートブリテン島）に関する偽史書。ホメーロスの『イーリアス』に登場するトロイア人たちの子孫がブリテン国家を建設するところから、7世紀のアングロ・サクソン人によるブリテン支配までの、2000年間のブリトン人王たちの生涯を年代順に物語っている。「アーサー王物語」など「ブリテンの話材」の核となっている。', 'answer': 'ジェフリー・オブ・モンマス'})
        exams.append({'question': '日本で最古の民間保険会社は何？', 'reference': '日本では慶応3年（1868年）に福澤諭吉が著書「西洋旅案内」の中で欧米の文化の一つとして近代保険制度(損害保険、生命保険)を紹介したことが発端となり[4]、1880年に岩倉使節団の随員だった若山儀一らによる日東保生会社（日本初の生命保険会社）開業されるが、倒産してしまう[5]。1881年（明治14年）7月、福沢諭吉門下の阿部泰蔵によって現存最古の保険会社・有限明治生命保険会社が開業された。1888年には国内で2番目の保険会社として帝国生命(現朝日生命)、3番目に1889年には日本生命が誕生した。だが、当初は「人の生死によって金儲けをするのか」という誤解に基づく批判も多く、その普及には時間がかかった。', 'answer': '有限明治生命保険会社'})

    elif lang_dataset_long == 'indonesian':
        exams.append({'question': 'Dimana James Hepburn meninggal?', 'reference': 'Dia dipenjarakan di Puri Dragsholm, 75 kilometer Kopenhagen. Dia ditahan dalam apa yang dikatakan sebagai kondisi yang mengerikan. Dia meninggal pada bulan April 1578.[8][10]', 'answer': 'Puri Dragsholm'})
        exams.append({'question': 'Berapakah luas Kabupaten Kendal ?', 'reference': 'Kabupaten Kendal dan terletak 25km di sebelah barat Kota Semarang Kendal dilalui jalan Pantura (jalan negara) yang menghubungkan Jakarta-Semarang-Surabaya. Kendal mempunyai luas wilayah sebesar 1.002,23 Km2 untuk daratan dan luas wilayah sebesar 313,20 Km2 totalnya seluas 1315,43 Km2 yang terbagi menjadi 20 Kecamatan dengan 265 Desa serta 20 Kelurahan.', 'answer': '1315,43 Km2'})
        exams.append({'question': 'kapankah Kerajaan Goguryeo didirikan pertama kali?', 'reference': 'Goguryeo adalah kerajaan paling besar di antara Tiga Kerajaan. Goguryeo didirikan tahun 37 SM oleh Jumong (Dongmyeongseong) pertama memeluk Buddhisme pada tahun 372 pada masa pemerintahan Raja Raja Sosurim.', 'answer': '37 SM'})
        exams.append({'question': 'Siapakah penulis Kitab Yosua?', 'reference': 'Tradisi Yahudi mengatakan bahwa penulis kitab ini adalah Yosua bin Nun, abdi Musa, yang ditahbiskan menjadi penerusnya dan memimpin orang Israel memasuki dan menduduki tanah Kanaan. Talmud mengatakan bahwa kitab ini ditulis oleh Yosua kecuali ayat-ayat terakhirnya (24:29-33) yang ditambahkan oleh Imam Besar Pinehas bin Eleazar.', 'answer': 'Yosua bin Nun'})
        exams.append({'question': 'Apa album pertama band Metallica?', 'reference': 'Formasi pertama Metallica adalah Lars Ulrich (drum), James Hetfield (vokal dan gitar), Lloyd Grant (gitar) dan Ron Mc Govney (bass). Formasi inilah yang kemudian melahirkan lagu pertama berjudul Hit The Light, yang kemudian masuk album kompilasi rock Metal Massacre tahun 1981.', 'answer': 'Metal Massacre'})

    elif lang_dataset_long == 'korean':
        exams.append({'question': '예수가 태어난 곳은 어디인가?', 'reference': '예수 그리스도는 갈릴리라는 시골 출신이었으므로 그분의 출생에 관한 정확한 역사적 기록은 드물다. 오늘날에는 복음서의 기록을 바탕으로 예수 그리스도의 출생일과 태어난 장소를 미루어 짐작하고 있으며, 대체로 역사상의 예수은 기원전 약 2년~4년 경에 태어난 것으로 추정된다.', 'answer': '갈릴리라'})
        exams.append({'question': '한라산의 높이는 얼마나 되나요?', 'reference': '한라산(漢拏山, )은 제주특별자치도에 있는 해발 1,947.06m, 면적 약 1,820km²의 화산으로, 제주도의 면적 대부분을 차지하고 있다. 금강산, 지리산과 함께 삼신산(三神山)이라 불러왔다.', 'answer': '1,947.06m'})
        exams.append({'question': '전두환은 언제 정권을 잡았는가?', 'reference': '1980년 3월에는 최규하, 신현확에게 중앙정보부장직을 요구, 그해 4월 14일 중앙정보부장 서리직을 겸직하였으며 대학생들의 민주화 운동을 진압하기 위해 5·17 비상계엄 전국확대 조치를 발동하고, 5·18 광주 민주화 운동의 유혈진압을 주도했다. 5월 27일에는 국보위를 조직하고 상임위원장이 되어 정부의 실권을 장악했다. 1980년 9월 1일 장충체육관에서의 간선제를 통해 스스로 대한민국의 제11대 대통령에 취임하였다.', 'answer': '1980년 9월 1일'})
        exams.append({'question': '영화 2012 제작 감독은 누구인가요?', 'reference': '《2012》는 롤란트 에머리히 감독 및 각본의 2009년 미국 SF 재난 모험 영화이다. 감독은 롤란트 에머리히가 맏았다. 원래는 로스앤젤레스에서 촬영하기로 계획했었으나, 2008년 8월 밴쿠버에서 촬영을 시작했다.', 'answer': '롤란트 에머리히'})
        exams.append({'question': '세상에서 가장 많은 희생자가 발생된 전쟁은 무엇인가?', 'reference': '제2차 세계 대전( 또는 World War II)은 1939년 9월 1일부터 1945년 9월 2일까지 치러진, 인류 역사상 가장 많은 인명 피해와 재산 피해를 남긴 가장 파괴적인 전쟁이다.', 'answer': '제2차 세계 대전'})

    elif lang_dataset_long == 'spanish':
        exams.append({'question': '¿Dónde nació Salvador Allende?', 'reference': 'Salvador Guillermo Allende Gossens​​ (Santiago,​ 26 de junio de 1908-ibidem, 11 de septiembre de 1973) fue un médico cirujano y político socialista chileno, presidente de Chile desde el 3 de noviembre de 1970 hasta el día de su muerte. Allende participó en política desde sus estudios en la Universidad de Chile. Fue sucesivamente diputado, ministro de Salubridad del gobierno de Pedro Aguirre Cerda​ y senador desde 1945 hasta 1970, ejerciendo la presidencia en la cámara alta del Congreso entre 1966 y 1969.', 'answer': 'Santiago'})
        exams.append({'question': '¿Cuánto dinero le quita el ejecutivo a su superior?', 'reference': 'Un ejecutivo publicitario descubre que su mujer le engaña con su jefe y decide vengarse de éste robándole un millón de dólares. El azar hará que se cruce en su destino con un ladrón de poca monta que está dispuesto a ayudarle en su empresa.', 'answer': 'un millón de dólares'})
        exams.append({'question': '¿Cuándo arribaron los judíos a Tesalónica?', 'reference': 'La historia de los judíos de Tesalónica es la de la comunidad judía, principalmente sefardí, que habitó la ciudad griega de Tesalónica, también llamada Salónica, desde su llegada a la misma, a finales del siglo XV, hasta su aniquilación casi completa durante la Segunda Guerra Mundial. La comunidad judía fue durante varios siglos mayoritaria en la ciudad, lo que convierte a Tesalónica en el único caso conocido de una ciudad de la Diáspora de este tamaño con predominio judío.', 'answer': 'a finales del siglo XV'})
        exams.append({'question': '¿Quién vive en el jardín del vecino?', 'reference': 'Allí se han ido los dos, la persona y la imagen. Dijo la palabra fea en francés y era lo menos que podía hacer un hombre con su apellido. Mi vecino es un psiquiatra un poco loco. Casi todos lo están y el hecho de haber elegido esa profesión parece ser un síntoma. En su jardín vive un búho.', 'answer': 'un búho'})
        exams.append({'question': '¿En qué partido milita Joan Boada?', 'reference': 'El diputado Joan Boada (IC-V) ha pedido a la Conselleria de Justícia que instale medidas de seguridad contra incendios en el edificio de los juzgados de Mollet, tras el reciente fuego que destruyó las oficinas de una empresa de la primera planta.', 'answer': 'IC-V'})

    elif lang_dataset_long == 'german':
        exams.append({'question': 'In welcher Stadt ist die Zentrale der BBC?', 'reference': 'London\n\n=== Rundfunk und Fernsehen ===\nLondon ist Hauptsitz bedeutender Rundfunk- und Fernsehanstalten wie (BBC, ITV, Channel 4, Five und Sky). Im Bush House zwischen Aldwych und Strand waren bis zum Jahr 2012 der BBC World Service und die Abteilung Neue Medien des BBCi beheimatet.\nDie BBC wurde am 18. Oktober 1922 in London als unabhängiger Radiosender gegründet. Die erste Ausstrahlung eines Programms fand am 14. November 1922 aus einem Londoner Studio statt. Die BBC betreibt mehrere Rundfunk- und Fernsehsender.', 'answer': 'London'})
        exams.append({'question': 'Wie lange sollte derIPod minis der ersten Generation lauf Herstellerangaben in Betrieb sein können?', 'reference': 'IPod\n\n==== Erste Generation ====\nDas Gerät wurde in den USA im Februar 2004 und in allen anderen Ländern am 24. Juli 2004 eingeführt. Der iPod mini ist ein wesentlich kleinerer iPod, der jedoch nur eine 4-GB-Festplatte enthielt. Er erhielt als erstes Modell das ''ClickWheel,'' das Scrollrad und Knöpfe vereinigt. Den iPod mini gab es in fünf verschiedenen Farben: Blau, Pink, Grün, Silber und Gold. Als Akkulaufzeit gab Apple acht Stunden bei mittlerer Lautstärke und ausgeschalteter Bildschirmbeleuchtung an.', 'answer': 'acht Stunden'})
        exams.append({'question': 'Seit wann ist Guam von Menschen bewohnt?', 'reference': 'Guam\n\n=== Frühgeschichte ===\nBlick auf den National Historic Park\nEs wird vermutet, dass die Insel um 2000 v. Chr. von Seefahrern aus dem südöstlichen Indonesien entdeckt und besiedelt wurde. Eine andere Theorie besagt, dass die Besiedlung der Insel von den Philippinen aus erfolgte. Als Quellen für die Zeit vor den Europäern gelten die Legenden und Mythen der Chamorros, archäologische Grabungen, Aufzeichnungen von Jesuiten und Forschungen von Wissenschaftlern wie Otto von Kotzebue und Louis de Freycinet.', 'answer': '2000 v. Chr. '})
        exams.append({'question': 'Wer gewann zwischen Bush und Kerry in Ohio?', 'reference': 'John_Kerry\n\n==== Wahltag ====\nNach den Wahlen des 2. November war das für den Ausgang der Wahl ausschlaggebende Ergebnis im Bundesstaat Ohio noch lange nicht entschieden. Am 3. November zeichnete sich jedoch ab, dass Bush auch diesen Staat und somit die Wahl gewinnen würde. Im Ergebnis unterlag Kerry mit 48 % der Stimmen, während Bush 51 % der Stimmen auf sich vereinen konnte. Daraufhin gratulierte Kerry seinem Konkurrenten zum Sieg und forderte die USA auf, nun die Bitterkeiten der Wahlen hinter sich zu lassen', 'answer': 'Bush'})
        exams.append({'question': 'Was ist der größte Flughafen in Miami?', 'reference': 'Miami\n\n=== Flugverkehr ===\nDer Miami International Airport ist einer der größten internationalen Flughäfen der Welt und ein bedeutendes Luftfahrt-Drehkreuz. 2016 wurden 44,6 Millionen Passagiere abgefertigt. Der Flughafen wird von vielen internationalen Fluggesellschaften angeflogen.\nWeitere Flughäfen für den internationalen Flugverkehr sind der Fort Lauderdale-Hollywood International Airport und der Palm Beach International Airport. National ist Miami auch über den Opa-locka Executive Airport zu erreichen.', 'answer': 'Der Miami International Airport'})
 
    elif lang_dataset_long == 'french':
        exams.append({'question': "Où se trouve la grande partie de l'Antarctique oriental ? ", 'reference': "L'Antarctique oriental s'étend du côté océan Indien de la chaîne Transantarctique et comprend la Terre de Coats, la Terre de la Reine-Maud, la Terre d'Enderby, la Terre de Mac Robertson, la Terre de Wilkes et la Terre Victoria. Toute cette région, sauf une petite partie, se trouve dans l'hémisphère est. L'Antarctique oriental est largement couvert par l'inlandsis Est-Antarctique.", 'answer': "l'hémisphère est"})
        exams.append({'question': "Combien le Vasa avait il d'obusiers de gros calibre ?", 'reference': "Les canons présents sur le Vasa étaient toujours fondus individuellement dans des moules à usage unique, mais avaient une telle précision d'uniformité dans leur conception que leur taille varie de quelques millimètres seulement et que leur calibre est toujours presque exactement de 146 mm (5,7 pouces. Le reste de l'armement du Vasa se composait de huit canons de trois livres, six obusiers de gros calibre pour l'utilisation lors des opérations d'abordage, et deux fauconneaux d'une livre. En outre, il emportait à son bord 894 kg de poudre pour le tir des canons.", 'answer': 'six'})
        exams.append({'question': 'Quand a eu lieu la guerre sociale ?', 'reference': 'Au Ier siècle av. J.-C., pendant la guerre sociale, les Étrusques ne prennent pas part à la lutte entre Rome et certains de ses alliés. Ils en retirent cependant un bénéfice lorsque Rome accorde le droit de cité à tous les Italiens. En revanche, lors de la première guerre civile entre Marius et Sylla, ils choisissent le mauvais camp. Le vainqueur, Sylla, se montre rancunier et châtie les villes qui ont pris parti pour Marius : en 81 et 80 av. J.-C., il confisque leurs biens et établit des colonies militaires à Arezzo et Fiesole.', 'answer': 'Ier siècle av. J.-C'})
        exams.append({'question': "Qui doit terminer la monoplace ?", 'reference': "L'Andrea Moda S921 est achevée en mars 1992, Simtec devant terminer la monoplace (notamment greffer le moteur et le système de transmission) au plus vite. Pour ce faire, certains mécaniciens d'Andrea Moda Formula, ainsi que quelques mécaniciens d'écuries rivales désireux de gagner un peu d'argent, ont prêté main-forte au bureau d'études britannique.", 'answer': 'Simtec'})
        exams.append({'question': "Qu'est-ce qu'une Springboard Hart Attack ?", 'reference': "Après le début du match, Bret Hart annonce au micro qu'il est en réalité l'allié de la Hart Dynasty (qui effectue alors un face turn). Ces derniers s'en prennent alors à McMahon (Natalya lui donne une gifle, tandis que les deux autres lui portent sur le sol et depuis le coin du ring leur prise de finition par équipe, la Springboard Hart Attack). Bret remporte ensuite le match en faisant abandonner McMahon dans le ring avec sa prise de finition, le Sharpshooter, après de nombreux coups de chaise,.", 'answer': "prise de finition par équipe"})

    elif lang_dataset_long == 'chinese':
        exams.append({'question': '成田國際機場所在的行政區其政府位在何地？', 'reference': '千葉縣是日本的一級行政區之一，位置在於本州的關東地區，西面緊臨東京都，屬於日本首都圈的範圍。有許多在東京的基礎建設，如成田國際機場和幕張展覽館都在千葉縣境內。千葉縣政府位於千葉市。人口次於埼玉縣，排名在日本各都道府縣第6位，面積為第28名。千葉縣位於關東地方的東南部，由原上總國、安房國、下總國三國合併設置而成，但下總國的猿島郡、結城郡、豐田郡以及相馬郡、葛飾郡的一部分現在屬茨城縣範圍，葛飾郡有一部分分別劃入東京都和埼玉縣，1873年6月15日，北部的印旛縣與南部的木更津縣合併，設立千葉縣，後於1875年5月7日劃入新治縣利根川以南的地區，同時將舊印旛縣利根川以北地區移到茨城縣，形成現在的縣域。', 'answer': '千葉市'})
        exams.append({'question': '廣珠城際鐵路平均每小時可以走多遠？', 'reference': '廣州是京廣鐵路、廣深鐵路、廣茂鐵路、廣梅汕鐵路的終點站。2009年末，武廣客運專線投入運營，多單元列車覆蓋980公里的路程，最高時速可達350公里/小時。2011年1月7日，廣珠城際鐵路投入運營，平均時速可達200公里/小時。廣州鐵路、長途汽車和渡輪直達香港，廣九直通車從廣州東站開出，直達香港九龍紅磡站，總長度約182公里，車程在兩小時內。繁忙的長途汽車每年會從城市中的不同載客點把旅客接載至香港。在珠江靠市中心的北航道有渡輪線路，用於近江居民直接渡江而無需乘坐公交或步行過橋。南沙碼頭和蓮花山碼頭間每天都有高速雙體船往返，渡輪也開往香港中國客運碼頭和港澳碼頭。', 'answer': '200公里'})
        exams.append({'question': '錄影回放技術的第一次在世俱盃應用是在哪一年？', 'reference': '2017年4月26日在智利聖地亞哥舉行的南美洲足協大會上，國際足總會長詹尼·因凡蒂諾在接受採訪時確認，2018年世界盃將啟用即時回放技術。據悉，在2018年的俄羅斯世界盃上，該技術將涉及十二碼、進球、紅牌以及誤判球員四個領域。詹尼·因凡蒂諾說：「我們將在俄羅斯世界盃上採用錄影回放技術，因為到目前為止這項技術給我們帶來的都是積極的結果。」「在2017年，現場球迷和電視機前的觀眾都能在幾秒鐘之內了解到判罰是否正確，而我不希望裁判成為唯一一位不知道自己犯錯的人。」詹尼·因凡蒂諾說。在2016年的世俱盃中，錄影回放技術首次在國際足總賽事中得到應用。在今年3月舉行的法國與西班牙的熱身賽上，此項技術兩次更正了裁判的判罰。目前已經有數個國家的國內聯賽應用了這一技術，澳大利亞A聯賽是第一個啟用視頻回放技術的足球頂級聯賽。', 'answer': '2016'})
        exams.append({'question': '非洲國家的元首中誰是最老的？', 'reference': '另一方面國內的經濟情況未見起色。2008年2月，通貨膨脹率達165,000%；2008年6月，通貨膨脹率達200,000%，該國央行並於2008年7月21日發行面值1000億元的辛巴威元鈔票。辛巴威從2008年8月1日起貨幣改制，100億舊辛巴威元相當於1新辛巴威元。2009年一月消息，辛巴威將於近日發行一套世界上最大面額的新鈔，這套面額在萬億以上的新鈔包括10兆、20兆、50兆和100兆辛元四種。在2009年4月12日時，該國政府更宣布因為已經難以維持貨幣價值，將停用本國貨幣一年。2013年8月執政33年的穆加比繼續連任總統，成為非洲最年長的國家元首。', 'answer': '穆加比'})
        exams.append({'question': '如果要做為資料流設計的傳輸格式最好使用什麼樣的形式？', 'reference': '圖像互換格式是一種點陣圖圖形檔案格式，以8位元色重現真彩色的圖像。它實際上是一種壓縮文件，採用壓縮演算法進行編碼，有效地減少了圖檔在網路上傳輸的時間。它是目前全球資訊網廣泛應用的網路傳輸圖像格式之一。優秀的壓縮演算法使其在一定程度上保證圖像品質的同時將體積變得很小。可插入多影格，從而實現動畫效果。可設定透明色以產生物件浮現於背景之上的效果。由於採用了8位元壓縮，最多只能處理256種顏色，故不宜應用於真彩色圖片。圖像互換格式主要是為資料流而設計的一種傳輸格式，而不是作為檔案的儲存格式。它具有順序組織形式而不是隨機組織形式。', 'answer': '順序組織形式'})

    return exams


def generation(model_name, model, tokenizer, prompt):
    if 'Mistral-Nemo' in model_name:
        tokenized_input = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
    else:
        tokenized_input = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **tokenized_input,
            max_new_tokens=30,
            do_sample=True,
            top_p=0.9,
            temperature=0.6
        )[0]

    output = tokenizer.decode(output, skip_special_tokens=False)
    output = output.replace(tokenizer.decode(
        tokenized_input['input_ids'][0], skip_special_tokens=False), '')
    return output


def normalize_pred(pred, eos_token, eos_token_2):
    pred = pred.strip().split('\n')[0]
    pred = pred.split(eos_token)[0].strip()
    if eos_token_2 is not None:
        pred = pred.split(eos_token_2)[0].strip()
    if len(pred) > 0:
        if pred[-1] == '.' or pred[-1] == '。':
            pred = pred[:-1]
        pred = pred.strip()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_dataset', '-ld', 
                        required=True, help='language of dataset', choices=['de', 'es', 'fr', 'id', 'ja', 'ko', 'zh'])
    parser.add_argument('--lang_prompt', '-lp',
                        required=True, help='language of instruction', choices=['en', 'tgt', 'tgt-translate'])
    parser.add_argument('--num_shot', '-ns', 
                        default='zero', help='num of examples', choices=['zero', 'few'])
    parser.add_argument('--model_name_ease', '-mne', 
                        required=True, help='short model name like qwen2-i', choices=['llama3-i', 'qwen2-i', 'mistraln-i', 'llama3-b', 'qwen2-b', 'mistraln-b'])
    parser.add_argument('--use_chat_template', '-uct', 
                        help='when using chat_template, please flag this', action='store_true')

    args = parser.parse_args()
    lang_dataset = args.lang_dataset
    lang_prompt = args.lang_prompt
    num_shot = args.num_shot
    model_name_ease = args.model_name_ease
    use_chat_template = args.use_chat_template

    lang_short2long = {'de':'german', 
                        'es':'spanish', 
                        'fr':'french',
                        'id':'indonesian', 
                        'ja':'japanese',
                        'ko':'korean',
                        'zh':'chinese'}
    lang_dataset_long = lang_short2long[lang_dataset]
    print('lang_dataset is', lang_dataset_long)

    if model_name_ease == 'llama3-b':
        model_name = 'meta-llama/Meta-Llama-3-8B'
        eos_token = '<|end_of_text|>'
        eos_token_2 = '<|eot_id|>'
    elif model_name_ease == 'llama3-i':
        model_name = 'lightblue/suzume-llama-3-8B-multilingual'
        eos_token = '<|end_of_text|>'
        eos_token_2 = '<|eot_id|>'
    elif model_name_ease == 'qwen2-b':
        model_name = 'Qwen/Qwen2-7B'
        eos_token = '<|endoftext|>'
        eos_token_2 = '<|im_end|>'
    elif model_name_ease == 'qwen2-i':
        model_name = 'Qwen/Qwen2-7B-Instruct'
        eos_token = '<|endoftext|>'
        eos_token_2 = '<|im_end|>'
    elif model_name_ease == 'mistraln-b':
        model_name = 'mistralai/Mistral-Nemo-Base-2407'
        eos_token = '</s>'
        eos_token_2 = None
    elif model_name_ease == 'mistraln-i':
        model_name = 'mistralai/Mistral-Nemo-Instruct-2407'
        eos_token = '</s>'
        eos_token_2 = None

    print(model_name)
    model, tokenizer = load_model(model_name)
    dataset = load_data_mrc(lang_dataset_long)
    exams = get_exams(lang_dataset_long)

    cnt_correct = 0  # count of correct instances
    cnt_wrong = 0  # count of wrong instances
    cnt_output_no_option = 0  # count of output not present in reference
    datas_new = list()

    prompt_funcs = {
        'en': prompt_mrc_en,
        'de': prompt_mrc_de,
        'es': prompt_mrc_es,
        'fr': prompt_mrc_fr,
        'id': prompt_mrc_id,
        'ja': prompt_mrc_ja,
        'ko': prompt_mrc_ko,
        'zh': prompt_mrc_zh,
        'de_translate': prompt_mrc_de_translate,
        'es_translate': prompt_mrc_es_translate,
        'fr_translate': prompt_mrc_fr_translate,
        'id_translate': prompt_mrc_id_translate,
        'ja_translate': prompt_mrc_ja_translate,
        'ko_translate': prompt_mrc_ko_translate,
        'zh_translate': prompt_mrc_zh_translate,
        'en_exam': prompt_mrc_en_with_example,
        'de_exam': prompt_mrc_de_with_example,
        'es_exam': prompt_mrc_es_with_example,
        'fr_exam': prompt_mrc_fr_with_example,
        'id_exam': prompt_mrc_id_with_example,
        'ja_exam': prompt_mrc_ja_with_example,
        'ko_exam': prompt_mrc_ko_with_example,
        'zh_exam': prompt_mrc_zh_with_example,
        'de_translate_exam': prompt_mrc_de_translate_with_example,
        'es_translate_exam': prompt_mrc_es_translate_with_example,
        'fr_translate_exam': prompt_mrc_fr_translate_with_example,
        'id_translate_exam': prompt_mrc_id_translate_with_example,
        'ja_translate_exam': prompt_mrc_ja_translate_with_example,
        'ko_translate_exam': prompt_mrc_ko_translate_with_example,
        'zh_translate_exam': prompt_mrc_zh_translate_with_example
    }

    example_funcs = {
        'en': example_mrc_en, 
        'de': example_mrc_de, 
        'es': example_mrc_es, 
        'fr': example_mrc_fr, 
        'id': example_mrc_id,
        'ja': example_mrc_ja,
        'ko': example_mrc_ko,
        'zh': example_mrc_zh,
        'de_translate': example_mrc_de_translate, 
        'es_translate': example_mrc_es_translate, 
        'fr_translate': example_mrc_fr_translate, 
        'id_translate': example_mrc_id_translate,
        'ja_translate': example_mrc_ja_translate, 
        'ko_translate': example_mrc_ko_translate, 
        'zh_translate': example_mrc_zh_translate
    }

    if lang_dataset_long in ['japanese', 'indonesian', 'korean', 'spanish', 'german']:
        for line in tqdm(dataset):
            if lang_dataset_long in ['japanese', 'indonesian', 'korean']:
                question = line['question_text'].strip()
                passage = line['passage_text'].strip()
                answers = line['answers']['text']
            elif lang_dataset_long in ['spanish', 'german']:
                question = line['question'].strip()
                passage = line['context'].strip()
                answers = line['answers']['text']

            if lang_prompt == 'en':
                if num_shot == 'zero':
                    prompt = prompt_funcs[lang_prompt](question, passage)
                elif num_shot == 'few':
                    exams_format = [example_funcs[lang_prompt](
                        exam['question'], exam['reference'], exam['answer']) for exam in exams]
                    exams_concat = '\n'.join(exams_format)
                    prompt = prompt_funcs[f'{lang_prompt}_exam'](question, passage, exams_concat)

            elif lang_prompt == 'tgt':
                if num_shot == 'zero':
                    prompt = prompt_funcs[lang_dataset](question, passage)
                elif num_shot == 'few':
                    exams_format = [example_funcs[lang_dataset](
                        exam['question'], exam['reference'], exam['answer']) for exam in exams]
                    exams_concat = '\n'.join(exams_format)
                    prompt = prompt_funcs[f'{lang_dataset}_exam'](question, passage, exams_concat)

            else:
                if num_shot == 'zero':
                    prompt = prompt_funcs[f'{lang_dataset}_translate'](question, passage)
                elif num_shot == 'few':
                    exams_format = [example_funcs[f'{lang_dataset}_translate'](
                        exam['question'], exam['reference'], exam['answer']) for exam in exams]
                    exams_concat = '\n'.join(exams_format)
                    prompt = prompt_funcs[f'{lang_dataset}_translate_exam'](question, passage, exams_concat)


            prompt = prompt.replace('</s>', tokenizer.eos_token)
            if use_chat_template:
                messages = [{'role':'user', 'content':prompt}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print(prompt)

            output = generation(model_name, model, tokenizer, prompt)
            output_normalize = normalize_pred(output, eos_token, eos_token_2)
            print(output_normalize)

            if output_normalize in answers:
                cnt_correct += 1
            else:
                cnt_wrong += 1
            if output_normalize not in passage:
                cnt_output_no_option += 1

            line['pred'] = output
            line['pred_normalize'] = output_normalize
            datas_new.append(line)

        f1 = cnt_correct / len(dataset)


    else:
        cnt_pred = 0  # count of pred
        for data in tqdm(dataset):
            for paragraph in data['paragraphs']:
                passage = paragraph['context'].strip()
                for qa in paragraph['qas']:
                    question = qa['question'].strip()
                    answers = [answer['text'] for answer in qa['answers']]
                    data_new = {'passage_text': passage, 'question_text': question, 'answers': answers}

                    if lang_prompt == 'en':
                        if num_shot == 'zero':
                            prompt = prompt_funcs[lang_prompt](question, passage)
                        elif num_shot == 'few':
                            exams_format = [example_funcs[lang_prompt](
                                exam['question'], exam['reference'], exam['answer']) for exam in exams]
                            exams_concat = '\n'.join(exams_format)
                            prompt = prompt_funcs[f'{lang_prompt}_exam'](question, passage, exams_concat)

                    elif lang_prompt == 'tgt':
                        if num_shot == 'zero':
                            prompt = prompt_funcs[lang_dataset](question, passage)
                        elif num_shot == 'few':
                            exams_format = [example_funcs[lang_dataset](
                                exam['question'], exam['reference'], exam['answer']) for exam in exams]
                            exams_concat = '\n'.join(exams_format)
                            prompt = prompt_funcs[f'{lang_dataset}_exam'](question, passage, exams_concat)

                    else:
                        if num_shot == 'zero':
                            prompt = prompt_funcs[f'{lang_dataset}_translate'](question, passage)
                        elif num_shot == 'few':
                            exams_format = [example_funcs[f'{lang_dataset}_translate'](
                                exam['question'], exam['reference'], exam['answer']) for exam in exams]
                            exams_concat = '\n'.join(exams_format)
                            prompt = prompt_funcs[f'{lang_dataset}_translate_exam'](question, passage, exams_concat)

                    prompt = prompt.replace('</s>', tokenizer.eos_token)
                    if use_chat_template:
                        messages = [{'role':'user', 'content':prompt}]
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # print(prompt)

                    output = generation(model_name, model, tokenizer, prompt)
                    output_normalize = normalize_pred(output, eos_token, eos_token_2)
                    print(output_normalize)

                    if output_normalize in answers:
                        cnt_correct += 1
                    else:
                        cnt_wrong += 1
                    if output_normalize not in passage:
                        cnt_output_no_option += 1

                    data_new['pred'] = output
                    data_new['pred_normalize'] = output_normalize
                    datas_new.append(data_new)

                    cnt_pred += 1
                    print(f'========{cnt_pred}========')

        f1 = cnt_correct / cnt_pred


    print('cnt_correct: ', cnt_correct)
    print('cnt_output_no_option: ', cnt_output_no_option)
    print('f1: ', f1)

    pred_dir = f'./../pred/mrc/{lang_dataset}/{num_shot}'
    result_dir = f'./../result/mrc/{lang_dataset}/{num_shot}'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(f'{pred_dir}/{model_name_ease}_{lang_prompt}.json', 'w') as f_pred:
        json.dump(datas_new, f_pred, ensure_ascii=False)

    with open(f'{result_dir}/{model_name_ease}_{lang_prompt}.txt', 'a') as f_result:
        f_result.write('cnt_correct: ' + str(cnt_correct) + '\n')
        f_result.write('cnt_wrong: ' + str(cnt_wrong) + '\n')
        f_result.write('cnt_output_no_option: ' +
                       str(cnt_output_no_option) + '\n')
        f_result.write('f1: ' + str(f1) + '\n')


if __name__ == '__main__':
    main()
