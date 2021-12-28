from flask import Flask, request, jsonify
from flask_restful import Api
from hanspell import spell_checker
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
from train_torch import KoGPT2Chat
import time
from answer_collection import filtering, filtering_6
import json


app = Flask(__name__)
app.config['JSON_AS_ASCII'] =False   #answer가 유니코드로 출력되지 않게 지정
api = Api(app)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",

            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KoGPT2Chat.load_from_checkpoint('2021_08_17/model_chp/model_-epoch=370-train_loss=24.78.ckpt').to(device)


@app.route("/predict", methods=["POST"])
def predict():


    data = request.get_json()
    abc = json.dumps(data, ensure_ascii=False)  # ensure_ascii = False로 안하면 유니코드로 읽어버림 무조건 해야댐, 또한 문자열로 읽기 때문에 다시금 딕셔너리 변환이 필요함
    aa = abc.replace('{', '') # 기존에 받았던 딕셔너리를 문자열로 읽어서 딕셔너리 {} 삭제
    aa = aa.replace('}', '')
    aa = aa.replace('"', '') # 값을 보낼 때 문자열도 같이 딸려들어오기 때문에 저것도 제외시켜줘야함
    aa = aa.replace("'", '') # 혹시 모르니 이것도 삭제 앱개발자들한테 얘기해봐야함


    data_list = aa.split(',')

    keys = []  # key끼리
    values = []  # value끼리

    for data in data_list:  # ['question':'value']
        pair = data.split(':')
        keys.append(pair[0])
        values.append(pair[1])

    my_dict = dict(zip(keys, values))  # 딕셔너리로 만들기
    print(my_dict)

    q = my_dict['question']

    if len(q) < 4:  # 질문길이가 짧을 때
        return jsonify({'answer': '질문이 짧습니다.', 'time': '0'})

    start = time.time()

    tok = TOKENIZER
    sent = '0'
    sent_tokens = tok.tokenize(sent)
    spell_check = spell_checker.check(q).as_dict()  # 네이버 맞춤법 검사기 기반 라이브러리를 활용해 맞춤법 교정
    q = spell_check['checked']

    with torch.no_grad():
        a = ''

        while 1:



            input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(
                dim=0)  # pre-trained - kogpt2가 inputs 값을 추론하는 인덱스
            # print(U_TKN + ' ' + q + ' ' + SENT + ' ' + sent + ' ' + S_TKN + ' ' + a)
            # print(tok.encode(U_TKN + q + SENT + sent + S_TKN + a))
            # print(input_ids)

            pred = model(input_ids)  # fine-tuning kogpt2 loading한 Model에 push

            gen = tok.convert_ids_to_tokens(
                torch.argmax(  # 리스트당 최대값이 존재하는 인덱스들
                    pred,
                    dim=-1).squeeze().numpy().tolist())[-1]
            # print( torch.argmax(pred,dim=-1))

            if gen == EOS:
                break

            a += gen.replace('▁', ' ')
            if len(a) >= 14 : #필터링에서 걸러주므로 쓸때없이 문장생성하는 시간을 줄임

                break


        if len(a) > 1:

            chatbot_answer = filtering_6(a.strip())
            #print("Alphado BOT > {}".format(filtering_6(a.strip())))
            #print('총 시간은: {}'.format(end-start))

            #print('confidence level: {}'.format(torch.max(
            #    pred,
            #    dim=-1)[0].tolist()[0][1]))

        else:
            chatbot_answer = 'Nothing'
        end = time.time()


    return jsonify({'answer':chatbot_answer, 'time':end-start})

@app.route('/',methods=['GET'])
def index():
    return 'Hello World!'


if __name__ == '__main__':

    PORT = 8080
    app.run(host="0.0.0.0", debug=True, port=PORT)

