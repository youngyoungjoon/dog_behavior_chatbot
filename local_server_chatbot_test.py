from flask import Flask
from hanspell import spell_checker
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
from train_torch import KoGPT2Chat
import time
from answer_collection import filtering, filtering_6
import pandas as pd
import csv


app = Flask(__name__)

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


@app.route("/predict", methods=["POST"])
def predict(model):
    tok = TOKENIZER
    sent = '0'

    csv_dataset = pd.read_csv(r'D:\KoGPT2-chatbot-master\KoGPT2-chatbot-master\Test_dataset\Test_len_30_dataset.csv',
                              encoding='utf-8')  # Test데이터 csv 파일

    q_list = list(csv_dataset["Q"].tolist())  # "Q"로 된 열 리스트로 변환   Q = 질문
    a_list = list(csv_dataset["A"].tolist())  # "A"로 된 열 리스트로 변환   A = 정답

    f = open(r"Test_result\08_17_local_server_ver_len_30_result_test.csv", 'a', newline='', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(["Q", "A", "Chat_Answer", "acc", "time", "pre_value"])

    num_sum = 0
    num_time_sum = 0
    num_acc_sum = 0
    for i in range(len(q_list)):

        acc = "2"
        with torch.no_grad():
            while acc != "1" or acc != "0":
                start = time.time()
                q = q_list[i]
                a = ''

                spell_check = spell_checker.check(q).as_dict()  # 네이버 맞춤법 검사기 기반 라이브러리를 활용해 맞춤법 교정
                q = spell_check['checked']  # 올바르게 바뀐 문장으로 바꿈

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
                    a += gen.replace('▁', ' ')
                    if len(a) >= 14:  # 필터링에서 걸러주므로 쓸때없이 문장생성하는 시간을 줄임
                        break
                    #if len(a) >= 6:
                     #   break
                    if gen == EOS:
                        break
                a = filtering_6(a.strip())
                end_time = time.time() - start  # 챗봇이 답변을 내리는 시간
                print(a)
                if a == a_list[i].strip():
                    acc = "1"
                    result_value = torch.max(
                        pred,
                        dim=-1)[0].tolist()[0][1]
                    num_sum += int(acc)
                    num_time_sum += end_time
                    num_acc_sum += result_value
                    wr.writerow([q, a_list[i], a, acc, str(end_time),
                                 result_value])  # ["Q", "A", "Chat_Answer","acc", "time", "threshold"]
                    break
                else:
                    acc = "0"
                    num_time_sum += end_time
                    # num_acc_sum += result_value
                    wr.writerow([q, a_list[i], a, acc, str(end_time), ''])  # ["Q", "A", "Chat_Answer","acc", "time"]
                    break

    all_num_acc_sum = num_acc_sum / len(q_list)  # 챗봇 답이 고른 정확도 평균 -> 소수점의 숫자가 나오므로 float으로 변환
    all_acc = num_sum / len(q_list)  # 실제 정확도 평균 -> 소수점의 숫자가 나오므로 float으로 변환
    all_time = num_time_sum / len(q_list)  # 평균 시간, filtering까지 포함해서
    wr.writerow(['', '', '', all_acc, all_time, all_num_acc_sum])  # 실제 정확도 평균, 평균 시간, 챗봇이 답을 고른 정확도 평균
    f.close()


if __name__ == '__main__':

    PORT = 50051
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KoGPT2Chat.load_from_checkpoint('2021_08_17/model_chp/model_-last.ckpt').to(device)
    predict(model)
    app.run(host="0.0.0.0", debug=True, port=PORT)