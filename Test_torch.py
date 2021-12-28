# -*- coding: utf-8 -*-
#한국어는 접사, 조사 등 단어 뒤에 붙으면 문장형태가 다양하게 표현되기 때문에 영어보다 hell임
import argparse
import logging
from hanspell import spell_checker
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import os
import csv
import time
from answer_collection import filtering, filtering_6

#GPU 0번 돌리기
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='2021_08_03_lightning/model_chp/model_-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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


class CharDataset(Dataset):
    def __init__(self, chats, max_len=256):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = self.tokenizer.tokenize(self.q_token + q + \
                                          self.sent_token + sentiment)
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),
               labels_ids)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        #self.loss_function = torch.nn.MSELoss(reduction='none')      #기존에 lossfunction으로 MSE를 사용
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')    # VER2(08_02)부터 CrossEntropyLoss를 사용

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=256,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=128,
                            help='batch size for training (default: 96) 3090으로 돌린건 128')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv(r'D:\KoGPT2-chatbot-master\KoGPT2-chatbot-master\Chatbot_data\Dog_dataset.csv')
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=0, #num_workers = 2 로 되어있었음 2로 되면 File "C:\Users\young\Anaconda3\envs\KoGPT2_chatbot_envs\lib\multiprocessing\reduction.py", line 60, in dump ForkingPickler(file, protocol).dump(obj) AttributeError: Can't pickle local object 'get_cosine_schedule_with_warmup.<locals>.lr_lambda' 이렇게 오류
            shuffle=True, collate_fn=self._collate_fn)  #pytorch에 있는 num_workers은 학습 도중 CPU의 작업을 몇 개의 코어를 사용해서 진행할지에 대한 설정 파라미터입니다.
                                                        #역시 적당히라는게 가장 어렵겠지만 하이퍼-파라미터를 튜닝하는 것처럼 결국 모델에 가장 적합한 num_workers 수치를 찾아내는 것도 파라미터 튜닝으로 볼 수 있습니다.
        return train_dataloader

    def chat(self, sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)

        csv_dataset = pd.read_csv(r'D:\KoGPT2-chatbot-master\KoGPT2-chatbot-master\Test_dataset\Test_len_30_dataset.csv', encoding='utf-8') #Test데이터 csv 파일

        q_list = list(csv_dataset["Q"].tolist())  #"Q"로 된 열 리스트로 변환   Q = 질문
        a_list = list(csv_dataset["A"].tolist())  #"A"로 된 열 리스트로 변환   A = 정답


        f = open(r"Test_result\08_02_len_30_result_test.csv", 'a', newline='', encoding='utf-8')
        wr = csv.writer(f)
        wr.writerow(["Q", "A", "Chat_Answer", "acc", "time", "pre_value"])

        num_sum = 0
        num_time_sum = 0
        num_acc_sum = 0
        for i in range(len(q_list)):

            acc = "2"
            with torch.no_grad():
                while acc != "1" or acc !="0":

                    q = q_list[i]
                    a = ''
                    start = time.time()

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
                        if a[-1] == '.': #필터링에서 걸러주므로 쓸때없이 문장생성하는 시간을 줄임
                           break


                        if gen == EOS:
                            break
                    a = filtering(a.strip())
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
                        #num_acc_sum += result_value
                        wr.writerow([q, a_list[i], a, acc, str(end_time), '']) # ["Q", "A", "Chat_Answer","acc", "time"]
                        break

        all_num_acc_sum = num_acc_sum/len(q_list) #챗봇 답이 고른 정확도 평균 -> 소수점의 숫자가 나오므로 float으로 변환
        all_acc = num_sum/len(q_list)            #실제 정확도 평균 -> 소수점의 숫자가 나오므로 float으로 변환
        all_time = num_time_sum/len(q_list)       #평균 시간, filtering까지 포함해서
        wr.writerow(['', '', '', all_acc, all_time, all_num_acc_sum]) #실제 정확도 평균, 평균 시간, 챗봇이 답을 고른 정확도 평균
        f.close()



parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    if args.train:

        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            prefix='model_'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        model = KoGPT2Chat(args)
        model.train()
        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    if args.chat:
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        model.chat()
