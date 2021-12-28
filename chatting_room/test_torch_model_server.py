# flask_server.py

import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request

from train_torch import KoGPT2Chat

import argparse
import logging

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')


parser = KoGPT2Chat.add_model_specific_args(parser)
args = parser.parse_args()
logging.info(args)

model = KoGPT2Chat(args)
model.load_state_dict(torch.load(r'D:\KoGPT2-chatbot-master\KoGPT2-chatbot-master\model_chp\model_-last.ckpt'), strict=False)
model.eval()

app = Flask(__name__)
@app.route('/inference', methods=['POST'])
def inference():

    result = model.chat()
    return result


if __name__ == '__main__':
    app.run(host='localhost', port=9999, threaded=False)