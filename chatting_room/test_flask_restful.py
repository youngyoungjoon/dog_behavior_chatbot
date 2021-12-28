from flask import Flask
from flask_restful import Api
import torchvision.models as models
import torchvision.transforms as transforms


app = Flask(__name__)
api = Api(app)


@app.route(r'D:\KoGPT2-chatbot-master\KoGPT2-chatbot-master\model_chp\model_-last.ckpt')
def chatbot(chat):
    try:
        return {'status': 'success',
                'request': chat
                }
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(host="localhost", port=9999, debug=True)