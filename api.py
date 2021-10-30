# import flask
# from flask_restful import Resource, Api, reqparse
# from predict import TextClassificationPredict
# import json

# app = flask.Flask(__name__)
# api = Api(app)

# class Response(Resource):
#     def get(self):
#         parser = reqparse.RequestParser()
#         parser.add_argument('question')
#         args = parser.parse_args()
#         print(args['question'])
#         tcp = TextClassificationPredict()
#         answer = json.dumps(tcp.get_train_data("Quên mật khẩu mydtu"))
#         return {"answer": answer},200

# api.add_resource(Response, '/response')

# if __name__ == '__main__':
#     app.run()

from fastapi import FastAPI
from pydantic import BaseModel
from predict import TextClassificationPredict

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post('/')
async def receiveAnswer(question: Question):
    tcp = TextClassificationPredict()
    answer = tcp.get_train_data(question.question)
    return answer

@app.get('/')
async def home():
    return 'Đây là home'