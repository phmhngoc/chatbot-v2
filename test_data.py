import json
from pyvi import ViTokenizer

with open('test_intent.json') as file:
  data = json.loads(file.read())

with open("vietnamese-stopwords-dash.txt") as file1:
  stopwords = file1.read()

def get_dbtrain():
    db_train = []
    for intent in data["intent"]:
        for pattern in intent["patterns"]:
            db_train.append({"Question": pattern, "Intent": intent["tag"]})
            pattern = ViTokenizer.tokenize(pattern)
            # pattern = [w for w in pattern if not w.lower() in stopwords]
            print(pattern)
    # return db_train
# def remove_stopwords(text):


get_dbtrain()
# def get_dbanswers():
#     db_answers = []
#     for intent in data["intents"]:
#         db_answers.append({"Answers": intent["response"], "Intent": intent["tag"]})
#     return db_answers


# def get_fallback_intent():
#     fallback_intent = ["Xin lỗi! tôi không hiểu ý của bạn, hãy nêu câu hỏi đầy đủ hơn.",
#                        "Vui lòng mô tả đầy đủ thông tin, để tôi có thể tìm câu trả lời phù hợp nhất!",
#                        "Tôi vẫn chưa hiểu được câu hỏi của bạn, vui lòng mô tả đầy đủ hơn nhé!",
#                        "Tôi chưa hiểu câu hỏi này, có thể mô tả đầy đủ thông tin hoặc tôi sẽ gửi câu hỏi này đến Phòng CSE để hỗ trợ bạn!"]
#     return fallback_intent