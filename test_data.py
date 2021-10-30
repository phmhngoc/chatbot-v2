import json
from pyvi import ViTokenizer
from text_preprocess import text_preprocess

with open('intent1.json',encoding="utf8") as file:
  data = json.loads(file.read())

def get_dbtrain():
    db_train = []
    for intent in data:
        for pattern in intent["patterns"]:
          pattern = text_preprocess(pattern)
          db_train.append({"Question": pattern, "Intent": intent["tag"]})
          # pattern = ViTokenizer.tokenize(pattern)
          # pattern = [w for w in pattern if not w.lower() in stopwords]
          # print(pattern)
    return db_train
# def remove_stopwords(text):


get_dbtrain()
def get_dbanswers():
    db_answers = []
    for intent in data:
        db_answers.append({"Answers": intent["response"], "Intent": intent["tag"]})
    return db_answers


# def get_fallback_intent():
#     fallback_intent = ["Xin lỗi! tôi không hiểu ý của bạn, hãy nêu câu hỏi đầy đủ hơn.",
#                        "Vui lòng mô tả đầy đủ thông tin, để tôi có thể tìm câu trả lời phù hợp nhất!",
#                        "Tôi vẫn chưa hiểu được câu hỏi của bạn, vui lòng mô tả đầy đủ hơn nhé!",
#                        "Tôi chưa hiểu câu hỏi này, có thể mô tả đầy đủ thông tin hoặc tôi sẽ gửi câu hỏi này đến Phòng CSE để hỗ trợ bạn!"]
#     return fallback_intent