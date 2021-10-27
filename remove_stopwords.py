import json
from text_preprocess import text_preprocess
total_label = 0
vocab = {}
label_vocab = {}

with open ('test_intent.json') as f:
    data = json.loads(f.read())

with open ('vietnamese-stopwords-dash.txt', 'r', encoding='UTF-8') as f:
    stop_words = f.read()

new_pattern = []
for intent in data:
    for pattern in intent['patterns']:
        pattern=text_preprocess(pattern).split()
        test = ' '.join([s for s in pattern if not s in stop_words])
        print(test)

# with open ('intent1.json') as f:
#     data = json.loads(f.read())

# for intent in data:
#     total_label +=1
# #    lưu ý từ đầu tiên là nhãn
#     label = intent['tag']
#     if label not in label_vocab:
#         label_vocab[label] = {}
#     for pattern in intent['patterns']:
#         words=text_preprocess(pattern)
#         words = words.split()
#         for word in words[0:]:
#             label_vocab[label][word] = label_vocab[label].get(word, 0) + 1
#             if word not in vocab:
#                 # print(word)
#                 vocab[word] = set()
#             vocab[word].add(label)
# print(label_vocab)
# count = {}
# for word in vocab:
#     # print(len(vocab[word]))
#     if len(vocab[word]) == 1:
#         count[word] = min([label_vocab[x][word] for x in label_vocab])
        
# sorted_count = sorted(count, key=count.get, reverse=True)
# for word in sorted_count[:100]:
#     print('1' ,word, count[word])

