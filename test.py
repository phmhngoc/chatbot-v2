from pyvi import ViTokenizer
from text_preprocess import text_preprocess, convert_unicode
# stop_words = 'a_lô'
# with open('vietnamese-stopwords-dash.txt', encoding='utf8') as f:
#     for line in f:
#         print(line)
#         if(convert_unicode(stop_words) == convert_unicode(line.split("\n")[0])):
#          break
# f = open('vietnamese-stopwords-dash.txt', "r")

sentence = ViTokenizer.tokenize('Vì sao không nghe được người khác nói trong zoom').lower().split(" ")
# print(sentence)
# for word in sentence:
#     for stop_word in stop_words:
#         if(word != stop_word):
#             print(word)
# print(sentence)
# print(ViTokenizer.tokenize("Vì sao không nghe được người khác nói trong zoom"))
# print('vì_sao' == 'vì_sao')
print(text_preprocess("Vì sao không nghe được người khác nói trong zoom"))