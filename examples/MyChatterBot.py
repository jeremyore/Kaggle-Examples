# from chatterbot import ChatBot
#
# chatbot = ChatBot(
#     'Ron Obvious',
#     trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
# )
#
# # Train based on the english corpus
# # chatbot.train("chatterbot.corpus.english")
#
# # Get a response to an input statement
# print(chatbot.get_response("Hello, how are you today?"))

# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
#
# chatbot = ChatBot("myBot")
# chatbot.set_trainer(ChatterBotCorpusTrainer)
#
# # 使用中文语料库训练它
# chatbot.train("chatterbot.corpus.chinese")
# lineCounter = 1
# # 开始对话
# while True:
#     print(chatbot.get_response(input("(" + str(lineCounter) + ") user:")))
#     lineCounter += 1


from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
my_bot = ChatBot("Training demo")
my_bot.set_trainer(ListTrainer)
my_bot.train([
    "你叫什么名字？",
    "我叫ChatterBot。",
    "今天天气真好",
    "是啊，这种天气出去玩再好不过了。",
    "那你有没有想去玩的地方？",
    "我想去有山有水的地方。你呢？",
    "没钱哪都不去",
    "哈哈，这就比较尴尬了",
])
while True:
    print(my_bot.get_response(input("user:")))
