from my_tools.ai_for_english import JesusAnswer

def prePropmt():
    message = [
        {
            "role": "system",
            "content": "你是一个AI prompt生成助手，帮助用户生成高质量的AI提示词。你需要理解用户的需求，并生成符合要求的提示词。输出的内容必须为英文，并且只包含提示词本身，不包含任何多余的解释或说明。",
        },
        {
            "role": "user",
            "content": """prompt如下：“主题为学习英语的提示词，你将会收到一些单词，这些单词具有一定的联系，通常是一个单词的多种变形。请你依据牛津词典，解释不同单词的含义区别，并给出对应的牛津释义与中文释义，给出例句与翻译，最终生成一篇学习手册文章。最终的生成文本中只使用markdown格式。并且不要包含任何多余的解释或说明。”"""
        }
    ]
    return message

def get_prompt():
    messages = prePropmt()
    print("="*10,"Generating prompt","="*10)
    Prompt = JesusAnswer(messages)
    return Prompt + " Answer in Chinese."

def gen_ai_answer(word_list:  list[str]):
    word_str = ", ".join(word_list)
    messages = [
        {
            "role": "system",
            "content": get_prompt(),
        },
        {
            "role": "user",
            "content": word_str
        },
    ]
    print("="*10,"Generating answer","="*10)
    answer = JesusAnswer(messages)
    return answer