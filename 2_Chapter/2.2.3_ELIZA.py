import re
import random

context = {}

# 定义规则库:模式(正则表达式) -> 响应模板列表
rules = {
    r'My name is (.*)': [  # 专门捕获姓名
        "Nice to meet you, {0}!",
        "Hello {0}, how can I help you?",
        "I will remember your name, {0}."
    ],
    r'I am (\d+) years old': [  # 捕获年龄
        "Being {0} years old is a great age.",
        "I see you are {0} years old."
    ],
    r'I work as a (.*)': [  # 捕获职业
        "A {0}! That sounds interesting.",
        "Tell me about your work as a {0}."
    ],
    r'I need (.*)': [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?"
    ],
    r'Why don\'t you (.*)\?': [
        "Do you really think I don't {0}?",
        "Perhaps eventually I will {0}.",
        "Do you really want me to {0}?"
    ],
    r'Why can\'t I (.*)\?': [
        "Do you think you should be able to {0}?",
        "If you could {0}, what would you do?",
        "I don't know -- why can't you {0}?"
    ],
    r'I am (.*)': [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?"
    ],
    r'.* mother .*': [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?"
    ],
    r'.* father .*': [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you?"
    ],
    r'.* work .*': [
        "What do you do for a living?",
        "What is your job?",
        "What do you do for a living?"
    ],
    r'.* study .*': [
        "What do you study?",
        "What is your major?",
        "What do you study?"
    ],

    r'.*': [
        "Please tell me more.",
        "Let's change focus a bit... Tell me about your family.",
        "Can you elaborate on that?"
    ]
}

# 定义代词转换规则
pronoun_swap = {
    "i": "you", "you": "i", "me": "you", "my": "your",
    "am": "are", "are": "am", "was": "were", "i'd": "you would",
    "i've": "you have", "i'll": "you will", "yours": "mine",
    "mine": "yours"
}


def update_context(user_input):
    name_match = re.search(r'(?:My name is|I am called|Call me) (\w+)', user_input, re.IGNORECASE)
    if name_match:
        context['name'] = name_match.group(1).capitalize()
    age_match = re.search(r'I am (\d+) years? old', user_input, re.IGNORECASE)
    if age_match:
        context['age'] = age_match.group(1)
    job_match = re.search(r'(?:I work as a|I am a|My job is) (\w+)', user_input, re.IGNORECASE)
    if job_match:
        context['job'] = job_match.group(1)


def inject_context(response):
    if context:
        mentions = []
        if 'name' in context:
            mentions.append(f"By the way, I remember your name is {context['name']}.")
        if 'age' in context:
            mentions.append(f"You told me you are {context['age']} years old.")
        if 'job' in context:
            mentions.append(f"I recall you work as a {context['job']}.")
        if mentions:
            response += " " + random.choice(mentions)
    return response



def swap_pronouns(phrase):
    words = phrase.lower().split()
    swapped_words = [pronoun_swap.get(word, word) for word in words]
    return " ".join(swapped_words)


def respond(user_input):
    update_context(user_input)

    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            captured_group = match.group(1) if match.groups() else ""
            swapped_group = swap_pronouns(captured_group)
            response = random.choice(responses).format(swapped_group)

            response = inject_context(response)
            return response

    return inject_context(random.choice(rules['.*']))


# 主循环
if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Therapist: Goodbye. It was nice talking to you.")
            break
        response = respond(user_input)
        print(f"Therapist: {response}")
