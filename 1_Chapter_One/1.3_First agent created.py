import requests
import os
from tavily import TavilyClient
from openai import OpenAI
import re


def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 获取请求链接
        response = requests.get(url)
        # 检查响应状态码200-299静默通过，400~599抛出HTTPError
        response.raise_for_status()
        # 获取json格式数据
        data = response.json()

        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]
        temp_c = current_condition['temp_C']

        return f"{city}当前天气:{weather_desc},气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量"

    # 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    query = f"'{city}' 在'{weather}'天⽓下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        # response["answer"]是基于所有结果的总结性回答
        if response.get("answer"):
            return response["answer"]
        # 如果没有总结,格式化输出原始结果
        formatted_results = []
        # 如何result存在返回相关值，否则默认返回空列表，避免KeyError
        for result in response.get("result", []):
            formatted_results.append(f"-{result['title']}:{result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        # "\n".join(formatted_results)，字符串拼接，在每个列表元素之间插入一个换行符
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执⾏Tavily搜索时出现问题 - {e}"


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}


# 接入大语言模型
class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("正在调用大语言模型")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误：{e}")
            return "错误调用语音服务模型出错"


# 配置LLM客户端
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_ID = "deepseek-r1"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
AGENT_SYSTEM_PROMPT = """
你是⼀个智能旅⾏助⼿。你的任务是分析⽤户的请求，并使⽤可⽤⼯具⼀步步地解决问题。

# 可⽤⼯具:
- `get_weather(city: str)`: 查询指定城市的实时天⽓。
- `get_attraction(city: str, weather: str)`: 根据城市和天⽓搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含⼀对Thought和Action：

Thought: [你的思考过程和下⼀步计划]
Action: [你要执⾏的具体⾏动]

Action的格式必须是以下之⼀：
1. 调⽤⼯具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出⼀对Thought-Action
- Action必须在同⼀⾏，不要换⾏
- 当收集到⾜够信息可以回答⽤户问题时，必须使⽤ Action: Finish[最终答案] 格式结束
请开始吧！
"""

llm = OpenAICompatibleClient(model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL)

# 初始化
user_prompt = "你好，请帮我查询⼀下今天北京的天⽓，然后根据天⽓推荐⼀个合适的旅游景点。"
prompt_history = [f"⽤户请求: {user_prompt}"]

print(f"用户输入:{user_prompt}\n" + "=" * 40)

# 运⾏主循环
for i in range(5):  # 设置最⼤循环次数
    print(f"--- 循环 {i + 1} --- \n")

    full_prompt = "\n".join(prompt_history)

    # 调用LLM进行思考
    print(full_prompt)
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)

    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    # 解析并行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)

    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循'Thought: ... Action: ...'的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break

    #获取工具名
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    #获取参数字符串
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    #参数字符串解析转换为字典
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的⼯具 '{tool_name}'"

    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)
