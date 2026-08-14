import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class HelloAgentsLLM:
    def __init__ (self, model: str = None,apiKey: str = None,baseUrl: str = None,timeout: int = 60):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.apiKey = apiKey or os.getenv("LLM_API_KEY")
        self.baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT"), 60)

        if not all([self.model,self.apiKey,self.baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env⽂件中定义")

        self.client = OpenAI(
            api_key=self.apiKey,
            base_url=self.baseUrl,
            timeout=self.timeout
        )

    def think(self,messages:List[Dict[str,str]], temperature: float = 0) -> str:
        print(f"正在调用{self.model}模型")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            print("大语言模型响应成功")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="",flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)

        except Exception as e:
            print(f"调用大语言模型时出错: {e}")
            return None

if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()

        exampleMessage = [
            {"role": "system", "content": "你是一个专业的Python开发人员，擅长编写Python代码"},
            {"role": "user", "content": "写⼀个快速排序算法"}
        ]

        print("---调用LLM---")
        responseText = llmClient.think(exampleMessage)
        if responseText:
            print("---完整模型响应---")
            print(responseText)
    except ValueError as e:
        print(f"初始化LLM客户端时出错: {e}")
