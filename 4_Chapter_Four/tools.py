import os
from typing import Any, Dict
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

def calculate(input_str: str):
    """
    一个简单的计算工具，支持加减乘除。
    输入格式: "a, b, operator"，例如 "123, 456, +"
    """
    parts = [p.strip() for p in input_str.split(",")]
    if len(parts) != 3:
        return "错误:输入格式应为 'a, b, operator'，例如 '123, 456, +'"

    a_str, b_str, operator = parts
    try:
        a = float(a_str) if "." in a_str else int(a_str)
        b = float(b_str) if "." in b_str else int(b_str)
    except ValueError:
        return f"错误:无法解析操作数 '{a_str}' 和 '{b_str}'"

    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        if b == 0:
            return "错误:除数不能为零"
        return a / b
    else:
        return f"错误:不支持的运算符 '{operator}'，请使用 +, -, *, /"


def search(query: str) -> str:
    """
    一个基于Tavily的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案。
    """
    print(f"正在执行[Tavily] 网页搜索： {query}")
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "错误:TAVILY_API_KEY 未在 .env 文件中配置。"

        client = TavilyClient(api_key=api_key)
        results = client.search(query=query, search_depth="basic")

        if results.get("answer"):
            return results["answer"]

        if results.get("results"):
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('content', '')}"
                for i, res in enumerate(results["results"][:3])
            ]
            return "\n".join(snippets)
        return "未找到相关结果"
    except Exception as e:
        return f"搜索时发生错误: {e}"



class ToolExecutor:
    """
    ⼀个⼯具执⾏器，负责管理和执⾏⼯具。
    """
    def __init__(self):
        self.tools: Dict[str,Dict[str,Any]] = {}

    def register_tool(self,name: str, description: str, func: callable):
        """
        向⼯具箱中注册⼀个新⼯具。
        """
        if name in self.tools:
            print(f"警告:⼯具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"⼯具 '{name}' 已注册。")

    def get_tool(self,name: str) -> callable:
        return self.tools.get(name,{}).get("func")

    def get_available_tools(self) -> str:
        return "\n".join([f"- {name}: {info['description']}" for name,info in self.tools.items()])


if __name__ == '__main__':
    toolExecutor = ToolExecutor()

    search_description = "⼀个⽹⻚搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使⽤此⼯具。"
    toolExecutor.register_tool("search", search_description, search)

    print("\n--- 可⽤的⼯具 ---")
    print(toolExecutor.get_available_tools())

    print("\n--- 执⾏ Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "search"
    tool_input = "英伟达最新的GPU型号是什么"
    tool_function = toolExecutor.get_tool(tool_name)

    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的⼯具。")