import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search, calculate

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是⼀个有能⼒调⽤外部⼯具的智能助⼿。

可⽤⼯具如下:
{tools}

请严格按照以下格式进⾏回应:

Thought: 你的思考过程，⽤于分析问题、拆解任务和规划下⼀步⾏动。
Action: 你决定采取的⾏动，必须是以下格式之⼀:
- `{{tool_name}}[{{tool_input}}]`:调⽤⼀个可⽤⼯具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到⾜够的信息，能够回答⽤户的最终问题时，你必须在Action:字段后使⽤ Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM,
                 tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.get_available_tools()
            history_str = "\n".join(self.history)

            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应")
                break

            thought, action = self._parse_output(response_text)

            if thought:
                print(f"思考: {thought}")
            if not action:
                print("错误:未解析到有效 Action")
                break

            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                continue

            print(f"调⽤工具: {tool_name}({tool_input})")

            tool_function = self.tool_executor.get_tool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的⼯具。"
            else:
                observation = tool_function(tool_input)

            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        print("已达到最大步数，流程终止")
        return None

    # 负责从LLM的完整响应中分离出 Thought 和 Action 两个主要部分。
    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到⽂本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    # 负责进⼀步解析 Action 字符串,提取工具调用或 Finish 指令。
    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""


if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.register_tool("search", search_desc, search)

    calculate_desc = "一个简单的计算工具，输入格式为 'a, b, operator'（用逗号分隔），例如 '123, 456, +'。支持加减乘除运算。operator支持+,-,*,/。"
    tool_executor.register_tool("calculate", calculate_desc, calculate)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "(123 +456) × 789/ 12 ="
    agent.run(question)