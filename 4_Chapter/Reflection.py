from memory import Memory
from llm_client import HelloAgentsLLM

# 初始执⾏提示词
INITIAL_PROMPT_TEMPLATE = """
你是⼀位资深的Python程序员。请根据以下要求，编写⼀个Python函数。
你的代码必须包含完整的函数签名、⽂档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

# 反思提示词
REFLECT_PROMPT_TEMPLATE = """
你是⼀位极其严格的代码评审专家和资深算法⼯程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在⼀种<strong>算法上更优</strong>的解决⽅案来显著提升性能。
如果存在，请清晰地指出当前算法的不⾜，并提出具体的、可⾏的改进算法建议（例如，使⽤筛法替代试除法）。
如果代码在算法层⾯已经达到最优，才能回答“⽆需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""

# 优化提示词
REFINE_PROMPT_TEMPLATE = """
你是⼀位资深的Python程序员。你正在根据⼀位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上⼀轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，⽣成⼀个优化后的新版本代码。
你的代码必须包含完整的函数签名、⽂档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""


class RelfectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.memory = Memory()

    def run(self,task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        #初始执行
        print("\n--- 正在进⾏初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        #迭代循环：反思与优化
        for i in range(self.max_iterations):
            print(f"\n--- 第{i+1}/{self.max_iterations}轮迭代 ---")

            # 反思
            print("\n--- 正在进行反思 ---")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            if "⽆需改进" in feedback:
                print("代码已达到最优，无需改进。")
                break

            # 优化
            print("\n--- 正在进行优化 ---")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(task=task, last_code_attempt=last_code, feedback=feedback)
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)

        final_code = self.memory.get_last_execution()
        print(f"\n--- 最终代码 ---\n{final_code}")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages) or ""
        return response_text

if __name__ == '__main__':
    task = "编写⼀个Python函数，找出1到n之间所有的素数 (prime numbers)"
    llm_client = HelloAgentsLLM()
    agent = RelfectionAgent(llm_client)
    agent.run(task)
