import openai
import os

myclient = openai.Client(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-5b7feef4d6064e3ea3e268dea23c10a7"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def JesusAnswer(messages, model: str = "qwen-plus-2025-07-14"):
    completion = myclient.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"enable_thinking": True},
        stream=True
    )  # type: ignore[arg-type]
    is_answering = False  # 是否进入回复阶段
    answer = ""
    print("AI thinking", end="", flush=True)
    # print("\n" + "=" * 20 + "思考过程" + "=" * 20)
    for chunk in completion:
        delta = chunk.choices[0].delta
        # if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        #     if not is_answering:
        #         print(delta.reasoning_content, end="", flush=True)
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
                # print("\n" + "=" * 20 + "回答内容" + "=" * 20)
            # print(delta.content, end="", flush=True)
            answer += delta.content
        print(".", end="", flush=True)
    # print("\n" + "=" * 50)
    print()
    return answer

# print("MyAI client initialized.")

__all__ = ["JesusAnswer"]