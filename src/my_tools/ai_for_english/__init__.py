import openai
import os
import rich.console as rc


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
    console = rc.Console()
    with console.status("[green]AI thinking...", spinner="dots") as status:
        for chunk in completion:
            delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
            answer += delta.content
    print("AI finished thinking.")
    return answer


__all__ = ["JesusAnswer"]