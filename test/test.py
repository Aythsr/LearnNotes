import time
import rich.console as rc

# 模拟一个耗时函数
def long_running_task():
    time.sleep(3) # 假装在干活
    return "Result Data"

if __name__ == "__main__":
    print("开始执行任务...")
    console = rc.Console()
    # 使用 with 语句包裹耗时任务
    with console.status("[green]Processing...", spinner="dots") as status:
        result = long_running_task()
    print(f"✅ 主程序继续执行，拿到结果: {result}")