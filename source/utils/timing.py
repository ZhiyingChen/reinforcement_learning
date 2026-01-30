import time
import logging
import functools

# 统一拿到命名 logger（与 LoggerManager 内一致）
def _get_logger():
    return logging.getLogger("RLLogger")

# 任务与时间的简单汇总（你已有）
tasks = []

def add_task(task_name: str, time_taken: float):
    global tasks
    tasks.append((task_name, time_taken))

def record_time_decorator(task_name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            st = time.time()
            result = func(*args, **kwargs)  # 调用原函数
            ed = time.time()
            total_time = round(ed - st, 4)
            current_task_name = task_name
            _get_logger().info("%s running time: %s seconds", current_task_name, total_time)  # ← 用命名 logger
            add_task(task_name=current_task_name, time_taken=total_time)
            return result
        return wrapper
    return decorator

def out_profile(output_folder: str):
    with open(f"{output_folder}time_profile.txt", "w", encoding="utf-8") as file:
        for task, time_taken in tasks:
            file.write(f"{task}: {time_taken}\n")
