
import time, hashlib, traceback

class Agent:
    def __init__(self, agent_id, name="UnnamedAgent", capabilities=None, config=None):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities or []
        self.config = config or {}
        self.is_registered = False

    def register(self, registry):
        if not self.is_registered:
            registry.register_agent(self)
            self.is_registered = True

    def get_extra_feedback(self, task_context):
        """
        子类可重写：返回dict，采集本业务特有的反馈字段。
        """
        return {}

    def execute(self, task_context, stop_event=None, user_score=None):
        task_id = getattr(task_context, "task_id", None)
        payload = getattr(task_context, "payload", "")
        payload_str = str(payload)

        # 构造可追踪摘要：人类可读的内容 + hash 签名
        input_hash = hashlib.md5(payload_str.encode("utf-8")).hexdigest()[:8]
        # 【注意】：此处简化了对输入内容进行摘要总结的功能，实际工程中需要结合实际需求重新实现这个步骤
        readable_input = payload_str[:80].replace("\n", " ").replace("\r", "")
        input_summary = f"[{input_hash}] {readable_input}..."

        start_time = time.time()
        task_context.start_time = start_time

        success = True
        error_type = None
        traceback_str = None

        try:
            task_context.log(f"{self.name} 开始执行任务: {input_summary}")
            timeout = self.config.get("timeout", 5)

            # 模拟执行过程（支持中断）
            for _ in range(timeout * 2):
                if stop_event and stop_event.is_set():
                    raise InterruptedError("任务被外部中断")
                time.sleep(0.1)

            # 任务成功返回输出
            task_context.outputs = f"{self.name} 处理完成输出"
            task_context.status = "success"
            task_context.log(f"{self.name} 任务完成: 用时 {round(time.time() - start_time, 2)} 秒")

        except Exception as e:
            success = False
            error_type = type(e).__name__
            traceback_str = traceback.format_exc()
            task_context.status = "failed"
            task_context.outputs = None
            task_context.log(
                f"{self.name} 任务异常: {error_type}\n{traceback_str.strip().splitlines()[-1]}"
            )

        end_time = time.time()
        task_context.end_time = end_time
        duration = round(end_time - start_time, 3)

        # ==== 采集结构化反馈 ====
        result_record = {
            "task_id": task_id,
            "agent": self.name,
            "input_hash": input_hash,
            "readable_input": readable_input,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "status": task_context.status,
            "error_type": error_type,
            "traceback": traceback_str,
            "user_score": user_score,
            "outputs": getattr(task_context, "outputs", None),
        }

        # 合并子类自定义采集
        result_record.update(self.get_extra_feedback(task_context))
        task_context.log(f"[采集结果] {result_record}")

        return task_context

# 子类Agent举例：ReportAgent
class ReportAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="data-003",
            name="ReportAgent",
            capabilities=["generate_report"],
            config={"timeout": 5}
        )

    def get_extra_feedback(self, task_context):
        # 模拟采集“PDF大小、生成图片数” 【注意，这些数据是要实实在在在Agent运行过程中，把数据传递到上下文中的，否则此处获取不到对应的数据】
        pdf_size = getattr(task_context, "pdf_size", None)
        num_figures = getattr(task_context, "num_figures", None)
        return {
            "pdf_size": pdf_size,
            "num_figures": num_figures
        }

    def execute(self, task_context, stop_event=None, user_score=None):
        # 模拟生成数据（此段代码的意思是在原基类已实现的功能上进行额外的添加）
        task_context["pdf_size"] = 100
        task_context["num_figures"] = 2000

""" 其他重复的内容省略 """