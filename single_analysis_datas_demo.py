"""
    单次流程逻辑串联，脚本代码，工程化不作参考
"""
import os
import time
from datetime import datetime

from functions.clean_datas import clean_batch
from functions.detect_datas import detect_batch
from functions.load_datas import ingest_batch
from functions.report import report_batch
from functions.trend_datas import trend_batch
from logs.logger_config import setup_async_logger

# ---------- 配置区域 ----------
# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 在此基础上拼接 data/batches 子目录
BATCH_ROOT = os.path.join(BASE_DIR, "data", "batches")  # 所有批次的根目录
INTERVAL_SEC = 300          # 若需要循环调用，可设置为 300 秒（5 分钟）
Agent_logger = setup_async_logger("Agent_logger")
# -------------------------------

def process_one_batch():
    """
    串行执行五个 Agent 的任务链，支持断点续传与失败重试。
    一旦某阶段连续失败，则终止整个流程。
    """

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(BATCH_ROOT, f"batch_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)

    print(f"[INFO] 开始处理批次 {batch_id}")

    def safe_execute(stage_name, output_file, action_func, max_retries=2):
        full_path = os.path.join(batch_dir, output_file)

        # 如果该阶段已完成，则跳过
        if os.path.exists(full_path):
            print(f"[SKIP] {stage_name} 已完成，跳过")
            return True  # 表示该阶段成功完成（或已完成）

        # 否则尝试执行并重试
        for attempt in range(1, max_retries + 2):  # 总共尝试 max_retries+1 次
            try:
                print(f"[RUN ] 执行 {stage_name}（第 {attempt} 次尝试）...")
                action_func(batch_dir, logger=Agent_logger)
                print(f"[DONE] {stage_name} 成功")
                return True
            except Exception as e:
                print(f"[ERROR] {stage_name} 第 {attempt} 次失败: {str(e)}")
                if attempt < max_retries + 1:
                    time.sleep(1)
                else:
                    print(f"[FAIL] {stage_name} 全部重试失败，终止流程")
                    return False  # 表示失败，需中止主流程

    # Agent 串行任务链
    steps = [
        ("数据采集",   "raw_data.jsonl",   ingest_batch),
        ("数据清洗",   "cleaned.jsonl",    clean_batch),
        ("异常检测",   "anomalies.json",   detect_batch),
        ("趋势计算",   "trend.json",       trend_batch),
        ("报告生成",   "report.html",      report_batch)
    ]

    # 顺序执行所有步骤，如中途失败则中断流程
    for stage_name, output_file, func in steps:
        success = safe_execute(stage_name, output_file, func)
        if not success:
            print(f"[STOP] 批次 {batch_id} 执行中断")
            return

    print(f"[INFO] 批次 {batch_id} 全部处理完成，目录：{batch_dir}")


if __name__ == "__main__":
    # 只跑一次
    process_one_batch()

    # 如果需要定时循环执行，取消下面注释：
    # while True:
    #     process_one_batch()
    #     time.sleep(INTERVAL_SEC)
