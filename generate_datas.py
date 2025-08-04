import os
import json
import random
from datetime import datetime, timedelta

# -------------- 配置区域 --------------
INCOMING_DIR = "incoming/"  # 待采集的原始数据目录
HIST_ROOT = "batches/"  # 历史批次根目录，供 TrendAgent 读取


# 生成 incoming/ 目录下的测试记录
def gen_incoming(n=10):
    """
    生成 n 条随机测试记录，保存为 JSON 文件
    每条记录包含：id, timestamp, temperature, pressure, status, yield, anomaly_rate
    """
    os.makedirs(INCOMING_DIR, exist_ok=True)
    for i in range(n):
        rec = {
            "id": i + 1,
            "timestamp": datetime.now().isoformat(),
            "temperature": round(random.uniform(20, 120), 2),
            "pressure": round(random.uniform(0, 10), 2),
            "status": random.choice(["OK", "WARN", "ERROR"]),
            "yield": round(random.uniform(90, 100), 2),
            "anomaly_rate": round(random.uniform(0, 5), 2)
        }
        fname = os.path.join(INCOMING_DIR, f"record_{i + 1}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已生成 {n} 条 incoming 测试记录")


# 生成历史 cleaned_data.json，供 TrendAgent 计算滑动趋势
def gen_history(days=7, per_day=5):
    """
    为过去 days 天生成测试历史数据
    每天生成 per_day 条清洗后记录，保存到对应批次目录的 cleaned_data.json
    """
    for i in range(1, days + 1):
        dt = datetime.now() - timedelta(days=i)
        batch_id = dt.strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(HIST_ROOT, f"batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)

        records = []
        for j in range(per_day):
            ts = dt + timedelta(minutes=j * (1440 // per_day))
            records.append({
                "id": j + 1,
                "timestamp": ts.isoformat(),
                "temperature": round(random.uniform(20, 100), 2),
                "pressure": round(random.uniform(0.5, 5), 2),
                "status": "OK",
                "yield": round(random.uniform(92, 98), 2),
                "anomaly_rate": round(random.uniform(0, 2), 2)
            })

        out_file = os.path.join(batch_dir, "cleaned_data.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已生成过去 {days} 天的历史 cleaned_data.json，每天 {per_day} 条记录")


if __name__ == "__main__":
    # 生成 incoming 数据和 7 天历史数据
    gen_incoming(n=1431)
    gen_history(days=7, per_day=24)
