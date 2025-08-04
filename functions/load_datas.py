# ingest_agent.py

import os
import json
import requests
from datetime import datetime

# --------- 配置区域 ---------
INCOMING_DIR = "data/incoming/"            # 放置待采集 JSON 文件的文件夹
API_ENDPOINT = "https://*****"        # 替换为实际的 REST API 接口地址

def generate_batch_id():
    """
    生成批次编号：示例 "20250615_1145"
    使用当前时间确保每个批次目录唯一
    """
    return datetime.now().strftime("%Y%m%d_%H%M")

def fetch_from_files(logger):
    """
    本地文件采集
      • 只处理 .json 后缀文件
      • 读取后将文件移动到 incoming/processed/，避免重复处理
      • 将每个 JSON 对象追加到 records 列表中
    """
    records = []
    proc_dir = os.path.join(INCOMING_DIR, "processed")
    os.makedirs(proc_dir, exist_ok=True)  # 确保 processed 子目录存在

    for fn in os.listdir(INCOMING_DIR):
        if not fn.endswith(".json"):
            continue  # 跳过非 JSON 文件
        src = os.path.join(INCOMING_DIR, fn)
        try:
            with open(src, 'r', encoding='utf-8') as f:
                data = json.load(f)       # 读取 JSON
                records.append(data)      # 加入记录列表
            os.rename(src, os.path.join(proc_dir, fn))  # 移动到 processed
        except Exception as e:
            logger.warning(f"DataIngestAgent: 无法处理文件 {fn}：{e}")
    return records

def fetch_from_api(logger):
    """
    REST API 数据采集
      • 发起 HTTP GET 请求
      • 若成功返回 JSON 数组
      • 若失败打印警告并返回空列表
    """
    try:
        resp = requests.get(API_ENDPOINT, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"DataIngestAgent: API 拉取失败：{e}")
        return []

def ingest_batch(batch_dir=None, logger=None):
    """
    单次采集执行函数（Agent 接口）
      - batch_dir: 指定批次目录，若为 None 则自动生成
      - logger: 外部传入的日志记录器，用于统一输出日志

    返回 raw_data.jsonl 文件的绝对路径
    """
    if logger is None:
        raise ValueError("ingest_batch 需要传入 logger 参数")

    # 1. 生成或确认批次目录
    if batch_dir is None:
        batch_id = generate_batch_id()
        batch_dir = os.path.join("batches", f"batch_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)

    # 2. 从文件和 API 两个渠道采集数据
    file_recs = fetch_from_files(logger)  # 读取 incoming/ 下的 JSON 文件
    api_recs  = fetch_from_api(logger)    # 调用 REST API 拉取数据
    records = file_recs + api_recs

    # 3. 将所有数据按 JSONL 格式写入 raw_data.jsonl
    out_file = os.path.join(batch_dir, "raw_data.jsonl")
    with open(out_file, 'w', encoding='utf-8') as f:
        for rec in records:
            # 每行一个 JSON 对象，方便下游按行读取
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 4. 日志输出，告诉使用者本批次采集了多少条记录
    logger.info(f"DataIngestAgent 完成，批次目录：{batch_dir}，共采集 {len(records)} 条记录")

    return out_file
