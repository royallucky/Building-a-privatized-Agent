# trend_agent.py

import os
import json
import yaml
import pandas as pd
from datetime import datetime, timedelta

# --------- 配置区域 ---------
CONFIG_NAME  = "functions/trend_config.yaml"  # 配置文件名，应放在批次目录或项目根
HIST_ROOT    = "batches/"           # 历史批次根目录，TrendAgent 会在此读取 cleaned_data.json
OUTPUT_FILE  = "trend.json"         # 本批次输出文件名

def load_config(config_path, logger):
    """
    加载 YAML 配置，返回需要计算的字段列表和日期字段名
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        fields = cfg.get("fields", [])
        date_field = cfg.get("date_field", "timestamp")
        return fields, date_field
    except Exception as e:
        logger.error(f"TrendAgent: 无法加载配置 {config_path}：{e}")
        return [], "timestamp"

def collect_past_data(batch_dir, days, date_field, logger):
    """
    收集最近 days 天的清洗数据：
      1. 从 HIST_ROOT/batch_<timestamp>/cleaned_data.json 中读取历史数据
      2. 再读取当前批次目录下的 cleaned_data.json
      3. 将所有读取的 DataFrame 合并并返回
    """
    dfs = []
    now = datetime.now()
    for i in range(1, days + 1):
        dt = (now - timedelta(days=i)).strftime("%Y%m%d_%H%M")
        path = os.path.join(HIST_ROOT, f"batch_{dt}", "cleaned_data.json")
        if os.path.exists(path):
            try:
                df = pd.read_json(path, convert_dates=[date_field])
                dfs.append(df)
            except Exception as e:
                logger.warning(f"TrendAgent: 读取历史文件失败 {path}：{e}")
    # 读取当前批次
    curr_path = os.path.join(batch_dir, "cleaned_data.json")
    if os.path.exists(curr_path):
        try:
            curr_df = pd.read_json(curr_path, convert_dates=[date_field])
            dfs.append(curr_df)
        except Exception as e:
            logger.warning(f"TrendAgent: 读取当前批次文件失败 {curr_path}：{e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def trend_batch(batch_dir, logger, days=7):
    """
    TrendAgent 接口：
      - batch_dir: 当前批次目录
      - logger: 日志记录器
      - days: 滑动窗口天数
    步骤：
      1. 加载配置
      2. 收集合并历史及当前批次数据
      3. 按日期分组计算 7 天滑动平均
      4. 输出折线图数据到 trend.json
    """
    # 1. 加载配置
    fields, date_field = load_config(CONFIG_NAME, logger)
    if not fields:
        logger.warning("TrendAgent: 无需计算字段，退出")
        return

    # 2. 收集数据
    data = collect_past_data(batch_dir, days, date_field, logger)
    if data.empty:
        logger.warning("TrendAgent: 无可用数据，跳过")
        return

    # 转换为仅含日期的列，方便按天统计
    data["date"] = pd.to_datetime(data[date_field]).dt.date

    result = []
    # 3. 计算滑动平均
    for fld in fields:
        try:
            daily = data.groupby("date")[fld].mean().sort_index()
            rolling = daily.rolling(window=days, min_periods=1).mean()
            result.append({
                "field": fld,
                "dates": [d.isoformat() for d in rolling.index],
                "values": [round(v, 3) for v in rolling.values]
            })
        except Exception as e:
            logger.warning(f"TrendAgent: 计算字段 {fld} 出错：{e}")

    # 4. 写入输出
    out_path = os.path.join(batch_dir, OUTPUT_FILE)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"TrendAgent: 趋势计算完成，输出 {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"TrendAgent: 写入趋势结果失败 {out_path}：{e}")

