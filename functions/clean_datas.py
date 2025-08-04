
import os
import json
import yaml

# --------- 配置区域 ---------
CLEAN_CONFIG = "functions/clean_config.yaml"  # 清洗规则配置文件路径
INPUT_FILE    = "raw_data.jsonl"    # 上游生成的原始 JSONL 文件
OUTPUT_FILE   = "cleaned_data.json" # 清洗后输出文件名

def load_config(config_path, logger):
    """
    加载 YAML 配置文件，返回配置字典
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"CleanAgent: 无法加载配置 {config_path}：{e}")
        return {}

def clean_record(record, rules):
    """
    按照规则对单条记录做标准化处理：
      1. 重命名：rule.rename_from → 新字段名
      2. 填充缺失：若原值为 None，则使用 rule.fillna
      3. 裁剪异常：若配置 clip，则限制在 [min, max] 之间
      4. 单位换算：如配置 multiply，则进行乘法转换
    """
    for target_field, rule in rules.items():
        # 1. 确定源字段名
        source_field = rule.get("rename_from", target_field)
        value = record.get(source_field)

        # 2. 缺失值填充
        if value is None:
            value = rule.get("fillna")

        # 3. 异常值裁剪
        if isinstance(value, (int, float)) and "clip" in rule:
            low, high = rule["clip"]
            value = max(low, min(value, high))

        # 4. 单位换算（可选）
        if isinstance(value, (int, float)) and "multiply" in rule:
            value = value * rule["multiply"]

        # 5. 写回标准字段
        record[target_field] = value

    return record

def clean_batch(batch_dir, logger):
    """
    CleanAgent 接口函数：
      - batch_dir: 批次目录，读取 raw_data.jsonl，写入 cleaned_data.json
      - logger: 外部传入的日志记录器
    """
    # 1. 加载清洗规则
    config = load_config(CLEAN_CONFIG, logger)
    rules = config.get("fields", {})

    # 2. 读取原始数据
    raw_path = os.path.join(batch_dir, INPUT_FILE)
    try:
        with open(raw_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"CleanAgent: 无法读取 {raw_path}：{e}")
        return

    cleaned = []
    # 3. 遍历清洗
    for rec in records:
        cleaned.append(clean_record(rec.copy(), rules))

    # 4. 写入清洗结果
    out_path = os.path.join(batch_dir, OUTPUT_FILE)
    try:
        with open(out_path, 'w', encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        logger.info(f"CleanAgent: 清洗完成，{len(cleaned)} 条记录，输出 {out_path}")
    except Exception as e:
        logger.error(f"CleanAgent: 写入清洗结果失败：{e}")
