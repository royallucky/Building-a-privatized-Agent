import os, json, yaml, logging
from simpleeval import simple_eval
from datetime import datetime
# 假设使用 joblib 加载 sklearn 模型
import joblib

CONFIG_FILE = "functions/detect_config.yaml"
INPUT_FILE  = "cleaned_data.json"
OUTPUT_FILE = "anomalies.json"

def load_config(path, logger):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"DetectAgent: 加载配置失败 ({path})：{e}")
        return {}

def apply_rules(record, rules, logger):
    for rule in rules:
        expr = rule.get("if", "")
        then = rule.get("then", {})
        try:
            if simple_eval(expr, names=record):
                record.update(then)
        except Exception as e:
            logger.warning(f"DetectAgent: 规则执行错误 '{expr}'：{e}")
    return record

def detect_batch(batch_dir, logger):
    # 1. 加载配置和规则
    cfg = load_config(CONFIG_FILE, logger)
    mode = cfg.get("mode", "rule")
    rules = cfg.get("rules", [])
    model = None

    # 2. 如果配置为模型模式，尝试加载模型
    if mode == "model":
        model_path = cfg.get("model_path")
        try:
            model = joblib.load(model_path)
            logger.info(f"DetectAgent: 模型模式，已加载模型 {model_path}")
        except Exception as e:
            logger.warning(f"DetectAgent: 模型加载失败，回退规则模式：{e}")
            model = None

    # 3. 读取清洗后数据
    input_path = os.path.join(batch_dir, INPUT_FILE)
    try:
        with open(input_path, encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        logger.error(f"DetectAgent: 读取输入失败 ({input_path})：{e}")
        return

    anomalies = []
    # 4. 遍历记录：模型优先、规则备用
    for rec in records:
        rec_copy = rec.copy()
        if model:
            try:
                # 假设模型接口：model.predict([features]) 返回 label 和 probability
                features = [rec_copy[f] for f in cfg.get("model_features", [])]
                pred = model.predict([features])[0]
                prob = max(model.predict_proba([features])[0])
                if pred == 1:  # 假设 1 表示异常
                    rec_copy.update({
                        "anomaly_type": "模型异常",
                        "score": float(prob)
                    })
            except Exception as e:
                logger.warning(f"DetectAgent: 模型预测失败，回退规则：{e}")
                rec_copy = apply_rules(rec_copy, rules, logger)
        else:
            rec_copy = apply_rules(rec_copy, rules, logger)

        # 收集标记异常的记录
        if "anomaly_type" in rec_copy:
            anomalies.append({
                "record_id": rec_copy.get("id"),
                "anomaly_type": rec_copy["anomaly_type"],
                "score": rec_copy.get("score"),
                "detected_at": datetime.now().isoformat()
            })

    # 5. 写入结果
    output_path = os.path.join(batch_dir, OUTPUT_FILE)
    try:
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(anomalies, f, ensure_ascii=False, indent=2)
        logger.info(f"DetectAgent: 完成异常检测，共 {len(anomalies)} 条，输出至 {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"DetectAgent: 写入结果失败 ({output_path})：{e}")
