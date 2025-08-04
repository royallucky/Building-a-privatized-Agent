# report_agent.py

import os
import json
import logging
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import markdown

# --------- 配置区域 ---------
TEMPLATE_DIR = "functions/templates"
MD_TPL = "report.md.j2"
OUT_MD = "report.md"
OUT_HTML = "report.html"
OUT_PNG = "report.png"  # 若生成静态图


def report_batch(batch_dir, logger):
    # 1. 读取并统计
    raw_count = sum(1 for _ in open(os.path.join(batch_dir, "raw_data.jsonl")))
    cleaned = json.load(open(os.path.join(batch_dir, "cleaned_data.json"), encoding="utf-8"))
    anomalies = json.load(open(os.path.join(batch_dir, "anomalies.json"),  encoding="utf-8"))
    trend = json.load(open(os.path.join(batch_dir, "trend.json"),      encoding="utf-8"))
    stats = {
        "raw_count": raw_count,
        "cleaned_count": len(cleaned),
        "anomaly_count": len(anomalies),
        "anomaly_breakdown": [
            {"type": t, "count": sum(1 for a in anomalies if a["anomaly_type"]==t)}
            for t in set(a["anomaly_type"] for a in anomalies)
        ]
    }
    trend_option = {
        "legend": {"data": [t["field"] for t in trend]},
        "xAxis":  {"data": trend[0]["dates"]},
        "series": [{"name":t["field"],"type":"line","data":t["values"]} for t in trend]
    }
    # 2. 渲染 Markdown
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    tpl = env.get_template(MD_TPL)
    md  = tpl.render(
        batch_id=os.path.basename(batch_dir),
        now=datetime.now().isoformat(),
        stats=stats,
        trend_option=json.dumps(trend_option),
        chart_path=OUT_PNG
    )
    open(os.path.join(batch_dir, OUT_MD), "w", encoding="utf-8").write(md)
    logger.info("ReportAgent: 生成 report.md 完成")
    # 3. 转 HTML
    html = markdown.markdown(md, extensions=["fenced_code"])
    open(os.path.join(batch_dir, OUT_HTML), "w", encoding="utf-8").write(html)
    logger.info("ReportAgent: 生成 report.html 完成")
    # 4. （可选）静态图生成
    # generate_static_chart(trend, os.path.join(batch_dir, OUT_PNG))
    # logger.info(f"ReportAgent: 生成 {OUT_PNG} 完成")