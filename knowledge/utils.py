import os
import re
import pandas as pd
from docx import Document
import pdfplumber
from paddleocr import PaddleOCR
from typing import List, Dict, Optional, Any, Union
from logging import Logger


#  文件内容提取器
class FileExtractor:
    def __init__(self) -> None:
        self.ocr_engine: PaddleOCR = PaddleOCR(use_angle_cls=True)

    def extract_pdf(self, file_path: str) -> str:
        text: str = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text: Optional[str] = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise RuntimeError(f"提取PDF失败：{file_path}, 错误: {e}")
        return text

    def extract_word(self, file_path: str) -> str:
        try:
            doc: Document = Document(file_path)
            text: str = "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            raise RuntimeError(f"提取Word失败：{file_path}, 错误: {e}")
        return text

    def extract_csv(self, file_path: str) -> pd.DataFrame:
        try:
            if file_path.endswith(".csv"):
                df: pd.DataFrame = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            raise RuntimeError(f"提取CSV/Excel失败：{file_path}, 错误: {e}")
        return df

    def extract_image(self, file_path: str) -> str:
        try:
            result: List[Any] = self.ocr_engine.ocr(file_path)
            text_lines: List[str] = [line[1][0] for line in result[0]]
            text: str = "\n".join(text_lines)
        except Exception as e:
            raise RuntimeError(f"OCR识别失败：{file_path}, 错误: {e}")
        return text

    def extract_txt(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text: str = f.read()
        except Exception as e:
            raise RuntimeError(f"提取TXT失败：{file_path}, 错误: {e}")
        return text


#  内容清洗器
class TextCleaner:
    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\-:.,]", "", text)
        return text.strip()

    def normalize_date(self, text: str) -> str:
        text = re.sub(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日]?", r"\1-\2-\3", text)
        return text


#  数据特性分析器
class DataAnalyzer:
    def analyze_text(self, text: str, keywords: Optional[List[str]] = None) -> Dict[str, Union[int, float]]:
        if keywords is None:
            keywords = ["故障", "报警", "电压", "温度", "启动"]
        length: int = len(text)
        chinese_chars: List[str] = re.findall(r"[\u4e00-\u9fa5]", text)
        chinese_ratio: float = len(chinese_chars) / length if length else 0.0

        keyword_count: int = sum(text.count(k) for k in keywords)
        keyword_density: float = keyword_count / length if length else 0.0

        return {
            "length": length,
            "chinese_ratio": round(chinese_ratio, 3),
            "keyword_density": round(keyword_density, 3)
        }


#  知识库构建调度器
class KnowledgeBaseBuilder:
    def __init__(self, logger: Logger) -> None:
        self.extractor: FileExtractor = FileExtractor()
        self.cleaner: TextCleaner = TextCleaner()
        self.analyzer: DataAnalyzer = DataAnalyzer()
        self.logger: Logger = logger

    def process_file(self, file_path: str) -> Dict[str, Any]:
        ext: str = os.path.splitext(file_path)[-1].lower()
        text: str = ""

        try:
            if ext == ".pdf":
                text = self.extractor.extract_pdf(file_path)
            elif ext in [".docx", ".doc"]:
                text = self.extractor.extract_word(file_path)
            elif ext in [".csv", ".xlsx"]:
                df: pd.DataFrame = self.extractor.extract_csv(file_path)
                text = df.to_string()
            elif ext in [".jpg", ".jpeg", ".png"]:
                text = self.extractor.extract_image(file_path)
            elif ext == ".txt":
                text = self.extractor.extract_txt(file_path)
            else:
                self.logger.warning(f"暂不支持的文件类型：{ext}，文件：{file_path}")
                return {"file_path": file_path, "error": f"Unsupported file type: {ext}"}

            text = self.cleaner.clean_text(text)
            text = self.cleaner.normalize_date(text)

            analysis: Dict[str, Union[int, float]] = self.analyzer.analyze_text(text)

            return {
                "file_path": file_path,
                "text": text,
                "analysis": analysis
            }

        except Exception as e:
            self.logger.error(f"处理文件出错：{file_path}，错误：{e}")
            return {"file_path": file_path, "error": str(e)}

    def batch_process(self, folder_path: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path: str = os.path.join(root, file)
                self.logger.info(f"开始处理文件：{file_path}")
                result: Dict[str, Any] = self.process_file(file_path)
                results.append(result)
        return results


#  示例执行
if __name__ == "__main__":
    import logging

    # 配置 logger
    logging.basicConfig(level=logging.INFO)
    logger: Logger = logging.getLogger("KnowledgeBase")

    builder: KnowledgeBaseBuilder = KnowledgeBaseBuilder(logger)
    result_list: List[Dict[str, Any]] = builder.batch_process(r"C:\Users\Admin\Desktop\DeepSeek驱动工业设备\正文")

    for res in result_list:
        if "error" in res:
            logger.error(f"文件：{res['file_path']} 处理失败，原因：{res['error']}")
        else:
            logger.info(f"文件：{res['file_path']}")
            logger.info(f"文本长度：{res['analysis']['length']}")
            logger.info(f"中文占比：{res['analysis']['chinese_ratio']}")
            logger.info(f"专业词密度：{res['analysis']['keyword_density']}")
            logger.info(f"清洗后文本示例：{res['text'][:100]}...")
