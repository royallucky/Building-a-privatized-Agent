# 存储接口基本规范
from abc import ABC, abstractmethod

class BaseStorage(ABC):
    @abstractmethod
    def add_texts(self, texts, meta_datas=None):
        """将文本数据及元信息写入向量数据库"""
        pass

    @abstractmethod
    def similarity_search(self, query, k=5, score_threshold=None):
        """基于向量相似度检索文本

        Args:
            query (str): 查询文本
            k (int): 返回前 k 个相似文本
            score_threshold (float, optional): 置信度阈值，过滤低于该值的结果

        Returns:
            list: 相似文本内容及其相关信息
        """
        pass

    @abstractmethod
    def delete_collection(self):
        """删除当前 collection"""
        pass

    @abstractmethod
    def list_collections(self):
        """列出本地存储目录下所有 collection 名称"""
        pass

    @abstractmethod
    def delete_by_ids(self, ids):
        """根据 ID 删除特定向量数据"""
        pass

    @abstractmethod
    def query_by_metadata(self, metadata_key, metadata_value):
        """根据 metadata 查询匹配数据

        Args:
            metadata_key (str): metadata 字段名
            metadata_value (str): metadata 值

        Returns:
            list: 符合条件的文本
        """
        pass

    @abstractmethod
    def update_text(self, id, new_text, new_metadata=None):
        """根据 ID 更新文本和/或 metadata"""
        pass

    @abstractmethod
    def export_data(self, file_path):
        """导出当前 collection 数据到本地文件"""
        pass

    @abstractmethod
    def import_data(self, file_path):
        """导入本地文件数据到当前 collection"""
        pass

    @abstractmethod
    def count(self):
        """返回当前 collection 中的向量数量"""
        pass

    @abstractmethod
    def clear_collection(self):
        """清空当前 collection 中的所有数据，但保留 collection"""
        pass
