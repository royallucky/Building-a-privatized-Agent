# Chroma 存储体
"""
    | 方法                               | 作用
    | :-------------------------------- | :------------------------------------- |
    | `add_texts(texts, metadatas)`     | 添加文本和元数据                               |
    | `similarity_search(query, k)`     | 相似度检索（返回 Document 列表）                  |
    | `similarity_search_with_score()`  | 相似度检索 + 分数                             |
    | `max_marginal_relevance_search()` | 基于 MMR（最大边际相关性）检索，防止结果过于相似      |
    | `as_retriever()`                  | 将当前 vector\_db 转换成 LangChain Retriever |
    | `delete(ids=None)`                | 删除指定 id 的文档                            |
    | `delete_collection()`             | 删除整个 collection                        |
    | `get(where=None)`                 | 查询 collection 内全部或根据 metadata 条件检索     |
    | `persist()`                       | 将内存中的数据保存到 persist\_directory          |
    | `_collection.count()`             | 当前 collection 的向量数量                    |
    | `get(include=["embeddings"])`     | 查询数据+返回 embedding 向量                   |
"""
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from .base_storage import BaseStorage
from .utils import storage_logger
import os
import pickle


class ChromaStorage(BaseStorage):
    def __init__(self, persist_directory, collection_name="default", model_name="llama3"):
        """初始化 Chroma 向量库"""
        try:
            self.embedding_function = OllamaEmbeddings(model=model_name)
            self.persist_directory = persist_directory
            self.collection_name = collection_name
            self.vector_db = Chroma(
                persist_directory=persist_directory,            # 本地持久化路径，设置后会将数据保存到本地目录，重启后自动加载
                collection_name=collection_name,                # 向量库集合名（相当于表名，多个 Agent/项目可用不同 collection 隔离）
                embedding_function=self.embedding_function,     # Embedding 模型对象，比如 OllamaEmbeddings、HuggingFaceEmbeddings 等
                client_settings = None,                         # 可选，自定义 Chroma Server 或 Client 的配置
                collection_metadata = None,                     # 可选，给 collection 附带自定义元数据，如 {"description": "Agent1的记忆库"}
                client = None,                                  # 可选，外部已创建好的 chromadb.Client 实例
                relevance_score_fn = None                       # 可选，自定义相似度分数计算方法，覆盖默认内置
            )
            storage_logger.info(f"Chroma collection [{collection_name}] 初始化成功")
        except Exception as e:
            storage_logger.exception(f"初始化异常: {e}")

    def add_texts(self, texts, meta_datas=None):
        """
        添加文本和元数据，返回生成的 id 列表

        参数：
            texts (List[str])         ：要添加的文本列表
            meta_datas (List[dict])   ：对应文本的元数据列表，可选
        """
        try:
            ids = self.vector_db.add_texts(texts=texts, metadatas=meta_datas)
            storage_logger.info(f"新增 {len(ids)} 条文本，ID：{ids}")
            return ids
        except Exception as e:
            storage_logger.exception(f"添加文本异常: {e}")
            return []

    def similarity_search(self, query, k=5, score_threshold=None):
        """
        相似度检索

        参数：
            query (str)               ：查询文本
            k (int)                   ：返回前 k 条相似度最高的结果，默认 5
            score_threshold (float)   ：相似度分数阈值，若设置则只返回大于等于该阈值的结果
        """
        try:
            results = self.vector_db.similarity_search_with_score(query, k=k)
            if score_threshold:
                results = [r for r in results if r[1] >= score_threshold]
            return results
        except Exception as e:
            storage_logger.exception(f"相似度检索异常: {e}")
            return []

    def delete_collection(self):
        """
        删除当前 collection
        （无参数）
        """
        try:
            self.vector_db.delete_collection()
            storage_logger.info(f"删除 collection [{self.collection_name}]")
        except Exception as e:
            storage_logger.exception(f"删除 collection 异常: {e}")

    def list_collections(self):
        """
        列出所有 collection

        返回：
            List[str] ：persist_directory 下的所有 collection 名称列表
        """
        try:
            return os.listdir(self.persist_directory)
        except Exception as e:
            storage_logger.exception(f"列出 collection 异常: {e}")
            return []

    def delete_by_ids(self, ids):
        """
        根据 ID 删除

        参数：
            ids (List[str]) ：要删除的向量 ID 列表
        """
        try:
            self.vector_db.delete(ids)
            storage_logger.info(f"删除向量 ID: {ids}")
        except Exception as e:
            storage_logger.exception(f"删除 ID 异常: {e}")

    def query_by_metadata(self, metadata_key, metadata_value):
        """
        metadata 查询

        参数：
            metadata_key (str)   ：要查询的 metadata 字段名
            metadata_value (any) ：对应的 metadata 值

        返回：
            dict ：符合条件的文档数据
        """
        try:
            return self.vector_db.get(where={metadata_key: metadata_value})
        except Exception as e:
            storage_logger.exception(f"metadata 查询异常: {e}")
            return []

    def update_text(self, id, new_text, new_metadata=None):
        """
        更新文本/metadata

        参数：
            id (str)                 ：要更新的向量 ID
            new_text (str)           ：新文本内容
            new_metadata (dict)      ：新元数据，可选
        """
        try:
            self.delete_by_ids([id])
            self.add_texts([new_text], [new_metadata])
            storage_logger.info(f"更新 ID: {id}")
        except Exception as e:
            storage_logger.exception(f"更新文本异常: {e}")

    def export_data(self, file_path):
        """
        导出数据

        参数：
            file_path (str) ：导出文件路径（.pkl）
        """
        try:
            data = self.vector_db.get()
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            storage_logger.info(f"导出数据到 {file_path}")
        except Exception as e:
            storage_logger.exception(f"导出异常: {e}")

    def import_data(self, file_path):
        """
        导入数据

        参数：
            file_path (str) ：导入文件路径（.pkl）
        """
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            self.add_texts(data["documents"], data.get("meta_datas"))
            storage_logger.info(f"导入数据 {len(data['documents'])} 条")
        except Exception as e:
            storage_logger.exception(f"导入异常: {e}")

    def count(self):
        """
        数量统计

        返回：
            int ：当前 collection 中的向量数量
        """
        try:
            return self.vector_db._collection.count()
        except Exception as e:
            storage_logger.exception(f"计数异常: {e}")
            return 0

    def clear_collection(self):
        """
        清空 collection
        （无参数）
        """
        try:
            self.vector_db.delete()
            storage_logger.info(f"清空 collection [{self.collection_name}]")
        except Exception as e:
            storage_logger.exception(f"清空异常: {e}")

    def mmr_search(self, query, k=5, lambda_mult=0.5):
        """
        基于最大边际相关性（MMR）检索

        参数：
            query (str)             ：查询文本
            k (int)                 ：返回前 k 条结果，默认 5
            lambda_mult (float)     ：多样性控制系数，0-1，越高多样性越大，默认 0.5

        返回：
            List[Document] ：符合条件的文档列表
        """
        try:
            results = self.vector_db.max_marginal_relevance_search(query, k=k, lambda_mult=lambda_mult)
            storage_logger.info(f"MMR 检索 {len(results)} 条")
            return results
        except Exception as e:
            storage_logger.exception(f"MMR 检索异常: {e}")
            return []

    def as_retriever(self, search_type="similarity", search_kwargs=None):
        """
        将 vector_db 转换成 LangChain Retriever

        参数：
            search_type (str)         ：检索类型，可选 'similarity' / 'mmr'，默认 'similarity'
            search_kwargs (dict)      ：检索参数，可选

        返回：
            BaseRetriever ：Retriever 实例
        """
        try:
            retriever = self.vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs or {})
            storage_logger.info(f"生成 Retriever 成功，类型：{search_type}")
            return retriever
        except Exception as e:
            storage_logger.exception(f"生成 Retriever 异常: {e}")
            return None

    def persist(self):
        """
        将内存数据保存到本地 persist_directory
        【建议在 add / delete / update 后手动 persist，一般放在外部业务逻辑里统一触发】

        （无参数）
        """
        try:
            self.vector_db.persist()
            storage_logger.info(f"持久化 collection [{self.collection_name}] 成功")
        except Exception as e:
            storage_logger.exception(f"持久化异常: {e}")

    def get_with_embeddings(self, where=None):
        """
        查询数据及其 embedding 向量

        参数：
            where (dict) ：筛选条件，键值对形式，None 则查询全部

        返回：
            dict ：包含 id、documents、embeddings 的数据字典
        """
        try:
            data = self.vector_db.get(where=where, include=["embeddings"])
            storage_logger.info(f"查询包含 embedding 数据，条数：{len(data.get('ids', []))}")
            return data
        except Exception as e:
            storage_logger.exception(f"查询 embedding 异常: {e}")
            return {}

    def save_cleaned_knowledge(self, cleaned_texts, meta_datas=None):
        """
        将清洗后的知识文本送入数据库

        参数：
            cleaned_texts (List[str])    ：清洗后的文本列表
            meta_datas (List[dict])      ：对应的元数据列表，可选

        返回：
            List[str] ：添加后的向量 ID 列表
        """
        try:
            # 添加文本到向量库
            ids = self.add_texts(cleaned_texts, meta_datas)
            # 持久化保存
            self.persist()
            storage_logger.info(f"成功保存清洗后的知识 {len(ids)} 条，ID：{ids}")
            return ids
        except Exception as e:
            storage_logger.exception(f"保存清洗知识异常: {e}")
            return []


