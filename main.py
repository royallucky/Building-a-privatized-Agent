from storage.chroma_storage import ChromaStorage
from config import config


def main():
    storage = ChromaStorage(
        persist_directory=config.PERSIST_DIRECTORY,
        collection_name=config.DEFAULT_COLLECTION_NAME,
        model_name=config.MODEL_NAME
    )

    # 添加内容
    # texts = ["老板天下第一", "LangChain 本地 Agent 真香"]
    # meta_datas = [{"source": "手打"}, {"source": "测试"}]
    # storage.add_texts(texts, meta_datas)

    # 检索相似内容
    query = "本地 Agent 真香"
    results = storage.similarity_search(query, k=1)
    for r in results:
        print(f"检索到内容：{r[0].page_content}, score: {r[1]}")

    # 查看 collections
    print("现有 collections:", storage.list_collections())

if __name__ == "__main__":
    main()
