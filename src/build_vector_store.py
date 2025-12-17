import os
import sys
# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.load_and_split import load_and_split_text

def build_and_save_vector_store(docs, embeddings, index_path):
    """
    根据文档和指定的 embedding 模型构建 FAISS 向量数据库，并将其保存到本地。
    """
    print("正在构建向量数据库...")
    vector_store = FAISS.from_documents(docs, embeddings)
    print("向量数据库构建完成。")
    
    print(f"正在将向量数据库保存到: {index_path}")
    vector_store.save_local(index_path)
    print("向量数据库已成功保存。")
    return vector_store

if __name__ == '__main__':
    # 定义模型和路径
    data_file_path = 'data/三体 (刘慈欣) (Z-Library).txt'
    vector_store_path = 'vector_store/faiss_index_three_body_full'
    embedding_model_name = 'moka-ai/m3e-base'

    # 1. 加载并切分文档
    if not os.path.exists(data_file_path):
        print(f"错误: 数据文件 {data_file_path} 未找到。")
    else:
        documents = load_and_split_text(data_file_path)
        
        # 2. 初始化 Embedding 模型
        print(f"正在初始化 Embedding 模型: {embedding_model_name}")
        # model_kwargs = {'device': 'cpu'} # 如果没有GPU，可以明确指定使用CPU
        # encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            # model_kwargs=model_kwargs,
            # encode_kwargs=encode_kwargs
        )
        print("Embedding 模型初始化完成。")

        # 3. 构建并保存向量数据库
        build_and_save_vector_store(documents, embeddings, vector_store_path)

        # 4. (可选) 测试加载和搜索
        print("\n--- 测试加载和搜索 ---")
        try:
            loaded_vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            query = "黑暗森林法则是什么？"
            results = loaded_vector_store.similarity_search(query, k=2)
            
            print(f"查询: '{query}'")
            print("找到的相关文档:")
            for doc in results:
                print("-" * 20)
                print(doc.page_content)
            print("-" * 20)
        except Exception as e:
            print(f"测试加载和搜索时发生错误: {e}")
