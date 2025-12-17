import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def load_and_split_text(file_path):
    """
    加载文本文件并将其分割成小块。
    """
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", "、", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    
    print(f"文件 {os.path.basename(file_path)} 被成功加载并切分。")
    print(f"原始文档数量: {len(documents)}")
    print(f"切分后文档块数量: {len(split_docs)}")
    
    return split_docs

if __name__ == '__main__':
    data_dir = 'data'
    file_name = '三体 (刘慈欣) (Z-Library).txt' # 使用完整版文件
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        chunks = load_and_split_text(file_path)
        
        # 将分割示例写入文件，方便PPT引用
        with open('split_demo.txt', 'w', encoding='utf-8') as f:
            f.write(f"源文件: {file_name}\n")
            f.write(f"总计分割成 {len(chunks)} 个知识片段。\n")
            f.write("="*40 + "\n\n")
            f.write("以下是前5个知识片段的示例：\n\n")

            for i, chunk in enumerate(chunks[:5]):
                f.write(f"--- 片段 {i+1} (长度: {len(chunk.page_content)} 字符) ---\n")
                f.write(chunk.page_content)
                f.write("\n" + "-"*40 + "\n\n")
        
        print("已生成分割示例文件: split_demo.txt")
        print("您可以在PPT中引用此文件的内容。")

    else:
        print(f"错误: 文件 {file_path} 未找到。")
