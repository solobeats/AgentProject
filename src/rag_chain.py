import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models import ChatZhipuAI
from prompt_toolkit import prompt

from src.history import get_session_history

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- LLM 初始化 ---

def get_deepseek_llm():
    """获取并初始化 DeepSeek LLM"""
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("错误: 未找到 DEEPSEEK_API_KEY。")
        return None
    return ChatDeepSeek(
        api_key=deepseek_api_key,
        model="deepseek-chat",
        temperature=0.1,
    )

def get_zhipu_llm(is_multimodal=False):
    """获取并初始化 ZhipuAI LLM"""
    zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
    if not zhipuai_api_key:
        print("错误: 未找到 ZHIPUAI_API_KEY。")
        return None
    model_name = "glm-4v" if is_multimodal else "glm-4"
    return ChatZhipuAI(
        api_key=zhipuai_api_key,
        model=model_name,
        temperature=0.1,
    )

# --- RAG 链构建 ---

DEFAULT_PROMPT_TEMPLATE = """
你是一个关于科幻小说《三体》的知识问答助手。
请根据下面提供的上下文和对话历史来连贯地回答问题。
如果你在上下文中找不到答案，就说你不知道。

上下文: 
{context}

问题: 
{question}
"""

def create_rag_chain(llm, prompt_template: str = DEFAULT_PROMPT_TEMPLATE):
    """
    创建并返回一个支持对话历史的 RAG 链。
    """
    # 加载环境变量
    load_dotenv()

    # 1. 加载向量数据库
    vector_store_path = 'vector_store/faiss_index_three_body_full'
    embedding_model_name = 'moka-ai/m3e-base'
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    try:
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"加载向量数据库失败: {e}")
        return None
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. 创建带有历史记录的 Prompt 模板
    # MessagesPlaceholder 用于为历史消息列表提供占位符
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # 3. 构建 RAG 链
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 核心链：接收包含 question 和 chat_history 的字典，
    # 添加 context，然后传递给 prompt, llm, parser
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. 使用 RunnableWithMessageHistory 包装 RAG 链
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return chain_with_history

def invoke_multimodal_chain(llm, image_url: str, question: str) -> str:
    """
    调用多模态 LLM (如 GLM-4V) 来处理结合了图片和文本的问题。
    
    参数:
    - llm: 一个已经初始化的多模态聊天模型实例。
    - image_url: 指向图片的公开可访问 URL。
    - question: 关于图片的文本问题或提示。
    
    返回:
    - LLM 生成的文本回答。
    """
    msg = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    response = llm.invoke([msg])
    return response.content

# (移除主执行块)
