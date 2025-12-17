import sys
import os
import threading
import requests
from flask import Flask, request, Response
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

# --- 项目初始化 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_chain import create_rag_chain, get_deepseek_llm

load_dotenv()

# --- 全局变量和配置 ---
app = Flask(__name__)
# 客服消息接口的 URL，需要从平台获取，这里使用文档中的示例
# 在真实部署时，您可能需要将其配置为环境变量
CUSTOMER_SERVICE_API_URL = "http://localhost:8000/send_custom_message" 

# --- RAG 链预加载 ---
print("正在初始化微信后端服务...")
llm = get_deepseek_llm()
if not llm:
    raise RuntimeError("DeepSeek LLM 初始化失败，请检查 API Key。")

PROMPT_TEMPLATE = """
你是一个关于科幻小说《三体》的知识问答助手。
请根据下面提供的上下文和对话历史来连贯地回答问题。
如果你在上下文中找不到答案，就说你不知道。

对话历史:
{chat_history}

上下文: 
{context}

问题: 
{question}

回答:
"""
rag_chain_with_history = create_rag_chain(llm, prompt_template=PROMPT_TEMPLATE)
if not rag_chain_with_history:
    raise RuntimeError("无法创建 RAG 链，请检查向量数据库。")

print("微信后端服务已就绪，等待请求...")

# --- 辅助函数 ---

def send_custom_message(user_id: str, content: str, msg_type: str = "text"):
    """调用客服消息 API 异步发送消息给用户。"""
    payload = {
        "openid": user_id,
        "message_type": msg_type,
        "content": content
    }
    try:
        print(f"正在向用户 [{user_id}] 发送异步客服消息...")
        res = requests.post(CUSTOMER_SERVICE_API_URL, json=payload, timeout=5)
        res.raise_for_status() # 如果请求失败则抛出异常
        print(f"异步消息发送成功: {res.json()}")
    except requests.exceptions.RequestException as e:
        print(f"错误：调用客服消息 API 失败: {e}")

def process_request_in_background(user_id: str, question: str, msg_type: str):
    """在后台线程中处理用户的请求并异步回复。"""
    print(f"后台线程开始处理用户 [{user_id}] 的请求...")
    
    # 注意：图片消息处理逻辑需要在这里集成
    if msg_type == "image":
        # 这里应该调用我们之前实现的多模态逻辑
        # 由于简化，我们暂时只回复文本
        answer = "图片消息处理功能正在集成中。"
    else: # 默认为 text
        config = RunnableConfig(configurable={"session_id": user_id})
        try:
            response = rag_chain_with_history.invoke({"question": question}, config=config)
            answer = response
        except Exception as e:
            print(f"后台处理 RAG 链时出错: {e}")
            answer = "抱歉，处理您的问题时遇到了内部错误。"

    # 将最终答案通过客服消息接口发回
    send_custom_message(user_id, answer)
    print(f"后台线程处理完成。")

# --- Flask API 路由 ---

@app.route('/', methods=['POST'])
def wechat_agent_handler():
    """
    处理来自平台服务器的 POST 请求。
    严格遵守 `application/x-www-form-urlencoded` 和 5 秒超时规范。
    """
    if request.content_type != 'application/x-www-form-urlencoded':
        return Response("Error: Content-Type must be application/x-www-form-urlencoded", status=400)

    try:
        user_id = request.form['from_user']
        content = request.form['content']
        msg_type = request.form['type']
    except KeyError:
        return Response("Error: Missing required form parameters (from_user, content, type)", status=400)

    print(f"\n收到来自用户 [{user_id}] 的 [{msg_type}] 消息，内容: {content[:50]}...")

    # 启动后台线程处理耗时任务
    thread = threading.Thread(
        target=process_request_in_background,
        args=(user_id, content, msg_type)
    )
    thread.start()

    # 立即返回纯文本响应，满足 5 秒超时要求
    print("立即返回 '正在处理' 响应...")
    return Response("您的问题正在思考中，请稍候...", mimetype='text/plain')

if __name__ == '__main__':
    # 注意：Flask 的 debug 模式会启动两个进程，可能导致初始化代码运行两次。
    # 在生产环境中应使用 Gunicorn 等 WSGI 服务器。
    app.run(host='0.0.0.0', port=8081, debug=False)
