import sys
import os
import threading
import requests
from flask import Flask, request, Response
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

# --- 项目初始化 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_chain import (
    create_rag_chain, 
    get_deepseek_llm, 
    get_zhipu_llm, 
    invoke_multimodal_chain
)

load_dotenv()

# --- 全局变量和配置 ---
app = Flask(__name__)
# 客服消息接口的 URL，需要从平台获取，这里使用文档中的示例
# 在真实部署时，您可能需要将其配置为环境变量
CUSTOMER_SERVICE_API_URL = "http://1.95.125.201/wx" 

# 用于存储每个用户会话状态的全局字典
# key: user_id, value: {"mode": "role_play", "prompt_template": "..."}
user_session_states = {}

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

def process_request_in_background(user_id: str, content: str, msg_type: str):
    """在后台线程中处理用户的请求并异步回复。"""
    print(f"后台线程开始处理用户 [{user_id}] 的请求...")
    answer = ""

    if msg_type == "image":
        print("检测到图片消息，正在执行复杂多模态分析流程...")
        image_url = content
        
        # 1. 初始化多模态 LLM
        multimodal_llm = get_zhipu_llm(is_multimodal=True)
        if not multimodal_llm:
            answer = "抱歉，多模态模型初始化失败，请检查ZhipuAI API Key。"
            send_custom_message(user_id, answer)
            return

        try:
            # 2. [步骤1/3] 图片描述阶段
            print("[步骤 1/3] 正在识别和描述图片内容...")
            desc_prompt = "你是一个专业的图像分析师。请详细、客观地描述这幅图像的内容，重点描述其主要物体、场景和风格。"
            image_description = invoke_multimodal_chain(multimodal_llm, image_url, desc_prompt)
            print(f"图片描述: {image_description[:100]}...")

            # 3. [步骤2/3] 文本检索阶段
            print("[步骤 2/3] 正在基于图片描述检索相关知识...")
            # 使用图片描述去文本知识库中检索
            config = RunnableConfig(configurable={"session_id": f"session_image_{user_id}"})
            retrieved_context = rag_chain_with_history.invoke({"question": image_description}, config=config)
            print("相关知识检索完成。")

            # 4. [步骤3/3] 最终回答生成阶段
            print("[步骤 3/3] 正在结合图文信息生成最终回答...")
            final_prompt = f"""
            你是一个知识渊博的《三体》专家。请结合以下所有信息，对用户提供的图片进行全面分析和解读。

            ---
            分析任务：
            - 原始图片内容描述: {image_description}
            - 从《三体》知识库中检索到的相关背景知识: {retrieved_context}
            ---

            请根据以上所有信息，给出一个关于这张图片的、结合了《三体》知识的、全面而深刻的分析。
            """
            answer = invoke_multimodal_chain(multimodal_llm, image_url, final_prompt)

        except Exception as e:
            print(f"后台处理复杂多模态链时出错: {e}")
            answer = "抱歉，分析图片时遇到了内部错误。"

    else:  # 默认为 text
        question = content
        config = RunnableConfig(configurable={"session_id": user_id})
        
        try:
            # --- 步骤1: 检查是否为模式切换命令 ---
            is_command = False
            if question.startswith("扮演：") or question.startswith("扮演:"):
                is_command = True
                char_name = question.split("：", 1)[-1].split(":", 1)[-1].strip()
                print(f"切换到角色扮演模式，角色：{char_name}")
                
                ROLE_PLAY_TEMPLATE = f"""
                你正在扮演科幻小说《三体》中的角色：【{char_name}】。
                请严格以【{char_name}】的口吻、性格、知识和视角来回答问题。
                在回答时，请自然地融入角色的特点，不要暴露你是一个AI模型。

                对话历史: {{chat_history}}
                上下文: {{context}}
                问题: {{question}}
                【{char_name}】的回答:
                """
                user_session_states[user_id] = {"mode": "role_play", "prompt_template": ROLE_PLAY_TEMPLATE}
                answer = f"模式已切换：我现在是【{char_name}】。你可以开始与我对话了。"

            elif question.startswith("分析：") or question.startswith("分析:"):
                # 分析模式是一次性的，不需要保存状态
                is_command = True
                decision_question = question.split("：", 1)[-1].split(":", 1)[-1].strip()
                print(f"执行一次性决策模拟，问题：{decision_question}")

                DECISION_TEMPLATE = """
                你是一位冷静、客观的《三体》世界战略分析家。
                请根据下面提供的背景资料，深入、多角度地分析用户提出的决策问题。
                你的分析报告需要结构清晰，至少包含：核心问题、正反论据、关键影响因素和最终结论。

                背景资料: {context}
                决策问题: {question}
                战略分析报告:
                """
                decision_rag_chain = create_rag_chain(llm, prompt_template=DECISION_TEMPLATE)
                answer = decision_rag_chain.invoke({"question": decision_question}, config=config)

            elif question in ["重置模式", "普通模式"]:
                is_command = True
                if user_id in user_session_states:
                    del user_session_states[user_id]
                answer = "模式已重置为普通问答模式。"

            # --- 步骤2: 如果不是命令，则根据当前状态处理 ---
            if not is_command:
                current_state = user_session_states.get(user_id)
                
                if current_state and current_state["mode"] == "role_play":
                    # 用户处于角色扮演模式
                    print(f"用户 [{user_id}] 处于角色扮演模式，使用专用链...")
                    prompt_template = current_state["prompt_template"]
                    dynamic_chain = create_rag_chain(llm, prompt_template=prompt_template)
                    answer = dynamic_chain.invoke({"question": question}, config=config)
                else:
                    # 默认使用普通问答模式
                    print(f"用户 [{user_id}] 处于普通模式，使用默认链...")
                    answer = rag_chain_with_history.invoke({"question": question}, config=config)

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
