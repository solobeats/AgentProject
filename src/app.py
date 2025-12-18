import os
import os
import sys
import base64
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_chain import create_rag_chain, get_deepseek_llm, get_zhipu_llm

def main_menu():
    """显示主菜单并获取用户选择。"""
    print("\n欢迎来到《三体》多功能智能 Agent！")
    print("====================================")
    print("请选择一个功能模式：")
    print("1. 普通问答模式")
    print("2. 角色扮演模式")
    print("3. 决策模拟模式")
    print("4. 图片问答模式")
    print("0. 退出")
    
    while True:
        choice = prompt("请输入选项 (0-4): ")
        if choice in ['0', '1', '2', '3', '4']:
            return choice
        else:
            print("无效输入，请重新选择。")

def normal_qa_mode(rag_chain):
    """普通问答模式。"""
    print("\n--- 进入普通问答模式 ---")
    print("你可以问任何关于《三体》的问题。")
    # 为本地测试定义一个固定的 session_id
    config = RunnableConfig(configurable={"session_id": "local_test_session_normal"})
    while True:
        try:
            query = prompt("Q (输入 '返回' 回到主菜单): ")
            if query.lower() == '返回':
                break
            if query:
                answer = rag_chain.invoke({"question": query}, config=config)
                print(f"\nA: {answer}\n")
        except (EOFError, KeyboardInterrupt):
            break

def role_play_mode(llm):
    """角色扮演模式。"""
    if not llm:
        print("文本模型未初始化，无法进入此模式。")
        return
    print("\n--- 进入角色扮演模式 ---")
    characters = ["罗辑", "叶文洁", "史强", "智子", "章北海","程心"]
    completer = WordCompleter(characters, ignore_case=True)
    
    while True:
        try:
            char_name = prompt("请选择你想对话的角色 (例如: 罗辑, 史强): ", completer=completer)
            if char_name in characters:
                break
            else:
                print("无效的角色，请从列表中选择。")
        except (EOFError, KeyboardInterrupt):
            return

    print(f"你现在正在与【{char_name}】对话...")

    ROLE_PLAY_TEMPLATE = f"""
    你正在扮演科幻小说《三体》中的角色：【{char_name}】。
    请严格以【{char_name}】的口吻、性格、知识和视角，并根据下面提供的上下文来回答问题。
    在回答时，请自然地融入角色的特点，不要暴露你是一个AI模型或是在扮演角色。

    上下文: 
    {{context}}

    问题: 
    {{question}}

    【{char_name}】的回答:
    """
    
    # 为这个特定角色创建一个专用的 RAG 链
    print(f"正在为【{char_name}】配置专属思维模块...")
    role_rag_chain = create_rag_chain(llm, prompt_template=ROLE_PLAY_TEMPLATE)
    if not role_rag_chain:
        print("角色模块配置失败，返回主菜单。")
        return
    
    # 为本地测试定义一个固定的 session_id
    config = RunnableConfig(configurable={"session_id": f"local_test_session_roleplay_{char_name}"})
    while True:
        try:
            query = prompt(f"你对【{char_name}】说 (输入 '返回' 结束对话): ")
            if query.lower() == '返回':
                break
            if query:
                answer = role_rag_chain.invoke({"question": query}, config=config)
                print(f"\n【{char_name}】: {answer}\n")
        except (EOFError, KeyboardInterrupt):
            break

def decision_simulation_mode(llm):
    """决策模拟模式。"""
    if not llm:
        print("文本模型未初始化，无法进入此模式。")
        return
    print("\n--- 进入决策模拟模式 ---")
    print("你可以提出一个《三体》世界中的困境或决策点，Agent 将为你进行分析。")
    
    DECISION_TEMPLATE = """
    你是一位冷静、客观的《三体》世界战略分析家。
    请根据下面提供的背景资料，深入、多角度地分析用户提出的决策问题或困境。
    你的分析报告需要结构清晰，至少包含以下几个方面：
    1.  **核心问题**：简要重述需要决策的关键点。
    2.  **正面论据**：支持该决策的理由，以及可能带来的短期和长期益处。
    3.  **反面论据**：反对该决策的理由，以及潜在的风险和负面后果。
    4.  **关键影响因素**：指出影响此决策成败的关键变量（如技术、人性、三体文明的反应等）。
    5.  **书中相关人物的可能观点**：引用或推断书中主要人物（如罗辑、章北海、维德等）对该决策可能持有的立场。
    6.  **最终结论**：给出一个综合性的、没有个人偏见的总结。

    背景资料: 
    {context}

    决策问题: 
    {question}

    战略分析报告:
    """
    
    print("正在配置战略分析模块...")
    decision_rag_chain = create_rag_chain(llm, prompt_template=DECISION_TEMPLATE)
    if not decision_rag_chain:
        print("战略分析模块配置失败，返回主菜单。")
        return

    # 为本地测试定义一个固定的 session_id
    config = RunnableConfig(configurable={"session_id": "local_test_session_decision"})
    while True:
        try:
            query = prompt("请输入你希望分析的决策问题 (输入 '返回' 回到主菜单): ")
            if query.lower() == '返回':
                break
            if query:
                answer = decision_rag_chain.invoke({"question": query}, config=config)
                print("\n--- 战略分析报告 ---")
                print(answer)
                print("---------------------\n")
        except (EOFError, KeyboardInterrupt):
            break

def image_to_base64(image_path):
    """将图片文件转换为 Base64 编码的字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取或编码图片时出错: {e}")
        return None

def multi_modal_mode(text_rag_chain):
    """多模态（图片问答）模式。"""
    print("\n--- 进入图片问答模式 ---")
    
    # 1. 初始化多模态 LLM
    print("正在初始化多模态识别核心 (ZhipuAI GLM-4V)...")
    zhipu_llm = get_zhipu_llm(is_multimodal=True)
    if not zhipu_llm:
        print("多模态核心初始化失败，请检查 ZHIPUAI_API_KEY。")
        return
    print("多模态核心已就绪！")

    # 2. 获取用户输入
    try:
        image_path = prompt("请输入图片路径: ")
        if not os.path.exists(image_path):
            print("错误：文件路径不存在。")
            return
        
        user_question = prompt("请输入你关于这张图片的问题: ")
    except (EOFError, KeyboardInterrupt):
        return

    # 3. 图片描述阶段
    print("\n[步骤 1/3] 正在识别和描述图片内容...")
    b64_image = image_to_base64(image_path)
    if not b64_image:
        return

    desc_message = HumanMessage(
        content=[
            {"type": "text", "text": "你是一个专业的图像分析师。请详细、客观地描述这幅图像的内容，重点描述其主要物体、场景和风格。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
        ]
    )
    
    try:
        image_description = zhipu_llm.invoke([desc_message]).content
        print(f"图片描述: {image_description}")
    except Exception as e:
        print(f"调用多模态模型描述图片时出错: {e}")
        return

    # 4. 文本检索阶段
    print("\n[步骤 2/3] 正在基于图片描述检索相关知识...")
    if not text_rag_chain:
        print("错误：文本知识库未初始化，无法进行检索。")
        return
    
    # 将图片描述和用户问题结合起来，进行更精准的检索
    retrieval_query = f"关于“{image_description}”，{user_question}"
    config = RunnableConfig(configurable={"session_id": "local_test_session_multimodal"})
    retrieved_context = text_rag_chain.invoke({"question": retrieval_query}, config=config)
    print("相关知识检索完成。")

    # 5. 最终回答生成阶段
    print("\n[步骤 3/3] 正在结合图文信息生成最终回答...")
    final_prompt = f"""
    你是一个知识渊博的《三体》专家。请结合以下所有信息，回答用户的问题。

    ---
    用户信息：
    - 原始问题: {user_question}
    - 提供的图片内容描述: {image_description}

    ---
    从《三体》知识库中检索到的相关背景知识：
    {retrieved_context}
    ---

    请根据以上所有信息，给出一个全面、准确的回答。
    """
    
    final_message = HumanMessage(
        content=[
            {"type": "text", "text": final_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
        ]
    )

    try:
        final_answer = zhipu_llm.invoke([final_message]).content
        print("\n--- 最终回答 ---")
        print(final_answer)
        print("------------------\n")
    except Exception as e:
        print(f"生成最终回答时出错: {e}")
        return

def run_app():
    """运行主应用程序。"""
    load_dotenv()
    
    # 预先加载文本处理 LLM
    print("正在初始化文本处理核心 (DeepSeek)，请稍候...")
    deepseek_llm = get_deepseek_llm()
    if not deepseek_llm:
        print("文本处理核心初始化失败，部分功能将不可用。")
    else:
        print("文本处理核心已就绪！")

    # 为普通问答模式预先构建一个基础 RAG 链
    base_rag_chain = None
    if deepseek_llm:
        print("正在构建基础知识库...")
        base_rag_chain = create_rag_chain(deepseek_llm)
        if base_rag_chain:
            print("基础知识库构建完成！")
        else:
            print("基础知识库构建失败。")


    while True:
        choice = main_menu()
        if choice == '1':
            if base_rag_chain:
                normal_qa_mode(base_rag_chain)
            else:
                print("错误：普通问答模式不可用，因为核心引擎初始化失败。")
        elif choice == '2':
            role_play_mode(deepseek_llm)
        elif choice == '3':
            decision_simulation_mode(deepseek_llm)
        elif choice == '4':
            # 图片问答模式需要基础的文本 RAG 链来进行上下文检索
            multi_modal_mode(base_rag_chain)
        elif choice == '0':
            print("感谢使用，再见！")
            break

if __name__ == '__main__':
    run_app()
