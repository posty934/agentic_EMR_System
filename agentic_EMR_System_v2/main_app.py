import streamlit as st
import time
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from core.workflow import EMRWorkflow

st.set_page_config(page_title="AI 预问诊系统", page_icon="🏥", layout="centered")
st.title("🏥 智能预问诊助手")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "emr_workflow" not in st.session_state:
    with st.spinner('⚙️ 系统初始化：正在加载多智能体核心引擎...'):
        st.session_state.emr_workflow = EMRWorkflow().workflow

graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}
current_state = st.session_state.emr_workflow.get_state(graph_config).values

if not current_state.get("messages"):
    initial_msg = AIMessage(content="您好，请问您今天哪里不舒服？可以详细描述一下您的症状吗？")
    st.session_state.emr_workflow.update_state(graph_config, {"messages": [initial_msg]})
    current_state = st.session_state.emr_workflow.get_state(graph_config).values

# ==========================================
# 3. 渲染历史对话记录
# ==========================================
for msg in current_state.get("messages", []):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        # 🚨 如果捕捉到质控拦截前缀，用红色的 Error 框高亮渲染
        if msg.content.startswith("🚨 **【系统质控拦截】**"):
            st.error(msg.content, icon="🚨")
        else:
            st.markdown(msg.content)

# ==========================================
# 4. 捕获病患输入并驱动 Agent 运转
# ==========================================
if prompt := st.chat_input("请在此输入您的症状或回答..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner('🤖 正在分析病情与推理中...'):
            inputs = {"messages": [HumanMessage(content=prompt)]}
            final_state = st.session_state.emr_workflow.invoke(inputs, config=graph_config)

        latest_msg = final_state["messages"][-1].content

        # 🚨 判断最新一条消息是否为拦截警告
        if latest_msg.startswith("🚨 **【系统质控拦截】**"):
            message_placeholder.error(latest_msg, icon="🚨")
        else:
            full_response = ""
            for chunk in latest_msg:
                full_response += chunk
                time.sleep(0.01)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    time.sleep(0.2)
    st.rerun()

# ==========================================
# 修复 Bug 1：增加 current_state.get("is_finished") 强判断
# 只有在确诊结束且质控通过时才渲染，用户一旦追加说话，草稿即折叠重算
# ==========================================
if current_state.get("is_finished") is True and current_state.get("is_valid") is True and current_state.get(
        "draft_record"):
    st.success("🎉 病历草稿生成完毕，且已通过医学逻辑质控！")

    draft = current_state["draft_record"]
    chief_complaint = draft.get('chief_complaint', '')
    hpi = draft.get('history_of_present_illness', '')

    st.markdown("### 📄 门诊病历草稿")
    st.markdown(f"**【主诉】**\n> {chief_complaint}")
    st.markdown(f"**【现病史】**\n> {hpi}")

    export_text = f"门诊病历记录\n\n【主诉】\n{chief_complaint}\n\n【现病史】\n{hpi}\n\n---\n由多智能体系统生成"
    st.download_button(
        label="💾 导出病历至本地 (TXT格式)",
        data=export_text,
        file_name="门诊病历.txt",
        mime="text/plain"
    )

with st.sidebar:
    st.header("⚙️ 智能体状态监控")
    st.markdown("### 结构化信息监控面板")

    entities = current_state.get("entities", [])
    if entities:
        st.success(f"✅ 已提取 {len(entities)} 组症状记录！")
        for idx, entity in enumerate(entities):
            symptom_name = entity.get('symptom_name', '未知症状')
            standard_term = entity.get('standard_term', '')
            status = entity.get('status', 'active')

            if status == "revoked":
                with st.expander(f"~~症状 {idx + 1}: {symptom_name}~~ (已排除)", expanded=False):
                    st.markdown("🚫 *患者已澄清无此症状，被质控系统排除*")
            else:
                card_title = f"症状 {idx + 1}: {symptom_name}"
                if standard_term and standard_term != "未知术语":
                    card_title += f" ➡️ [{standard_term}]"

                with st.expander(card_title, expanded=True):
                    # 1. 渲染基础固定字段
                    slot_cn_map = {
                        "onset_time": "时间", "characteristics": "性质",
                        "inducement": "诱因", "frequency": "频率", "alleviating_factors": "缓解/加重"
                    }
                    for k, cn_name in slot_cn_map.items():
                        val = entity.get(k)
                        if val:
                            st.markdown(f"**{cn_name}:** `{val}`")

                    # 2. 🚨 渲染指南动态要求的拓展字段
                    dynamic_details = entity.get("dynamic_details", {})
                    if isinstance(dynamic_details, dict):
                        for d_key, d_val in dynamic_details.items():
                            st.markdown(f"**{d_key}:** `{d_val}`")
    else:
        st.info("等待病患输入并提取...")

    st.divider()
    if st.button("🗑️ 清空当前对话与记忆"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()