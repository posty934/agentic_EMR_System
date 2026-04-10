import streamlit as st
import json
from html import escape
import time
import uuid
import threading
from langchain_core.messages import HumanMessage, AIMessage
from core.workflow import EMRWorkflow
from knowledge.patient_memory import PatientLongTermMemory
from datetime import datetime
import re

st.set_page_config(page_title="AI 预问诊系统", page_icon="🏥", layout="centered")


@st.cache_resource(show_spinner=False)
def load_core_workflow():
    return EMRWorkflow().workflow
@st.cache_data(show_spinner=False)
def load_clinical_guidelines():
    with open("knowledge/clinical_guidelines.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_guideline_slots(symptom_term: str) -> list[str]:
    guideline = load_clinical_guidelines().get(symptom_term, {})
    ordered = []

    if isinstance(guideline, dict):
        for section in ["必问核心要素", "必问鉴别要素", "高危红旗征(必须排查)"]:
            values = guideline.get(section, [])
            if isinstance(values, list):
                ordered.extend([str(v).strip() for v in values if str(v).strip()])

    deduped = []
    seen = set()
    for slot in ordered:
        if slot and slot not in seen:
            deduped.append(slot)
            seen.add(slot)

    return deduped


def get_display_slot_answers(entity: dict) -> list[tuple[str, str]]:
    slot_answers = entity.get("slot_answers", {})
    if not isinstance(slot_answers, dict):
        return []

    standard_term = (entity.get("standard_term") or entity.get("symptom_name") or "").strip()
    ordered_slots = get_guideline_slots(standard_term)

    items = []
    seen = set()

    for slot in ordered_slots:
        answer = str(slot_answers.get(slot, "")).strip()
        if answer:
            items.append((slot, answer))
            seen.add(slot)

    for slot, answer in slot_answers.items():
        slot = str(slot).strip()
        answer = str(answer).strip()
        if slot and answer and slot not in seen:
            items.append((slot, answer))

    return items


def trigger_background_load():
    load_core_workflow()


if "bg_load_triggered" not in st.session_state:
    st.session_state.bg_load_triggered = True
    t = threading.Thread(target=trigger_background_load)
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx

        add_script_run_ctx(t)
    except ImportError:
        pass
    t.start()

st.markdown("""
<style>
        .slot-answer-row {
        margin: 0 0 8px 0;
    }

    .slot-answer-chip {
        display: block;
        width: 100%;
        padding: 8px 10px;
        border-radius: 10px;
        background: #f7f8fa;
        border: 1px solid rgba(49, 51, 63, 0.12);
        color: #1f2937;
        line-height: 1.5;
        cursor: help;
        transition: all 0.2s ease;
    }

    .slot-answer-chip:hover {
        background: #eef6ff;
        border-color: rgba(59, 130, 246, 0.28);
    }
    /* 缩小主容器的顶部留白，让卡片重心稍微上移，确保底部不被遮挡 */
    .block-container {
        padding-top: 3.5rem; 
        max-width: 850px; /* 宽度适中收拢 */
    }

    /* 美化 Form 表单外框：精简上下内边距，保留左右呼吸感 */
    [data-testid="stForm"] {
        background-color: transparent;
        border-radius: 16px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        padding: 30px 40px 30px 40px; /* 【核心修改】大幅压缩上下留白，保留左右宽度 */
        transition: all 0.3s ease-in-out;
    }

    [data-testid="stForm"]:hover {
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }

    /* 优化 Primary 按钮样式：等比例缩小 */
    [data-testid="stFormSubmitButton"] button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 16px !important; /* 字体改小 */
        letter-spacing: 4px; /* 字间距适中 */
        padding: 8px 0px; /* 按钮厚度变薄 */
        transition: all 0.3s ease !important;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
    }

    /* 文字样式定制，彻底杜绝换行，同时缩小字号 */
    .title-box {
        text-align: center; 
        padding-top: 0px; 
        padding-bottom: 0px;
    }
    .main-title {
        font-weight: 600; 
        margin-bottom: 8px; 
        font-size: 28px; /* 【核心修改】标题字号缩小 */
        letter-spacing: 2px;
        white-space: nowrap; 
    }
    .sub-title {
        color: #888888; 
        font-size: 14px; /* 【核心修改】副标题字号缩小 */
        font-weight: 300;
        letter-spacing: 1px;
        white-space: nowrap; 
    }
</style>
""", unsafe_allow_html=True)

if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

if not st.session_state.is_logged_in:
    col1, col2, col3 = st.columns([1, 2.2, 1])

    with col2:
        with st.form("login_form", border=True):

            st.markdown(
                """
                <div class="title-box">
                    <div style='font-size: 42px; margin-bottom: 5px;'>🏥</div>
                    <h2 class="main-title">AI 智能预问诊系统</h2>
                    <p class="sub-title">请验证您的就诊信息，建立专属健康档案</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
            st.divider()
            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

            patient_id = st.text_input("💳 就诊卡号 / 身份证号", placeholder="请输入您的卡号，例如：P_001")

            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

            patient_name = st.text_input("👤 患者姓名", placeholder="请输入您的真实姓名，例如：张三")

            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

            submit = st.form_submit_button("登 录 系 统", use_container_width=True, type="primary")

            if submit:
                if patient_id.strip() and patient_name.strip():
                    st.session_state.is_logged_in = True
                    st.session_state.patient_id = patient_id.strip()
                    st.session_state.patient_name = patient_name.strip()
                    st.rerun()
                else:
                    st.error("登陆信息缺失，请重新输入！", icon="🚨")

    st.stop()

if "memory_saved" not in st.session_state:
    st.session_state.memory_saved = False

if "patient_memory" not in st.session_state or st.session_state.get(
        "current_patient_id") != st.session_state.patient_id:
    with st.spinner('📂 正在同步患者历史健康档案...'):
        st.session_state.patient_memory = PatientLongTermMemory(st.session_state.patient_id)
        st.session_state.current_patient_id = st.session_state.patient_id

header_col1, header_col2, header_col3 = st.columns([7, 1, 2.5])
with header_col1:
    st.title("🏥 智能预问诊助手")

with header_col3:
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    with st.popover("👤 个人中心", use_container_width=True):
        st.markdown(f"**姓名:** {st.session_state.patient_name}")
        st.markdown(f"**ID:** `{st.session_state.patient_id}`")
        st.divider()
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "emr_workflow" not in st.session_state:
    with st.spinner('⚙️ 正在为您准备问诊环境，请稍候...'):
        st.session_state.emr_workflow = load_core_workflow()

graph_config = {"configurable": {"thread_id": st.session_state.thread_id}}
current_state = st.session_state.emr_workflow.get_state(graph_config).values

if not current_state.get("messages"):
    initial_msg = AIMessage(
        content=f"您好，{st.session_state.patient_name}，请问您今天哪里不舒服？可以详细描述一下您的症状吗？")
    st.session_state.emr_workflow.update_state(graph_config, {"messages": [initial_msg]})
    current_state = st.session_state.emr_workflow.get_state(graph_config).values

for msg in current_state.get("messages", []):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        if msg.content.startswith("🚨 **【系统质控拦截】**"):
            st.error(msg.content, icon="🚨")
        else:
            st.markdown(msg.content)

if prompt := st.chat_input("请在此输入您的症状或回答..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner('🤖 正在分析病情与查阅既往史...'):

            current_entities_for_memory = current_state.get("entities", [])
            long_term_memory_str = st.session_state.patient_memory.retrieve_memory(
                current_query=prompt,
                current_entities=current_entities_for_memory,
                top_k=2
            )
            inputs = {
                "messages": [HumanMessage(content=prompt)],
                "long_term_memory": long_term_memory_str
            }
            final_state = st.session_state.emr_workflow.invoke(inputs, config=graph_config)

        latest_msg = final_state["messages"][-1].content

        if latest_msg.startswith("🚨 **【系统质控拦截】**"):
            message_placeholder.error(latest_msg, icon="🚨")
        else:
            full_response = ""
            for chunk in latest_msg:
                full_response += chunk
                time.sleep(0.01)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    st.rerun()

if current_state.get("is_finished") is True and current_state.get("is_valid") is True and current_state.get(
        "draft_record"):
    st.success("病历草稿生成完毕，且已通过医学逻辑质控！")

    draft = current_state["draft_record"]
    chief_complaint = draft.get('chief_complaint', '')
    hpi = draft.get('history_of_present_illness', '')

    st.markdown("### 📄 门诊病历草稿")
    st.markdown(f"**【主诉】**\n> {chief_complaint}")
    st.markdown(f"**【现病史】**\n> {hpi}")

    if not st.session_state.memory_saved:
        summary = f"主诉：{chief_complaint}。现病史：{hpi}"
        st.session_state.patient_memory.save_memory(summary)
        st.session_state.memory_saved = True

    export_text = f"门诊病历记录\n就诊人：{st.session_state.patient_name} (ID: {st.session_state.patient_id})\n\n【主诉】\n{chief_complaint}\n\n【现病史】\n{hpi}\n\n---\n由多智能体系统生成"

    current_date = datetime.now().strftime("%Y_%m_%d")

    st.download_button(
        label="💾 导出病历至本地 (TXT格式)",
        data=export_text,
        file_name=f"{st.session_state.patient_id}_{current_date}_门诊病历.txt",
        mime="text/plain"
    )

with st.sidebar:
    st.header("📂 既往病史档案")
    if "patient_memory" in st.session_state:
        all_records = st.session_state.patient_memory.get_all_memories()
        if all_records:
            for idx, record in enumerate(all_records):
                with st.expander(f"历史就诊记录 {idx + 1}", expanded=(idx == len(all_records) - 1)):
                    st.info(record)
        else:
            st.markdown("<p style='color:#888;font-size:14px;'>该患者为首次就诊，暂无既往病史。</p>",
                        unsafe_allow_html=True)

    st.divider()

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

                status = entity.get("status", "active")
                if status == "revoked":
                    card_title += " (已撤销)"

                with st.expander(card_title, expanded=True):
                    display_items = get_display_slot_answers(entity)

                    if display_items:
                        st.markdown("**问诊要点记录：**")
                        for slot_label, answer in display_items:
                            st.markdown(
                                f"""
                                <div class="slot-answer-row">
                                    <span class="slot-answer-chip" title="{escape(slot_label, quote=True)}">
                                        {escape(answer)}
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("当前还没有可展示的问诊要点答案。")



    else:
        st.info("等待病患输入并提取...")

    st.divider()
    if st.button("🗑️ 清空当前对话与记忆"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.memory_saved = False
        st.rerun()
