import streamlit as st
import time
from agents.agent1_extractor import Agent1Extractor
from knowledge.retriever import MedicalRetriever
from agents.agent2_generator import Agent2Generator
from agents.agent3_reviewer import Agent3Reviewer

st.set_page_config(page_title="AI 预问诊系统", page_icon="🏥", layout="centered")
st.title("🏥 智能预问诊助手")

# === 1. 初始化对话状态和 Agent ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "您好，请问您今天哪里不舒服？可以详细描述一下您的症状吗？\n请您初次描述时尽量详细描述您的症状，感谢配合。"}
    ]

# 实例化 Agent1
if "extractor" not in st.session_state:
    st.session_state.extractor = Agent1Extractor()

# 实例化 Agent2
if "generator" not in st.session_state:
    st.session_state.generator = Agent2Generator()
if "draft_record" not in st.session_state:
    st.session_state.draft_record = None  # 用于保存生成的病历

#实例化Agent3
if "reviewer" not in st.session_state:
    st.session_state.reviewer = Agent3Reviewer()

# 实例化重排检索器 (因为模型比较大，用 st.spinner 给个提示)
if "retriever" not in st.session_state:
    with st.spinner('⚙️ 系统初始化：正在加载医学知识库与重排模型，请稍候...'):
        st.session_state.retriever = MedicalRetriever()

if "all_entities" not in st.session_state:
    st.session_state.all_entities = []

# === 2. 渲染历史对话记录 ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === 3. 捕获病患输入并处理 ===
if prompt := st.chat_input("请在此输入您的症状或回答..."):
    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 处理 Assistant 回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner('Agent 1 正在提取病情并进行术语级映射...'):
            last_ai_msg = ""
            # 倒序遍历历史消息，找到最近的一条 assistant 回复
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    last_ai_msg = msg["content"]
                    break
            # 调用大模型进行粗提取
            new_entities = st.session_state.extractor.extract(prompt, st.session_state.all_entities,last_ai_msg)

            # --- 智能去重、合并与【标准术语映射】 ---
            current_turn_symptoms = []

            if new_entities:
                for new_ent in new_entities:
                    raw_name = new_ent.get("symptom_name", "")
                    if isinstance(raw_name, dict):
                        sym_name = str(list(raw_name.values())[0]) if raw_name else ""
                    else:
                        sym_name = str(raw_name)

                    if not sym_name or sym_name == "未知症状":
                        continue

                    current_turn_symptoms.append(sym_name)

                    # 查找全局列表里是否已经记录过这个症状了
                    existing_ent = next(
                        (item for item in st.session_state.all_entities if item.get("symptom_name") == sym_name), None)

                    if existing_ent:
                        # === 🚀 撤销拦截 ===
                        if new_ent.get("status") == "revoked":
                            existing_ent["status"] = "revoked"
                            continue  # 已经被否认的症状，不需要再缝合细节了
                        # 缝合新的细节
                        updated = False
                        for key in ["onset_time", "characteristics", "inducement", "frequency", "alleviating_factors"]:
                            if new_ent.get(key) and not existing_ent.get(key):
                                existing_ent[key] = new_ent.get(key)
                                updated = True

                        # === 🚀 核心优化：全量上下文动态重映射 ===
                        if updated:
                            combined_query = existing_ent.get("symptom_name", "")
                            if existing_ent.get("characteristics"):
                                combined_query += "，表现为：" + existing_ent.get("characteristics")
                            if existing_ent.get("inducement"):
                                combined_query += "，诱因：" + existing_ent.get("inducement")

                            # 带着更丰富的细节，去知识库里重新查一次标准术语
                            new_standard_term = st.session_state.retriever.get_standard_term(combined_query)
                            existing_ent["standard_term"] = new_standard_term

                    else:
                        # 如果是全新的症状，第一次调用映射
                        standard_term = st.session_state.retriever.get_standard_term(sym_name)

                        new_ent["symptom_name"] = sym_name
                        new_ent["standard_term"] = standard_term
                        st.session_state.all_entities.append(new_ent)

            # --- 生成动态追问话术 ---
            reply_text = st.session_state.extractor.generate_reply(prompt, st.session_state.all_entities)

            # --- 打字机效果输出 ---
            full_response = ""
            for chunk in reply_text:
                full_response += chunk
                time.sleep(0.03)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    # 保存系统回复
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # ========================================================
    # === 🚀 触发 Agent 2&3：病历生成及校验  ===
    # ========================================================
    if "病情信息已收集完毕" in full_response:
        with st.chat_message("assistant"):
            st.markdown("---")

            # 【阶段 1：Agent 2 生成病历】
            with st.spinner('Agent 2 (病历书写员) 正在撰写标准病历草稿...'):
                record = st.session_state.generator.generate_record(st.session_state.all_entities)
                st.session_state.draft_record = record

            # 【阶段 2：Agent 3 逻辑校验】
            with st.spinner('Agent 3 (质控主任) 正在进行医学逻辑校验...'):
                validation = st.session_state.reviewer.validate(record, st.session_state.all_entities)

            # 【阶段 3：决策分流 - 回滚 vs 导出】
            if validation.get("is_valid") is False:
                # ❌ 校验失败：拦截并回滚
                # 将警告信息直接写入对话历史，实现永久可视化展示！
                warning_text = f"⚠️ **触发系统内部质控拦截**\n\n**拦截原因：** {validation.get('feedback')}"
                st.session_state.messages.append({"role": "assistant", "content": warning_text})

                # 拿到 Agent 3 提出的追问，交还给 Agent 1
                rollback_msg = validation.get("rollback_question")
                st.session_state.messages.append({"role": "assistant", "content": rollback_msg})

                # 稍微停顿一下刷新
                time.sleep(1.0)
                st.rerun()

            else:
                # ✅ 校验通过：展示并允许导出
                st.success("🎉 病历草稿生成完毕，且已通过医学逻辑质控！")
                st.markdown("### 📄 门诊病历草稿")

                chief_complaint = record.get('chief_complaint', '')
                hpi = record.get('history_of_present_illness', '')

                st.markdown(f"**【主诉】**\n> {chief_complaint}")
                st.markdown(f"**【现病史】**\n> {hpi}")

                # --- 构建导出文件文本 ---
                export_text = f"门诊病历记录\n\n【主诉】\n{chief_complaint}\n\n【现病史】\n{hpi}\n\n---\n由 Agentic EMR System 自动生成并质控"

                # --- 提供一键下载按钮 ---
                st.download_button(
                    label="💾 导出病历至本地 (TXT格式)",
                    data=export_text,
                    file_name="门诊病历草稿.txt",
                    mime="text/plain"
                )

                # 保存最终记录
                record_msg = (
                    f"### 📄 门诊病历 (已通过质控)\n"
                    f"**【主诉】**\n> {chief_complaint}\n\n"
                    f"**【现病史】**\n> {hpi}"
                )
                st.session_state.messages.append({"role": "assistant", "content": record_msg})

# === 4. 侧边栏：系统后台状态监控 ===
with st.sidebar:
    st.header("⚙️ 系统后台状态监控")
    st.markdown("### Agent 1 (结构化信息库)")

    if st.session_state.all_entities:
        st.success(f"✅ 已提取 {len(st.session_state.all_entities)} 组症状记录！")

        # 遍历字典列表，采用卡片式的样式渲染
        # 遍历字典列表，采用卡片式的样式渲染
        for idx, entity in enumerate(st.session_state.all_entities):
            symptom_name = entity.get('symptom_name', '未知症状')
            standard_term = entity.get('standard_term', '')
            status = entity.get('status', 'active')

            # === 🚀 UI 特效：如果是被排除的症状，加上删除线 ===
            if status == "revoked":
                card_title = f"~~症状 {idx + 1}: {symptom_name}~~ (已排除)"
                with st.expander(card_title, expanded=False):  # 排除的卡片默认折叠
                    st.markdown("🚫 *患者已澄清无此症状，被质控系统排除*")
            else:
                card_title = f"症状 {idx + 1}: {symptom_name}"
                if standard_term and standard_term != "未知术语":
                    card_title += f" ➡️ [{standard_term}]"

                with st.expander(card_title, expanded=True):
                    if entity.get("onset_time"):
                        st.markdown(f"⏱️ **时间:** `{entity.get('onset_time')}`")
                    if entity.get("characteristics"):
                        st.markdown(f"🔍 **性质:** `{entity.get('characteristics')}`")
                    if entity.get("inducement"):
                        st.markdown(f"⚡ **诱因:** `{entity.get('inducement')}`")
                    if entity.get("frequency"):
                        st.markdown(f"🔄 **频率:** `{entity.get('frequency')}`")
                    if entity.get("alleviating_factors"):
                        st.markdown(f"⚖️ **缓解/加重:** `{entity.get('alleviating_factors')}`")
    else:
        st.info("等待病患输入并提取...")

    st.divider()
    if st.button("清空当前对话"):
        st.session_state.messages = []
        st.session_state.all_entities = []
        st.rerun()