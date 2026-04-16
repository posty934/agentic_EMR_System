import streamlit as st
import json
from html import escape, unescape
import os
import time
import uuid
import threading
import shutil
import subprocess
import tempfile
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from core.workflow import EMRWorkflow
from knowledge.patient_memory import PatientLongTermMemory  # 🚀 引入长期记忆引擎
from datetime import datetime
import re

st.set_page_config(page_title="AI 预问诊系统", page_icon="🏥", layout="centered")

APP_DIR = Path(__file__).resolve().parent
PRECONSULTATION_TEMPLATE_PATH = APP_DIR / "预问诊单模板.html"


# ==========================================
# 0. 并行静默预加载核心模型
# ==========================================
@st.cache_resource(show_spinner=False)
def load_core_workflow():
    return EMRWorkflow().workflow
@st.cache_data(show_spinner=False)
def load_clinical_guidelines():
    with open("knowledge/clinical_guidelines.json", "r", encoding="utf-8") as f:
        return json.load(f)


def format_minute_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def normalize_age_for_storage(age_input: str) -> str:
    age_text = str(age_input or "").strip().replace("岁", "")
    age_text = re.sub(r"\s+", "", age_text)
    if not re.fullmatch(r"\d{1,3}", age_text):
        return ""

    age = int(age_text)
    if 0 <= age <= 130:
        return str(age)
    return ""


def format_age_for_export(age: str) -> str:
    age_text = str(age or "").strip()
    if not age_text:
        return ""
    if age_text.endswith("岁"):
        return age_text
    return f"{age_text}岁"


def clean_export_text(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def normalize_past_history_for_export(value) -> str:
    text = clean_export_text(value)
    if not text:
        return ""

    empty_markers = [
        "首次就诊",
        "无既往病史记录",
        "暂无与当前症状直接相关的既往史",
        "暂无与当前症状高度相关的既往史",
    ]
    if any(marker in text for marker in empty_markers):
        return ""
    return text


def build_preconsultation_data(profile: dict, draft: dict, past_history: str) -> dict:
    return {
        "name": st.session_state.patient_name,
        "age": format_age_for_export(st.session_state.get("patient_age") or profile.get("age", "")),
        "sex": st.session_state.get("patient_sex") or profile.get("sex", ""),
        "visit_time": st.session_state.get("visit_time", ""),
        "record_time": st.session_state.get("record_time", ""),
        "chief_complaint": clean_export_text(draft.get("chief_complaint", "")),
        "present_history": clean_export_text(draft.get("history_of_present_illness", "")),
        "past_history": normalize_past_history_for_export(past_history),
        "diagnostic_suggestion": "",
    }


def render_preconsultation_html(data: dict) -> str:
    template = PRECONSULTATION_TEMPLATE_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(data, ensure_ascii=False, indent=6).replace("</", "<\\/")
    return re.sub(
        r"const\s+data\s*=\s*\{.*?\};",
        f"const data = {data_json};",
        template,
        count=1,
        flags=re.S,
    )


def find_pdf_browser() -> str:
    for executable in ["msedge", "chrome", "chromium"]:
        resolved = shutil.which(executable)
        if resolved:
            return resolved

    candidate_paths = [
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]

    for path in candidate_paths:
        if path and path.exists():
            return str(path)
    return ""


@st.cache_data(show_spinner=False)
def html_to_pdf_bytes(html: str) -> tuple[bytes | None, str]:
    browser_path = find_pdf_browser()
    if not browser_path:
        return None, "未找到可用的 Microsoft Edge / Google Chrome 浏览器，无法生成 PDF。"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        html_path = tmp_path / "preconsultation.html"
        pdf_path = tmp_path / "preconsultation.pdf"
        html_path.write_text(html, encoding="utf-8")

        common_args = [
            browser_path,
            "--disable-gpu",
            "--disable-extensions",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1000",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ]

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        last_error = ""
        for headless_arg in ["--headless=new", "--headless"]:
            cmd = [common_args[0], headless_arg, *common_args[1:]]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    creationflags=creationflags,
                )
            except Exception as exc:
                last_error = str(exc)
                continue

            if result.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
                return pdf_path.read_bytes(), ""

            last_error = (result.stderr or result.stdout or "").strip()

        return None, last_error or "浏览器 PDF 转换失败。"


def wrap_pdf_text(text: str, max_chars: int) -> list[str]:
    wrapped = []
    for paragraph in clean_export_text(text).splitlines() or [""]:
        while len(paragraph) > max_chars:
            wrapped.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        wrapped.append(paragraph)
    return wrapped


def pdf_text_command(text: str, x: int, y: int, size: int = 12) -> str:
    hex_text = str(text or "").encode("utf-16-be").hex().upper()
    return f"BT /F1 {size} Tf 1 0 0 1 {x} {y} Tm <{hex_text}> Tj ET"


def build_pdf_document(content_stream: str) -> bytes:
    stream_bytes = content_stream.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [6 0 R] >>",
        b"<< /Length " + str(len(stream_bytes)).encode("ascii") + b" >>\nstream\n" + stream_bytes + b"\nendstream",
        (
            b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light "
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 2 >> /DW 1000 >>"
        ),
    ]

    pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"

    xref_position = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode("ascii")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode("ascii")
    return pdf


def render_basic_preconsultation_pdf(data: dict) -> bytes:
    commands = [
        "0.8 w",
        "50 50 495 742 re S",
        pdf_text_command("ABC市人民医院", 245, 790, 14),
        pdf_text_command("预问诊单", 240, 755, 24),
        "50 720 m 545 720 l S",
        pdf_text_command(f"姓名: {data.get('name', '')}", 70, 690, 12),
        pdf_text_command(f"年龄: {data.get('age', '')}", 230, 690, 12),
        pdf_text_command("性别:", 360, 690, 12),
        pdf_text_command("男", 405, 690, 12),
        "425 686 10 10 re S",
        pdf_text_command("女", 455, 690, 12),
        "475 686 10 10 re S",
        pdf_text_command("√" if data.get("sex") == "男" else "", 427, 688, 10),
        pdf_text_command("√" if data.get("sex") == "女" else "", 477, 688, 10),
        pdf_text_command(f"问诊时间: {data.get('visit_time', '')}", 70, 660, 12),
        pdf_text_command(f"记录时间: {data.get('record_time', '')}", 310, 660, 12),
        "50 635 m 545 635 l S",
        pdf_text_command("病 史", 270, 610, 18),
    ]

    sections = [
        ("主诉:", data.get("chief_complaint", ""), 575, 2, 36),
        ("现病史:", data.get("present_history", ""), 515, 8, 36),
        ("既往史:", data.get("past_history", ""), 365, 4, 36),
        ("诊断建议:", data.get("diagnostic_suggestion", ""), 265, 5, 34),
    ]

    for label, text, y, max_lines, max_chars in sections:
        commands.append(pdf_text_command(label, 70, y, 13))
        line_y = y
        for line in wrap_pdf_text(text, max_chars)[:max_lines]:
            commands.append(pdf_text_command(line, 150, line_y, 12))
            line_y -= 20

    return build_pdf_document("\n".join(commands))


def render_patient_profile_form(form_key: str):
    st.caption("首次就诊需要完善基本信息，用于生成预问诊单。")
    with st.form(form_key, border=False):
        age_input = st.text_input("年龄", placeholder="例如：29")
        sex_input = st.radio("性别", ["男", "女"], horizontal=True)
        submitted = st.form_submit_button("保存并进入问诊", use_container_width=True, type="primary")

        if submitted:
            age = normalize_age_for_storage(age_input)
            if not age:
                st.error("请输入 0-130 之间的有效年龄。", icon="🚨")
                return

            st.session_state.patient_memory.save_patient_profile(
                name=st.session_state.patient_name,
                age=age,
                sex=sex_input,
            )
            st.session_state.patient_age = age
            st.session_state.patient_sex = sex_input
            st.rerun()


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


def clean_slot_display_text(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_display_slot_answers(entity: dict) -> list[tuple[str, str]]:
    slot_answers = entity.get("slot_answers", {})
    slot_display_answers = entity.get("slot_display_answers", {})
    dynamic_details = entity.get("dynamic_details", {})

    if not isinstance(slot_answers, dict):
        slot_answers = {}
    if not isinstance(slot_display_answers, dict):
        slot_display_answers = {}
    if not isinstance(dynamic_details, dict):
        dynamic_details = {}

    standard_term = (entity.get("standard_term") or entity.get("symptom_name") or "").strip()
    ordered_slots = get_guideline_slots(standard_term)

    items = []
    seen = set()

    for slot in ordered_slots:
        raw_answer = clean_slot_display_text(slot_answers.get(slot, ""))
        display_answer = clean_slot_display_text(slot_display_answers.get(slot, ""))

        if raw_answer:
            items.append((slot, display_answer or raw_answer))
            seen.add(slot)

    for slot, raw_answer in slot_answers.items():
        slot = str(slot).strip()
        raw_answer = clean_slot_display_text(raw_answer)
        display_answer = clean_slot_display_text(slot_display_answers.get(slot, ""))

        if slot and raw_answer and slot not in seen:
            items.append((slot, display_answer or raw_answer))
            seen.add(slot)

    history_relation = clean_slot_display_text(dynamic_details.get("与过往病史关联", ""))
    if history_relation and "与既往相比" not in seen:
        items.append(("与既往相比", history_relation))

    return items


def get_slot_review_meta(entity: dict, slot: str) -> dict:
    review_status = entity.get("slot_review_status", {})
    if not isinstance(review_status, dict):
        return {}

    meta = review_status.get(slot, {})
    return meta if isinstance(meta, dict) else {}


def render_slot_answer_chip(slot_label: str, answer: str, review_meta: dict) -> str:
    status = (review_meta.get("status") or "").strip()
    css_class = "slot-status-normal"
    badge = ""
    hint = ""

    if status == "invalid":
        css_class = "slot-status-invalid"
        badge = "<span class='slot-answer-badge slot-answer-badge-invalid'>需重新确认</span>"
        reason = (review_meta.get("reason") or "当前回答与症状或问诊要点不匹配。").strip()
        hint = f"<div class='slot-answer-hint'>原因：{escape(reason)}</div>"
    elif status == "reasked":
        css_class = "slot-status-reasked"
        badge = "<span class='slot-answer-badge slot-answer-badge-reasked'>已重新追问</span>"
        hint = "<div class='slot-answer-hint'>已按患者最新回复更新。</div>"

    return (
        '<div class="slot-answer-row">'
        f'<div class="slot-answer-chip {css_class}">'
        '<div class="slot-answer-topline">'
        f'<span class="slot-answer-label">{escape(slot_label)}</span>'
        f"{badge}"
        "</div>"
        f'<div class="slot-answer-value">{escape(answer)}</div>'
        f"{hint}"
        "</div>"
        "</div>"
    )


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

# ==========================================
# 1. 登录拦截与鉴权模块 (黄金比例平衡版)
# ==========================================

# --- 自定义 CSS 样式注入 ---
st.markdown("""
<style>
        .slot-answer-row {
        margin: 0 0 8px 0;
    }

    .slot-answer-chip {
        display: block;
        width: 100%;
        padding: 8px 10px;
        border-radius: 8px;
        background: #f7f8fa;
        border: 1px solid rgba(49, 51, 63, 0.12);
        color: #1f2937;
        line-height: 1.5;
        cursor: default;
        transition: all 0.2s ease;
    }

    .slot-answer-chip:hover {
        background: #eef6ff;
        border-color: rgba(59, 130, 246, 0.28);
    }

    .slot-answer-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 3px;
    }

    .slot-answer-label {
        flex: 1 1 120px;
        min-width: 0;
        color: #64748b;
        font-size: 12px;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .slot-answer-value {
        color: #111827;
        font-size: 14px;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .slot-answer-badge {
        flex: 0 0 auto;
        border-radius: 8px;
        padding: 1px 6px;
        font-size: 11px;
        line-height: 1.5;
        font-weight: 600;
    }

    .slot-answer-hint {
        margin-top: 4px;
        color: #475569;
        font-size: 12px;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .slot-status-invalid {
        background: #fff7f7;
        border-color: #f0b4b4;
        border-left: 4px solid #d94b4b;
        box-shadow: inset 0 0 0 1px rgba(217, 75, 75, 0.04);
    }

    .slot-status-invalid:hover {
        background: #fff1f1;
        border-color: #e58b8b;
    }

    .slot-answer-badge-invalid {
        background: #fee2e2;
        color: #9f1239;
    }

    .slot-status-reasked {
        background: #f0fdfa;
        border-color: #8fd8cc;
        border-left: 4px solid #0f9f8f;
    }

    .slot-status-reasked:hover {
        background: #e6fbf7;
        border-color: #5fc8ba;
    }

    .slot-answer-badge-reasked {
        background: #ccfbf1;
        color: #115e59;
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

# --- 状态初始化 ---
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = ""
if "patient_sex" not in st.session_state:
    st.session_state.patient_sex = ""
if "visit_time" not in st.session_state:
    st.session_state.visit_time = ""

# --- 登录页面 UI ---
if not st.session_state.is_logged_in:

    # 列宽比例调整为 1:2:1，既有宽屏感，又不会显得太肥胖
    col1, col2, col3 = st.columns([1, 2.2, 1])

    with col2:
        with st.form("login_form", border=True):

            # 头部信息，缩减了 icon 的尺寸和下边距
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

            # 缩减分割线前后的多余空间
            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
            st.divider()
            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

            # 输入框组
            patient_id = st.text_input("💳 就诊卡号 / 身份证号", placeholder="请输入您的卡号，例如：P_001")

            # 缩减输入框之间的间距
            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

            patient_name = st.text_input("👤 患者姓名", placeholder="请输入您的真实姓名，例如：张三")

            # 按钮上方的留白从 40px 砍到了 20px
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

            # 提交按钮
            submit = st.form_submit_button("登 录 系 统", use_container_width=True, type="primary")

            if submit:
                if patient_id.strip() and patient_name.strip():
                    st.session_state.is_logged_in = True
                    st.session_state.patient_id = patient_id.strip()
                    st.session_state.patient_name = patient_name.strip()
                    st.session_state.patient_age = ""
                    st.session_state.patient_sex = ""
                    st.session_state.visit_time = format_minute_timestamp()
                    st.session_state.pop("record_time", None)
                    st.rerun()
                else:
                    st.error("登陆信息缺失，请重新输入！", icon="🚨")

    st.stop()

# ==========================================
# 🚀 2. 初始化用户的专属长期记忆引擎
# ==========================================
if "memory_saved" not in st.session_state:
    st.session_state.memory_saved = False  # 锁：防止单次病历被重复保存

if "patient_memory" not in st.session_state or st.session_state.get(
        "current_patient_id") != st.session_state.patient_id:
    with st.spinner('📂 正在同步患者历史健康档案...'):
        st.session_state.patient_memory = PatientLongTermMemory(st.session_state.patient_id)
        st.session_state.current_patient_id = st.session_state.patient_id

patient_profile = st.session_state.patient_memory.get_patient_profile()
if patient_profile:
    st.session_state.patient_age = patient_profile.get("age", "")
    st.session_state.patient_sex = patient_profile.get("sex", "")

if not st.session_state.get("visit_time"):
    st.session_state.visit_time = format_minute_timestamp()

requires_initial_profile = (
    not st.session_state.patient_memory.get_all_memories()
    and not st.session_state.patient_memory.has_patient_profile()
)

if requires_initial_profile:
    if hasattr(st, "dialog"):
        @st.dialog("完善基本信息")
        def initial_patient_profile_dialog():
            render_patient_profile_form("initial_patient_profile_form_dialog")

        initial_patient_profile_dialog()
    else:
        st.info("首次就诊需要先完善基本信息。")
        render_patient_profile_form("initial_patient_profile_form_page")
    st.stop()

# ==========================================
# 3. 顶部导航栏
# ==========================================
header_col1, header_col2, header_col3 = st.columns([7, 1, 2.5])
with header_col1:
    st.title("🏥 智能预问诊助手")

with header_col3:
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    with st.popover("👤 个人中心", use_container_width=True):
        st.markdown(f"**姓名:** {st.session_state.patient_name}")
        st.markdown(f"**ID:** `{st.session_state.patient_id}`")
        if st.session_state.get("patient_age"):
            st.markdown(f"**年龄:** {format_age_for_export(st.session_state.patient_age)}")
        if st.session_state.get("patient_sex"):
            st.markdown(f"**性别:** {st.session_state.patient_sex}")
        st.divider()
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# ========================================================
# 核心业务流转
# ========================================================
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

# 捕获病患输入并驱动 Agent 运转
if prompt := st.chat_input("请在此输入您的症状或回答..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner('🤖 正在分析病情与查阅既往史...'):

            # 🚀 动态检索长期记忆，并注入全局 State
            current_entities_for_memory = current_state.get("entities", [])
            long_term_memory_str = st.session_state.patient_memory.retrieve_memory(
                current_query=prompt,
                current_entities=current_entities_for_memory,
                top_k=2
            )
            # print('--------------------------------')
            # print(long_term_memory_str)
            # print('--------------------------------')
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

# 病历草稿生成及展示
if current_state.get("is_finished") is True and current_state.get("is_valid") is True and current_state.get(
        "draft_record"):
    st.success("病历草稿生成完毕，且已通过医学逻辑质控！")

    draft = current_state["draft_record"]
    chief_complaint = draft.get('chief_complaint', '')
    hpi = draft.get('history_of_present_illness', '')
    if not st.session_state.get("record_time"):
        st.session_state.record_time = format_minute_timestamp()

    st.markdown("### 📄 门诊病历草稿")
    st.markdown(f"**【主诉】**\n> {chief_complaint}")
    st.markdown(f"**【现病史】**\n> {hpi}")

    # 🚀 长期记忆写入：如果这轮对话没存过，则存档
    if not st.session_state.memory_saved:
        summary = f"主诉：{chief_complaint}。现病史：{hpi}"
        st.session_state.patient_memory.save_memory(summary)
        st.session_state.memory_saved = True

    export_text = f"门诊病历记录\n就诊人：{st.session_state.patient_name} (ID: {st.session_state.patient_id})\n\n【主诉】\n{chief_complaint}\n\n【现病史】\n{hpi}\n\n---\n由多智能体系统生成"

    # 🚀 获取当天精确日期用于防覆盖命名
    current_date = datetime.now().strftime("%Y_%m_%d")

    past_history = current_state.get("long_term_memory", "")
    patient_profile = st.session_state.patient_memory.get_patient_profile()
    preconsultation_data = build_preconsultation_data(patient_profile, draft, past_history)
    preconsultation_html = render_preconsultation_html(preconsultation_data)

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            label="💾 导出预问诊单至本地 (TXT格式)",
            data=export_text,
            file_name=f"{st.session_state.patient_id}_{current_date}_门诊病历.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with download_col2:
        with st.spinner("正在生成预问诊单 PDF..."):
            pdf_bytes, pdf_error = html_to_pdf_bytes(preconsultation_html)
            used_basic_pdf_fallback = False
            if not pdf_bytes:
                pdf_bytes = render_basic_preconsultation_pdf(preconsultation_data)
                used_basic_pdf_fallback = True

        if pdf_bytes:
            st.download_button(
                label="📄 下载预问诊单 (PDF格式)",
                data=pdf_bytes,
                file_name=f"{st.session_state.patient_id}_{current_date}_预问诊单.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            if used_basic_pdf_fallback:
                st.caption(f"未能调用浏览器转换 HTML，已生成基础版 PDF：{pdf_error}")
        else:
            st.warning(f"PDF 生成失败：{pdf_error}", icon="⚠️")

with st.sidebar:
    # 🚀 在侧边栏顶部新增：患者历史病历展示区域
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
                            review_meta = get_slot_review_meta(entity, slot_label)
                            st.markdown(
                                render_slot_answer_chip(slot_label, answer, review_meta),
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("当前还没有可展示的问诊要点答案。")



    else:
        st.info("等待病患输入并提取...")

    st.divider()
    if st.button("🗑️ 清空当前对话与记忆"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.memory_saved = False  # 🚀 解锁记忆保存限制
        st.session_state.pop("record_time", None)
        st.rerun()
