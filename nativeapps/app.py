import datetime
import json
import re
import base64
import streamlit as st
import _snowflake
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete
from fpdf import FPDF

# -- Configuration ------------------------------------------------------------

session         = get_active_session()
LLM_MODEL       = "claude-3-5-sonnet"
CORTEX_API_EP   = "/api/v2/cortex/inference:complete"
REQUEST_HEADERS = {"Accept": "application/json, text/event-stream"}

TOOLS = [
    {
        "tool_spec": {
            "type": "generic",
            "name": "landing_lens_inference",
            "input_schema": {
                "type": "object",
                "properties": {"file": {"type": "string"}},
                "required": ["file"],
            },
        }
    },
    {
        "tool_spec": {
            "type": "generic",
            "name": "supply_chain_downstream_impact",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sku_name": {"type": "string"},
                    "site_name": {"type": "string"},
                },
                "required": ["sku_name", "site_name"],
            },
        }
    },
    {
        "tool_spec": {
            "type": "generic",
            "name": "generate_pdf_report",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file":    {"type": "string"},
                    "history": {"type": "string"},
                },
                "required": ["file", "history"],
            },
        }
    }
]

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are an automotive parts analysis assistant.\n\n"
        "🔧 **Available tools (one per user question):**\n"
        "1. `landing_lens_inference(file)`: Detect defects on the selected part image.\n"
        "2. `supply_chain_downstream_impact(sku_name, site_name)`: Analyze downstream supply chain impact. Only use if the user explicitly asks for “impact.”\n"
        "3. `generate_pdf_report(file, history)`: Generate a PDF report embedding the image **and** a summary of the entire conversation history. Only use if the user explicitly asks for a “report.”\n\n"
        "🚨 **Rules (must follow exactly):**\n"
        "- Call **exactly one** tool per user question.\n"
        "- If the question is about defects only, call `landing_lens_inference` and stop.\n"
        "- If the question mentions “impact,” call `supply_chain_downstream_impact` and stop.\n"
        "- If the question mentions “report” or “PDF,” call `generate_pdf_report` and stop.\n"
        "- If the question is unrelated, politely decline and call no tools.\n"
        "- Always emit exactly one JSON `tool_use` with proper input, then end your response."
    )
}

# -- Helpers ------------------------------------------------------------------

def get_sku_site(selected_image: str) -> str:
    sql = f"""
    SELECT
      sk.name AS SKU_NAME,
      s.name  AS SITE_NAME
    FROM RAI_DEMO.RAI_LAI_MANUFACTURING.IMAGES i
    LEFT JOIN RAI_DEMO.RAI_LAI_MANUFACTURING.SITE s  ON s.id  = i.site
    LEFT JOIN RAI_DEMO.RAI_LAI_MANUFACTURING.SKU  sk ON sk.id = i.sku
    WHERE i.file_key = '{selected_image}'
    """
    df = session.sql(sql).to_pandas()
    return df.to_json(orient="records")

# -- Tool implementations -----------------------------------------------------

def landing_lens_inference(file: str) -> str:
    sql = f"""
    SELECT LANDINGLENS__VISUAL_AI_PLATFORM_NON_COMMERCIAL_USE.core.run_inference(
      '{file}', '12987d0c-6915-4b79-9e25-cabecd2fe85f'
    ) AS inference
    """
    df = session.sql(sql).to_pandas()
    return str(df["INFERENCE"][0])

@st.cache_data
def supply_chain_downstream_impact(sku_name: str, site_name: str) -> str:
    session.sql(
        f"CALL RAI_DEMO.RAI_LAI_MANUFACTURING.impact_analysis_cache("
        f"'{sku_name}', '{site_name}', 'RAI_DEMO.RAI_LAI_MANUFACTURING.GET_IMPACT_RESULT')"
    ).collect()
    df = session.sql(
        "SELECT * FROM RAI_DEMO.RAI_LAI_MANUFACTURING.GET_IMPACT_RESULT"
    ).to_pandas()
    return df.to_json(orient="records") if not df.empty else "[]"

def generate_pdf_report(file: str, history: str) -> str:
    # 1) Summarize the chat history
    prompt = (
        "You are a manufacturing analyst. Summarize the following chat into a concise, "
        "well-structured markdown report with headings and bullet points:\n\n"
        f"{history}"
    )
    summary_md = complete(LLM_MODEL, prompt, session=session)

    # 2) Build the PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Defect Analysis Report", ln=True, align="C")
    pdf.ln(5)

    # Embed the image with more vertical space
    try:
        img_data = session.file.get_stream(file, decompress=False).read()
        tmp = "/tmp/part.jpg"
        with open(tmp, "wb") as f:
            f.write(img_data)
        pdf.image(tmp, x=50, y=30, w=110, h=80)
    except:
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, "[Failed to embed image]", ln=True)
    pdf.ln(90)

    # Write the markdown summary
    pdf.set_font("Helvetica", "", 12)
    for line in summary_md.splitlines():
        if line.startswith("##"):
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 8, line.replace("##", "").strip(), ln=True)
            pdf.ln(1)
            pdf.set_font("Helvetica", "", 12)
        else:
            pdf.multi_cell(0, 6, line)

    # Footer
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(
        0, 6,
        datetime.datetime.now().strftime("Generated on %Y-%m-%d %H:%M:%S"),
        ln=True, align="R"
    )

    raw = pdf.output(dest="S").encode("latin1")
    return base64.b64encode(raw).decode("ascii")

# -- Cortex integration -------------------------------------------------------

def call_cortex_api(messages, event_callback=None):
    resp = _snowflake.send_snow_api_request(
        "POST", CORTEX_API_EP, REQUEST_HEADERS, {},
        {"model":LLM_MODEL, "messages":messages, "tools":TOOLS,
         "max_tokens":4096, "top_p":1, "stream":True},
        None, 60_000
    )
    if resp["status"] != 200:
        raise RuntimeError(f"Cortex HTTP {resp['status']} – {resp.get('reason')}")
    raw = resp["content"]
    events = []
    for chunk in raw.split("\n\n"):
        if not chunk.startswith("data:"):
            continue
        payload = chunk[len("data:"):].strip()
        if payload and payload != "[DONE]":
            ev = json.loads(payload)
            events.append(ev)
            if event_callback:
                event_callback(ev)
    if not events:
        arr = json.loads(raw)
        for ev in arr:
            if event_callback:
                event_callback(ev)
        return arr
    return events

def extract_tool_use(events):
    name = tuid = None
    frags = []
    active = False
    for ev in events:
        data = ev.get("data", ev)
        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("type") == "tool_use" and "tool_use_id" in delta:
                name, tuid, active = delta["name"], delta["tool_use_id"], True
            elif active and delta.get("type") == "tool_use" and "input" in delta:
                frags.append(delta["input"])
    if not (name and tuid):
        return None
    raw = "".join(frags)
    try:
        inp = json.loads(raw)
    except:
        inp = {}
        for p in raw.strip('{}" ').split(','):
            if ':' in p:
                k, v = p.split(':', 1)
                key = k.strip().strip('"')
                val = v.strip().strip('" ')
                inp[key] = float(val) if re.fullmatch(r"-?\d+(\.\d+)?", val) else val
    return {"tool_use_id": tuid, "name": name, "input": inp}

def extract_text(events):
    out = ""
    for ev in events:
        data = ev.get("data", ev)
        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            if "content" in delta:
                out += delta["content"]
    return out

def run_agent_chain(messages, status_callback=None, event_callback=None):
    convo = list(messages)
    while True:
        if status_callback:
            status_callback("💭 Thinking…")
        events = call_cortex_api(convo, event_callback=event_callback)
        invocation = extract_tool_use(events)
        if invocation is None:
            final = extract_text(events)
            return convo + [{"role":"assistant","content":final}], final

        tool = invocation["name"]
        if status_callback:
            status_callback(f"⚙️ Running tool: {tool}")
        inp = invocation["input"]

        if tool == "landing_lens_inference":
            text = landing_lens_inference(inp["file"])
        elif tool == "supply_chain_downstream_impact":
            text = supply_chain_downstream_impact(
                inp["sku_name"], inp["site_name"]
            )
        elif tool == "generate_pdf_report":
            full_hist = json.dumps(convo, default=str)
            text = generate_pdf_report(inp["file"], full_hist)
        else:
            text = f"[No handler for '{tool}']"

        convo.append({
            "role": "assistant",
            "content_list": [{"type":"tool_use","tool_use":invocation}],
            "content": f"Calling {tool}"
        })
        convo.append({
            "role": "user",
            "content_list": [{
                "type": "tool_results",
                "tool_results": {
                    "tool_use_id": invocation["tool_use_id"],
                    "name":        tool,
                    "content":     [{"type":"text","text": text}]
                }
            }],
            "content": f"Results of {tool}"
        })

# -- Streamlit Chat UI --------------------------------------------------------

def main():
    st.set_page_config(page_title="🚗 Auto Manufacturing Defect Analysis", page_icon="🏭")
    st.title("🚗 Auto Manufacturing Defect Analysis Chatbot 🏭")

    # Sidebar with instructions + quick actions
    with st.sidebar:
        st.title("🛠️ Cortex Tool Calling")
        st.markdown("""
#### Cortex Tool Calling with Chain-of-Thought
- Computer Vision (LandingAI)
- Graph Reasoning (RelationalAI)

**Usage Notes**
- First graph analysis takes 2–3 minutes; subsequent calls are faster.
- Conversation history is preserved per session.

        """)
        st.divider()

    if "history" not in st.session_state:
        st.session_state.history = [SYSTEM_PROMPT]

    with st.sidebar:
        # Quick-buttons
        st.markdown("""
        ### Quick Questions
        *Here are some questions to get you started:* 
        """)
        
        quick = None
        if st.button("📋 Is this part defective?"):
            quick = "Is this part defective?"
        if st.button("🔖 What is the SKU and site value?"):
            quick = "What is the SKU and site value?"
        if st.button("📊 What is the downstream impact?"):
            quick = "What is the downstream impact of this defect?"
        if st.button("📝 Generate PDF report"):
            quick = "Generate a PDF report of findings"
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.history = [SYSTEM_PROMPT]
            st.session_state["last_file"] = None

    # Image selection
    selected_image = st.selectbox(
        "Select an image",
        [
            'defects/cast_def_0_33.jpeg','defects/cast_def_0_40.jpeg','defects/cast_def_0_0.jpeg',
            'defects/cast_def_0_105.jpeg','defects/cast_def_0_107.jpeg','defects/cast_def_0_11.jpeg',
            'defects/cast_def_0_110.jpeg','defects/cast_def_0_118.jpeg','defects/cast_def_0_133.jpeg',
            'defects/cast_def_0_144.jpeg','defects/cast_def_0_148.jpeg','no-defects/cast_ok_0_102.jpeg'
        ]
    )
    FILE     = f'@llens_sample_ds_manufacturing.ball_bearing.dataset/data/{selected_image}'
    sku_info = get_sku_site(selected_image)

    # Inject context if new file
    ctx = {"role":"system", "content": f"FILENAME = {FILE}\nSELECTED_PART_INFO = {sku_info}"}
    if st.session_state.get("last_file") != FILE:
        st.session_state.history.append(ctx)
        st.session_state["last_file"] = FILE

    st.image(
        session.file.get_stream(FILE, decompress=False).read(),
        caption=selected_image
    )

    # Render history
    for msg in st.session_state.history:
        if msg["role"] in ("user","assistant") and "content_list" not in msg:
            st.chat_message(msg["role"]).markdown(msg["content"])

    thought_box = st.empty()
    status_box  = st.empty()
    st.session_state["thought_buf"] = ""

    user_input = st.chat_input("Type your question…")
    prompt     = quick or user_input

    if prompt:
        start = len(st.session_state.history)
        st.session_state.history.append({"role":"user","content":prompt})
        st.chat_message("user").markdown(prompt)

        def show_thought(ev):
            data    = ev.get("data", ev)
            choices = data.get("choices", [])
            if not choices: return
            delta = choices[0].get("delta", {})

            # only show if we actually have a name
            if delta.get("type") == "tool_use" and delta.get("name"):
                thought_box.markdown(f"🔧 **Planning to call tool:** `{delta['name']}`")

            if "content" in delta:
                buf = st.session_state["thought_buf"] + delta["content"]
                st.session_state["thought_buf"] = buf
                thought_box.markdown(f"💡 **Thinking:** {buf}")

        updated, reply = run_agent_chain(
            st.session_state.history,
            status_callback=status_box.text,
            event_callback=show_thought
        )

        status_box.empty()
        thought_box.empty()
        st.session_state["thought_buf"] = ""

        st.session_state.history = updated
        st.chat_message("assistant").markdown(reply)

        # If PDF was generated, offer download
        for msg in reversed(updated):
            if "content_list" in msg:
                for part in msg["content_list"]:
                    if part["type"] == "tool_results" and part["tool_results"]["name"] == "generate_pdf_report":
                        b64 = "".join(c.get("text","") for c in part["tool_results"]["content"])
                        pdf_bytes = base64.b64decode(b64)
                        st.download_button(
                            "📥 Download PDF report",
                            data=pdf_bytes,
                            file_name="defect_report.pdf",
                            mime="application/pdf"
                        )
                        break
                else:
                    continue
                break

        with st.expander("🔍 Examine Tool Inputs/Outputs"):
            for msg in updated[start:]:
                if "content_list" not in msg:
                    continue
                for part in msg["content_list"]:
                    if part["type"] == "tool_use":
                        st.markdown(f"▶️ **Invoked** `{part['tool_use']['name']}`")
                        st.json(part["tool_use"]["input"])
                    elif part["type"] == "tool_results":
                        st.markdown(f"✅ **Result from** `{part['tool_results']['name']}`")
                        st.write("".join(c.get("text","") for c in part["tool_results"]["content"]))

if __name__ == "__main__":
    main()
