import datetime
import json
import re
import base64
import uuid
import unicodedata               
import pandas as pd
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
    { "tool_spec": { "type": "generic", "name": "landing_lens_inference",
        "input_schema": { "type": "object",
            "properties": { "file": {"type": "string"} },
            "required": ["file"] } } },
    { "tool_spec": { "type": "generic", "name": "supply_chain_downstream_impact",
        "input_schema": { "type": "object",
            "properties": {
                "sku_name":  {"type": "string"},
                "site_name": {"type": "string"},
            },
            "required": ["sku_name", "site_name"] } } },
    { "tool_spec": { "type": "generic", "name": "generate_pdf_report",
        "input_schema": { "type": "object",
            "properties": {
                "file":    {"type": "string"},
                "history": {"type": "string"},
            },
            "required": ["file", "history"] } } },
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
    ),
}

# -- Helpers ------------------------------------------------------------------

def _latin1(text: str) -> str:
    """Strip/convert chars so the result is safe for FPDF's Latin-1 font."""
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

def _truncate(text: str, limit: int = 1_000) -> str:
    return text if len(text) <= limit else f"[{len(text)} bytes omitted]"

def get_sku_site(selected_image: str) -> str:
    sql = f"""
    SELECT sk.name AS SKU_NAME, s.name AS SITE_NAME
      FROM RAI_DEMO.RAI_LAI_MANUFACTURING.IMAGES i
      LEFT JOIN RAI_DEMO.RAI_LAI_MANUFACTURING.SITE s
        ON s.id = i.site
      LEFT JOIN RAI_DEMO.RAI_LAI_MANUFACTURING.SKU sk
        ON sk.id = i.sku
     WHERE i.file_key = '{selected_image}'
    """
    return session.sql(sql).to_pandas().to_json(orient="records")

# -- Tool implementations -----------------------------------------------------

def landing_lens_inference(file: str) -> str:
    sql = f"""
    SELECT LANDINGLENS__VISUAL_AI_PLATFORM_NON_COMMERCIAL_USE.core.run_inference(
      '{file}', '12987d0c-6915-4b79-9e25-cabecd2fe85f'
    ) AS inference
    """
    return str(session.sql(sql).to_pandas()["INFERENCE"][0])

@st.cache_data
def supply_chain_downstream_impact(sku_name: str, site_name: str) -> str:
    session.sql(
        f"CALL RAI_DEMO.RAI_LAI_MANUFACTURING.impact_analysis_cache("
        f"'{sku_name}','{site_name}','RAI_DEMO.RAI_LAI_MANUFACTURING.GET_IMPACT_RESULT')"
    ).collect()
    df = session.sql("SELECT * FROM RAI_DEMO.RAI_LAI_MANUFACTURING.GET_IMPACT_RESULT").to_pandas()
    return df.to_json(orient="records") if not df.empty else "[]"

def generate_pdf_report(file: str, history: str) -> str:
    prompt = (
        "You are a manufacturing analyst. Summarise the following chat into a concise, "
        "well-structured markdown report with headings and bullet points:\n\n"
        f"{history}"
    )
    summary_md = complete(LLM_MODEL, prompt, session=session)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Defect Analysis Report", ln=True, align="C")
    pdf.ln(5)

    try:
        img_data = session.file.get_stream(file, decompress=False).read()
        tmp = "/tmp/part.jpg"
        with open(tmp, "wb") as f:
            f.write(img_data)
        pdf.image(tmp, x=50, y=30, w=110, h=90)
    except:
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, "[Failed to embed image]", ln=True)
    pdf.ln(100)

    pdf.set_font("Helvetica", "", 12)
    for line in summary_md.splitlines():
        safe = _latin1(line)
        if safe.startswith("##"):
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 8, safe.replace("##", "").strip(), ln=True)
            pdf.ln(1)
            pdf.set_font("Helvetica", "", 12)
        else:
            pdf.multi_cell(0, 6, safe)

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(
        0, 6, datetime.datetime.now().strftime("Generated on %Y-%m-%d %H:%M:%S"),
        ln=True, align="R"
    )

    raw = pdf.output(dest="S").encode("latin-1", "replace")
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
        if not chunk.startswith("data:"): continue
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
        for choice in ev.get("data", ev).get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("type")=="tool_use" and "tool_use_id" in delta:
                name, tuid, active = delta["name"], delta["tool_use_id"], True
            elif active and delta.get("type")=="tool_use" and "input" in delta:
                frags.append(delta["input"])
    if not (name and tuid):
        return None
    raw = "".join(frags)
    try:
        inp = json.loads(raw)
    except:
        inp = {}
        for p in raw.strip('{}" ').split(","):
            if ":" in p:
                k,v = p.split(":",1)
                inp[k.strip().strip('"')] = float(v) if re.fullmatch(r"-?\d+(\.\d+)?", v) else v
    return {"tool_use_id": tuid, "name": name, "input": inp}

def extract_text(events):
    out = ""
    for ev in events:
        for choice in ev.get("data", ev).get("choices", []):
            if "content" in choice.get("delta", {}):
                out += choice["delta"]["content"]
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
        inp  = invocation["input"]

        #if status_callback:
        #    status_callback(f"🔧 Planning to call tool: {tool}")

        # run with single spinner
        with st.spinner(f"⚙️ Running tool {tool}…"):
            if tool == "landing_lens_inference":
                text = landing_lens_inference(inp["file"])
            elif tool == "supply_chain_downstream_impact":
                text = supply_chain_downstream_impact(inp["sku_name"], inp["site_name"])
            elif tool == "generate_pdf_report":
                # generate PDF, store b64, but don't pass it back to the LLM
                b64 = generate_pdf_report(inp["file"], json.dumps(convo, default=str))
                st.session_state["last_pdf_b64"] = b64
                text = "📄 PDF report created"
            else:
                text = f"[No handler for '{tool}']"

        convo.append({
            "role":"assistant",
            "content_list":[{"type":"tool_use","tool_use":invocation}],
            "content":f"Calling {tool}"
        })
        convo.append({
            "role":"user",
            "content_list":[{
                "type":"tool_results","tool_results":{
                    "tool_use_id":invocation["tool_use_id"],
                    "name":tool,
                    "content":[{"type":"text","text":text}]
                }
            }],
            "content":f"Results of {tool}"
        })

# -- Feedback logging & session ID -------------------------------------------

def _ensure_session_id():
    if "_session_id" not in st.session_state:
        st.session_state["_session_id"] = str(uuid.uuid4())

def write_chat_to_sf(history_json, thinking_json, rating=None):
    _ensure_session_id()
    user_name = session.sql("SELECT CURRENT_USER()").collect()[0][0]
    df = pd.DataFrame([{
        "SESSION_ID": st.session_state["_session_id"],
        "USER_NAME":  user_name,
        "CHAT_HISTORY": history_json,
        "THINKING_TRACE": thinking_json,
        "RATING": rating
    }])
    session.write_pandas(df, table_name="FEEDBACK_LOG",
                         database="RAI_DEMO", schema="RAI_LAI_MANUFACTURING",
                         auto_create_table=True, overwrite=False)

# -- Streamlit Chat UI --------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="🚗 Auto Manufacturing Defect Analysis", page_icon="🏭")
    st.title("🚗 Auto Manufacturing Defect Analysis Assistant 🏭")
    _ensure_session_id()

    # Sidebar
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
    if "thought_trace" not in st.session_state:
        st.session_state.thought_trace = []

    with st.sidebar:
        st.markdown("### Quick Questions")
        quick = None
        if st.button("📋 Is this part defective?"): quick = "Is this part defective?"
        if st.button("🔖 What is the SKU and site value?"): quick = "What is the SKU and site value?"
        if st.button("📊 What is the downstream impact?"): quick = "What is the downstream impact of this defect?"
        if st.button("📝 Generate PDF report"): quick = "Generate a PDF report of findings"
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.history = [SYSTEM_PROMPT]
            st.session_state["last_file"] = None
            st.session_state.thought_trace = []
        st.divider()
        st.subheader("💾 Log this conversation")
        if st.button("Save chat to Snowflake"):
            write_chat_to_sf(json.dumps(st.session_state.history, default=str),
                            json.dumps(st.session_state.thought_trace, default=str))
            st.success("Chat saved!")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("👍 Mark good"):
                write_chat_to_sf(json.dumps(st.session_state.history, default=str),
                                json.dumps(st.session_state.thought_trace, default=str),
                                rating="good")
                st.success("Marked good ✔️")
        with c2:
            if st.button("👎 Mark bad"):
                write_chat_to_sf(json.dumps(st.session_state.history, default=str),
                                json.dumps(st.session_state.thought_trace, default=str),
                                rating="bad")
                st.success("Marked bad ✔️")

    # Image selection
    selected_image = st.selectbox("Select an image", [
        "defects/cast_def_0_33.jpeg","defects/cast_def_0_40.jpeg","defects/cast_def_0_0.jpeg",
        "defects/cast_def_0_105.jpeg","defects/cast_def_0_107.jpeg","defects/cast_def_0_11.jpeg",
        "defects/cast_def_0_110.jpeg","defects/cast_def_0_118.jpeg","defects/cast_def_0_133.jpeg",
        "defects/cast_def_0_144.jpeg","defects/cast_def_0_148.jpeg","no-defects/cast_ok_0_102.jpeg"
    ])
    FILE     = f"@llens_sample_ds_manufacturing.ball_bearing.dataset/data/{selected_image}"
    sku_info = get_sku_site(selected_image)
    ctx = {"role":"system","content":f"FILENAME = {FILE}\nSELECTED_PART_INFO = {sku_info}"}
    if st.session_state.get("last_file") != FILE:
        st.session_state.history.append(ctx)
        st.session_state["last_file"] = FILE

    st.image(session.file.get_stream(FILE, decompress=False).read(), caption=selected_image, width=500)

    # Render chat history
    for msg in st.session_state.history:
        if msg["role"] in ("user","assistant") and "content_list" not in msg:
            st.chat_message(msg["role"]).markdown(msg["content"])

    thought_box = st.empty()
    status_box  = st.empty()

    user_input = st.chat_input("Type your question…")
    prompt     = quick or user_input

    if prompt:
        start = len(st.session_state.history)
        st.session_state.history.append({"role":"user","content":prompt})
        st.chat_message("user").markdown(prompt)

        def show_thought(ev):
            delta = ev.get("data",ev).get("choices",[{}])[0].get("delta",{})
            if delta.get("type")=="tool_use" and delta.get("name"):
                thought_box.markdown(f"🔧 **Planning to call tool:** `{delta['name']}`")
            if "content" in delta:
                st.session_state.thought_trace.append(delta["content"])
                thought_box.markdown(f"💡 **Thinking:** {delta['content']}")

        updated, reply = run_agent_chain(
            st.session_state.history,
            status_callback=status_box.text,
            event_callback=show_thought
        )

        status_box.empty()
        thought_box.empty()

        st.session_state.history = updated
        st.chat_message("assistant").markdown(reply)

        # PDF download
        if st.session_state.get("last_pdf_b64"):
            pdf_bytes = base64.b64decode(st.session_state.pop("last_pdf_b64"))
            st.download_button(
                "📥 Download PDF report",
                data=pdf_bytes,
                file_name="defect_report.pdf",
                mime="application/pdf"
            )

        # Inspect tool I/O
        with st.expander("🔍 Examine Tool Inputs/Outputs"):
            for msg in updated[start:]:
                if "content_list" not in msg: continue
                for part in msg["content_list"]:
                    if part["type"]=="tool_use":
                        st.markdown(f"▶️ **Invoked** `{part['tool_use']['name']}`")
                        st.json(part["tool_use"]["input"])
                    elif part["type"]=="tool_results":
                        name = part["tool_results"]["name"]
                        st.markdown(f"✅ **Result from** `{name}`")
                        if name=="generate_pdf_report":
                            st.write("📄 PDF report created")
                        else:
                            text = part["tool_results"]["content"][0]["text"]
                            st.write(_truncate(text))

if __name__ == "__main__":
    main()
