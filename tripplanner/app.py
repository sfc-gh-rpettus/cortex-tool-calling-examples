import json
import random
import re
import streamlit as st
import _snowflake
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete


# -- Configuration --
 
session = get_active_session()
LLM_MODEL = "claude-3-5-sonnet"
CORTEX_API_EP = "/api/v2/cortex/inference:complete"
REQUEST_HEADERS = {"Accept": "application/json, text/event-stream"}

TOOLS = [
    {
        "tool_spec": {
            "type": "generic",
            "name": "get_weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                },
                "required": ["lat", "lon"],
            },
        }
    },
    {
        "tool_spec": {
            "type": "generic",
            "name": "get_activities",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        }
    },
    {
        "tool_spec": {
            "type": "generic",
            "name": "get_dining",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        }
    },
]

# System prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Tools available:\n"
        "- get_weather(lat, lon): returns weather summary\n"
        "- get_activities(location): lists fun activities\n"
        "- get_dining(location): lists dining options\n\n"
        "Use only the necessary tools and emit calls programmatically."
        "If the user asks an off-topic quesiton, politely decline and describe what you can help with."
    )
}


# -- Tool implementations --

def get_weather(lat: float, lon: float) -> str:
    try:
        temp = random.randint(30, 100)
        cond = random.choice(["sunny", "cloudy", "rainy", "windy", "snowy"])
        return f"The current temperature at ({lat}, {lon}) is {temp}°F with {cond} conditions."
    except Exception as e:
        return f"[Error in get_weather: {e}]"


def get_activities(location: str) -> list[str]:
    try:
        prompt = f"List 10 fun activities in {location}, output as a JSON array of strings."
        resp   = complete(LLM_MODEL, prompt, session=session)
        return json.loads(resp)
    except Exception as e:
        return [f"[Error in get_activities: {e}]"]


def get_dining(location: str) -> list[str]:
    try:
        prompt = f"List 10 popular restaurants or cafes in {location}, output as a JSON array of strings."
        resp   = complete(LLM_MODEL, prompt, session=session)
        return json.loads(resp)
    except Exception as e:
        return [f"[Error in get_dining: {e}]"]


# -- Cortex calls and event processing --

def call_cortex_api(messages: list[dict], event_callback=None) -> list[dict]:
    resp = _snowflake.send_snow_api_request(
        "POST", CORTEX_API_EP, REQUEST_HEADERS,
        {}, 
        {
            "model":      LLM_MODEL,
            "messages":   messages,
            "tools":      TOOLS,
            "max_tokens": 4096,
            "top_p":      1,
            "stream":     True,
        },
        None, 60_000
    )
    if resp["status"] != 200:
        raise RuntimeError(f"Cortex HTTP {resp['status']} – {resp.get('reason')}\n"
                           f"{resp.get('content','')[:300]}")

    raw = resp["content"]
    events = []

    # 1) First pass: parse SSE frames
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data:"):
            continue
        payload = chunk[len("data:"):].strip()
        if payload and payload != "[DONE]":
            ev = json.loads(payload)
            events.append(ev)
            if event_callback:
                event_callback(ev)

    # 2) If we found no SSE frames, treat the entire body as JSON array
    if not events:
        arr = json.loads(raw)
        for ev in arr:
            if event_callback:
                event_callback(ev)
        return arr

    return events



# -- Helper functions to extract tool_use & assistant text --

def extract_tool_use(events: list[dict]) -> dict | None:
    name = tuid = None
    frags = []
    collecting = False

    for ev in events:
        data    = ev.get("data", ev)
        choices = data.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})

        if delta.get("type") == "tool_use" and "tool_use_id" in delta:
            name, tuid, collecting = delta["name"], delta["tool_use_id"], True
        elif collecting and delta.get("type") == "tool_use" and "input" in delta:
            frags.append(delta["input"])

    if not (name and tuid):
        return None

    raw_input = "".join(frags)
    try:
        inp = json.loads(raw_input)
    except json.JSONDecodeError:
        parts = raw_input.strip('{}" ').split(',')
        inp = {}
        for p in parts:
            if ':' in p:
                k, v = p.split(':', 1)
                key = k.strip().strip('"')
                val = v.strip().strip('" ')
                inp[key] = float(val) if re.fullmatch(r"-?\d+(\.\d+)?", val) else val

    return {"tool_use_id": tuid, "name": name, "input": inp}


def extract_text(events: list[dict]) -> str:
    out = ""
    for ev in events:
        data    = ev.get("data", ev)
        choices = data.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        if "content" in delta:
            out += delta["content"]
    return out


# -- Agent chain logic with status & event callbacks --

def run_agent_chain(
    messages: list[dict],
    status_callback=None,
    event_callback=None) -> tuple[list[dict], str]:
    convo = messages.copy()
    while True:
        if status_callback:
            status_callback("💭 Thinking…")
        events = call_cortex_api(convo, event_callback=event_callback)
        invocation = extract_tool_use(events)

        # No more tools → final answer
        if invocation is None:
            final = extract_text(events)
            return convo + [{"role": "assistant", "content": final}], final

        # Tool invocation
        tool = invocation["name"]
        if status_callback:
            status_callback(f"⚙️ Running tool: {tool}")

        inp = invocation["input"]
        if tool == "get_weather":
            res  = get_weather(inp.get("lat"), inp.get("lon"))
            text = res
        elif tool == "get_activities":
            loc  = inp.get("location") if isinstance(inp, dict) else inp
            acts = get_activities(loc)
            text = json.dumps(acts)
        elif tool == "get_dining":
            loc    = inp.get("location") if isinstance(inp, dict) else inp
            diners = get_dining(loc)
            text   = json.dumps(diners)
        else:
            text = f"[No handler for '{tool}']"

        # Echo back invocation & result (non-empty content)
        convo.append({
            "role": "assistant",
            "content_list": [{"type":"tool_use","tool_use":invocation}],
            # ensure a non-empty content field for API validation
            "content": f"Calling {tool}"
        })
        convo.append({
            "role": "user",
            "content_list": [{
                "type":"tool_results",
                "tool_results":{
                    "tool_use_id": invocation["tool_use_id"],
                    "name":        tool,
                    "content":   [{"type":"text","text": text}],
                }
            }],
            "content": f"Results of {tool}"
        })


# 
# Streamlit Chat UI
# 

def main():
    st.set_page_config(page_title="Weather & Activities Bot", page_icon="💬")
    st.title("💬 Weather & Activities Chatbot")

    # Description
    st.markdown(
        "This app can: ☀️ **Check weather**, 🏞️ **List activities**, "
        "🍽️ **Find dining options**\n\n"
        "Select a quick example or type your own question below."
    )

    # Sidebar: Quick Questions + Clear Chat
    with st.sidebar:
        st.header("⚡ Quick Questions")
        quick = None
        if st.button("☀️ Weather in Denver, CO"):
            quick = "What is the weather in Denver, CO?"
        if st.button("🏞️ Activities in Boulder, CO"):
            quick = "List fun activities in Boulder, CO."
        if st.button("🍽️ Dining in New York, NY"):
            quick = "List popular restaurants in New York, NY."
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.history = [SYSTEM_PROMPT]

    # Initialize history
    if "history" not in st.session_state:
        st.session_state.history = [SYSTEM_PROMPT]

    # Render existing chat
    for msg in st.session_state.history:
        if msg["role"] in ("user","assistant") and "content_list" not in msg:
            st.chat_message(msg["role"]).markdown(msg["content"])

    # Placeholders
    thought_box = st.empty()
    status_box  = st.empty()

    # ALWAYS show chat_input
    user_typed = st.chat_input("Type your question…")
    # Decide final prompt
    prompt = quick or user_typed

    if prompt:
        start_idx = len(st.session_state.history)

        # Append & render user
        st.session_state.history.append({"role":"user","content":prompt})
        st.chat_message("user").markdown(prompt)

        # Streamed chain‐of‐thought callback
        thought_buffer = ""
        def show_thought(ev):
            nonlocal thought_buffer
            data = ev.get("data", ev)
            choices = data.get("choices", [])
            if not choices: return
            delta = choices[0].get("delta", {})
            txt = delta.get("content","")
            if txt:
                thought_buffer += txt
                thought_box.markdown(f"💡 **LLM thinking:** {thought_buffer}")

        # Run the agent
        with st.spinner("Processing…"):
            updated, reply = run_agent_chain(
                st.session_state.history,
                status_callback=status_box.text,
                event_callback=show_thought
            )

        # Clear placeholders
        status_box.empty()
        thought_box.empty()

        # Update & render assistant
        st.session_state.history = updated
        st.chat_message("assistant").markdown(reply)

        # Expander: this turn’s tools
        with st.expander("🔍 Behind the scenes"):
            for msg in st.session_state.history[start_idx:]:
                if "content_list" not in msg: continue
                for part in msg["content_list"]:
                    if part["type"] == "tool_use":
                        tu = part["tool_use"]
                        st.markdown(f"▶️ **Invoked** `{tu['name']}`")
                        st.json(tu["input"])
                    elif part["type"] == "tool_results":
                        tr = part["tool_results"]
                        st.markdown(f"✅ **Result from** `{tr['name']}`")
                        st.write("".join(c.get("text","") for c in tr["content"]))
 

if __name__ == "__main__":
    main()
