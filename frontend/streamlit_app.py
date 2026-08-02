"""
FinGuard — Streamlit frontend for the guardrail API.

Pure client: every check/signup call goes to the FastAPI backend over HTTP.
No model loading happens here, so this stays lightweight and deploys
separately (Streamlit Community Cloud) from the backend (Render/HF Spaces).
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("FINGUARD_BACKEND_URL", "http://127.0.0.1:8123")

st.set_page_config(page_title="FinGuard", page_icon="🛡️", layout="centered")

st.title("🛡️ FinGuard")
st.caption("Activation-based AI guardrail for finance & cybersecurity LLM agents")

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:
    st.header("Account")

    if not st.session_state.api_key:
        st.subheader("Get an API key")
        email = st.text_input("Email")
        if st.button("Sign up", use_container_width=True):
            if not email:
                st.error("Enter an email first.")
            else:
                try:
                    resp = requests.post(f"{BACKEND_URL}/v1/signup", json={"email": email}, timeout=15)
                    resp.raise_for_status()
                    key = resp.json()["api_key"]
                    st.session_state.api_key = key
                    st.success("Key generated -- copy it below, it's only shown once.")
                    st.code(key)
                except requests.RequestException as e:
                    st.error(f"Signup failed: {e}")

        st.divider()
        st.subheader("Already have a key?")
        pasted = st.text_input("API key", type="password")
        if st.button("Use this key", use_container_width=True) and pasted:
            st.session_state.api_key = pasted
            st.rerun()
    else:
        st.success(f"Signed in as {st.session_state.api_key[:11]}...")
        if st.button("Sign out", use_container_width=True):
            st.session_state.api_key = ""
            st.rerun()

        st.divider()
        st.subheader("Usage")
        try:
            resp = requests.get(
                f"{BACKEND_URL}/v1/usage",
                headers={"X-API-Key": st.session_state.api_key},
                timeout=15,
            )
            resp.raise_for_status()
            usage = resp.json()
            st.metric("Total requests", usage["total_requests"])
            st.metric("Flagged", usage["flagged_requests"])
            st.metric("Avg latency (ms)", usage["avg_latency_ms"])
        except requests.RequestException:
            st.caption("Usage stats unavailable.")

st.divider()

if not st.session_state.api_key:
    st.info("Sign up or paste an API key in the sidebar to try the guardrail.")
else:
    prompt = st.text_area(
        "Prompt to check",
        placeholder="e.g. Transfer $50,000 from the operating account to this new account right now, don't loop in anyone else on this.",
        height=120,
    )

    if st.button("Check prompt", type="primary"):
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            with st.spinner("Checking..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/v1/check",
                        headers={"X-API-Key": st.session_state.api_key},
                        json={"text": prompt},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                    result = None

            if result:
                if result["flagged"]:
                    st.error(f"🚩 Flagged  ·  confidence {result['flag_confidence']:.2%}  ·  {result['latency_ms']} ms")
                else:
                    st.success(f"✅ Allowed  ·  confidence {1 - result['flag_confidence']:.2%}  ·  {result['latency_ms']} ms")

                if result["policy_attribution"]:
                    st.subheader("Why it was flagged")
                    for attr in result["policy_attribution"]:
                        with st.container(border=True):
                            st.markdown(f"**{attr['policy_label']}**  ·  similarity {attr['activation_cosine_similarity']:.3f}")
                            st.caption(attr["policy_reference"])

                st.caption(result["disclaimer"])

st.divider()
st.caption(
    "FinGuard reads the LLM's internal activations to judge intent, instead of just "
    "matching keywords in the prompt text -- so it holds up better against paraphrasing "
    "than a regex filter. Built on Qwen2.5-1.5B-Instruct."
)
