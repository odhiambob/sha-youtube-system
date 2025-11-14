# app.py
# SHA YouTube Sentiment Analyzer — AfriBERTa (Fine-Tuned, Local)
# Focus: KPIs, Media Framing, and Social Sharing.

import os
import re
import time
import urllib.parse
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

# =============================
# 0) CONFIG (MODERNIZED & BALANCED)
# =============================
st.set_page_config(
    page_title="SHA Sentiment Analyzer",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="auto"
)

# Define the colors used for the KPI box backgrounds for consistency
KPI_COLORS = {
    "negative": "#ffebee", # Light Red
    "neutral": "#f5f5f5",  # Light Grey
    "positive": "#e8f5e9", # Light Green
    "total": "#e0f7fa"     # Light Cyan
}

# Custom CSS for modern/appealing look and COLORED KPI boxes
st.markdown(f"""
<style>
    /* Gradient Title */
    .big-font {{
        font-size:3em !important;
        font-weight: 700;
        text-align: left;
        color: #262730; 
        margin-bottom: -15px; 
    }}
    .subtitle-font {{
        font-size:1.2em !important;
        font-weight: 400;
        color: #8C8C9A; 
        margin-bottom: 20px;
    }}
    
    /* NEW: Custom KPI Box Styling */
    .kpi-box {{
        padding: 20px;
        border-radius: 12px; /* Smoother corners */
        text-align: left;
        color: #1a1a1a;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Subtle shadow for depth */
        margin-bottom: 15px;
        height: 100%; /* Ensure all boxes are the same height */
    }}

    /* Background Colors for specific sentiment - USING DEFINED COLORS */
    .kpi-total {{ background-color: {KPI_COLORS["total"]}; border-left: 5px solid #00bcd4; }} /* Cyan for Total */
    .kpi-negative {{ background-color: {KPI_COLORS["negative"]}; border-left: 5px solid #ff5252; }} /* Light Red for Negative */
    .kpi-neutral {{ background-color: {KPI_COLORS["neutral"]}; border-left: 5px solid #bdbdbd; }} /* Grey for Neutral */
    .kpi-positive {{ background-color: {KPI_COLORS["positive"]}; border-left: 5px solid #4caf50; }} /* Light Green for Positive */

    .kpi-label {{ font-size: 0.9em; color: #555555; margin: 0; }}
    .kpi-value {{ font-size: 2.2em; font-weight: 600; margin: 0; }}
    .kpi-percent {{ font-size: 1em; font-weight: 500; margin: 0; }}

    /* Streamlit overrides */
    .stButton>button {{
        border-radius: 8px;
        border: 1px solid #4CAF50; 
        background-color: #4CAF50;
    }}
</style>
""", unsafe_allow_html=True)


YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""))

USE_FINETUNED = True
# Keep your specific local path
DEFAULT_FINETUNED_PATH = r"C:\Users\HP\Desktop\usiu\afriberta_ft_ckpt\checkpoint-90"
BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Link mode uses an internal cap for speed/quota
LINK_COMMENTS_LIMIT = 400

OUTLETS = ["Citizen TV Kenya", "NTV Kenya", "KTN News", "KBC Channel 1", "TV47 Kenya"]

SHA_KEYWORDS = [
    "sha",
    "social health authority",
    "social health insurance",
    "shif",
    "social health insurance act",
    "social health insurance fund",
    "nhif to sha",
    "uhc kenya",
    "health authority kenya",
]

# Rule cues for title rationale (plain-language explanation)
FRAME_RULES = {
    "economic": [
        "cost", "costs", "afford", "affordability", "burden", "premium", "premiums",
        "fee", "fees", "levy", "levies", "penalty", "penalties", "payment", "payments",
        "tax", "taxes", "deduct", "deduction", "charge", "charges", "funding", "budget",
        "financing", "expensive", "price", "prices"
    ],
    "governance": [
        "corruption", "graft", "accountability", "transparency", "illicit", "illegal",
        "fraud", "embezzle", "embezzlement", "scandal", "investigation", "probe",
        "audit", "auditor", "court", "ruling", "judge", "judiciary", "petition",
        "tribunal", "arrest", "arrested", "charged", "charge sheet", "ethics", "eacc",
        "dci", "abuse of office", "misconduct"
    ],
    "implementation": [
        "rollout", "roll-out", "registration", "register", "portal", "system", "systems",
        "deadline", "deadlines", "capacity", "guidelines", "policy",
        "phased", "pilot", "deployment", "onboarding", "queue", "downtime", "outage",
        "training", "logistics", "readiness", "framework", "implementation"
    ],
    "citizen welfare": [
        "access", "accessibility", "quality", "equity", "coverage", "benefit", "benefits",
        "service", "services", "patients", "patient", "hospital", "hospitals", "clinics",
        "facility", "facilities", "care", "treatment", "medicines", "drugs", "referrals",
        "affordable care", "universal health", "waiting time", "experience"
    ],
}

ID2LABEL_FALLBACK = {0: "negative", 1: "neutral", 2: "positive"}

SEARCH_QUERIES = [
    "Social Health Authority Kenya",
    "SHA Kenya policy",
    "Social Health Insurance Kenya",
    "SHIF Kenya",
    "NHIF to SHA Kenya",
]

# =============================
# Session helpers
# =============================
def _reset_run():
    for k in [
        "ready", "table_df", "videos", "video_meta", "date_range",
        "limit_n", "mode", "model_path", "in_run", "prev_mode",
        "show_adv_table", "feedback_given"  # Reset feedback on new run
    ]:
        st.session_state.pop(k, None)

if "in_run" not in st.session_state:
    st.session_state["in_run"] = False

# =============================
# 1) HELPERS
# =============================
def normalize_local_path(p: str) -> Optional[str]:
    if not p:
        return None
    return str(Path(str(p))).replace("\\", "/")

def resolve_ft_path() -> Optional[str]:
    candidates = [
        st.secrets.get("AFRIBERTA_FT_PATH") if hasattr(st, "secrets") else None,
        os.getenv("AFRIBERTA_FT_PATH"),
        DEFAULT_FINETUNED_PATH,
    ]
    for c in candidates:
        if c and str(c).strip():
            return normalize_local_path(c)
    return None

def is_sha_related(text: str) -> bool:
    return bool(text) and any(k in (text or "").lower() for k in SHA_KEYWORDS)

def detect_title_focus_and_keywords(title: str) -> Tuple[str, List[str]]:
    t = (title or "").lower()
    scores = {f: sum(1 for kw in kws if kw in t) for f, kws in FRAME_RULES.items()}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        matched = [kw for kw in FRAME_RULES[best] if kw in t][:8]
        return best, matched
    return "other", []

def frame_label_for_display(key: str) -> str:
    mapping = {
        "citizen welfare": "Citizen Welfare (Access & Quality)",
        "economic": "Economic (Cost & Funding)",
        "governance": "Governance (Integrity & Trust)",
        "implementation": "Implementation (Rollout & Systems)",
        "other": "Other / Unclear",
    }
    return mapping.get(key, "Other / Unclear")

def frame_meaning_text(key: str) -> str:
    if key == "citizen welfare":
        return "focuses on people’s care experience such as **access, quality, equity, and benefits**."
    if key == "economic":
        return "focuses on affordability and money such as **costs, charges, levies, and funding**."
    if key == "governance":
        return "focuses on integrity and oversight including **investigations, audits, legal processes, and public trust**."
    if key == "implementation":
        return "focuses on day to day rollout including **registration, systems, deadlines, capacity, and readiness**."
    return "has **no clear emphasis** on cost, trust, rollout, or everyday patient experience."

def parse_video_id_from_url(url: str):
    for p in [r"v=([A-Za-z0-9_-]{11})", r"youtu\.be/([A-Za-z0-9_-]{11})"]:
        m = re.search(p, url or "")
        if m:
            return m.group(1)
    return None

# Robust HTTP with retries/backoff
def http_get_with_retries(url: str, params: Dict, timeout: int = 60, max_retries: int = 5) -> requests.Response:
    backoff = 1.5
    delay = 1.0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if 500 <= resp.status_code < 600:
                raise requests.HTTPError(f"Server error {resp.status_code}", response=resp)
            return resp
        except (requests.ReadTimeout, requests.ConnectTimeout, requests.HTTPError, requests.ConnectionError) as e:
            last_exc = e
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay *= backoff
    if isinstance(last_exc, requests.HTTPError) and last_exc.response is not None:
        raise last_exc
    raise requests.ReadTimeout(f"GET {url} failed after {max_retries} retries: {last_exc}")

def yt_get(path: str, params: dict):
    base = "https://www.googleapis.com/youtube/v3/"
    params = {**params, "key": YOUTUBE_API_KEY}
    r = http_get_with_retries(base + path, params=params, timeout=60, max_retries=5)
    r.raise_for_status()
    return r.json()

def fetch_video_metadata(video_id: str):
    data = yt_get("videos", {"part": "snippet,statistics", "id": video_id})
    items = data.get("items", [])
    return items[0] if items else None

def fetch_comments(video_id: str, limit: int = 20):
    comments, token = [], None
    while len(comments) < limit:
        batch = yt_get("commentThreads", {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, limit - len(comments)),
            "pageToken": token or ""
        })
        for it in batch.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "author": s.get("authorDisplayName"),
                "text": s.get("textOriginal", ""),
                "likes": s.get("likeCount", 0),
                "time": s.get("publishedAt", ""),
            })
            if len(comments) >= limit:
                break
        token = batch.get("nextPageToken")
        if not token:
            break
    return comments

def _standardize_label(raw_label, id2label_map: dict):
    if not raw_label:
        return None
    lbl = str(raw_label)
    if lbl.startswith("LABEL_"):
        try:
            idx = int(lbl.split("_")[-1])
            return id2label_map.get(idx, ID2LABEL_FALLBACK.get(idx, lbl))
        except Exception:
            return lbl
    lwr = lbl.lower().strip()
    if lwr in {"negative", "neutral", "positive"}:
        return lwr
    return lbl

def confidence_badge(score: float) -> str:
    if score is None:
        return "—"
    if score >= 0.80:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"

def search_sha_videos(start_dt_utc: datetime, end_dt_utc: datetime, max_videos: int = 10):
    videos = []
    seen = set()
    for q in SEARCH_QUERIES:
        if len(videos) >= max_videos:
            break
        params = {
            "part": "snippet",
            "type": "video",
            "q": q,
            "maxResults": 50,
            "order": "date",
            "publishedAfter": start_dt_utc.isoformat().replace("+00:00", "Z"),
            "publishedBefore": end_dt_utc.isoformat().replace("+00:00", "Z"),
        }
        data = yt_get("search", params)
        for it in data.get("items", []):
            vid = it["id"]["videoId"]
            if vid in seen:
                continue
            sn = it["snippet"]
            title = sn.get("title", "")
            desc = sn.get("description", "")
            if not is_sha_related(title + "\n" + desc):
                continue
            videos.append({
                "video_id": vid,
                "title": title,
                "publishedAt": sn.get("publishedAt", ""),
                "channelTitle": sn.get("channelTitle", ""),
            })
            seen.add(vid)
            if len(videos) >= max_videos:
                break
    return videos

# ====== ensure review columns exist for full table ======
def ensure_review_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["human_label", "reviewer", "action"]:
        if c not in df.columns:
            df[c] = ""
    return df

# =============================
# 2) MODEL LOADER
# =============================
@st.cache_resource(show_spinner=False)
def load_pipeline():
    if USE_FINETUNED:
        local_dir = resolve_ft_path()
        if not local_dir:
            raise ValueError("AFRIBERTA_FT_PATH not found; set it in secrets or use DEFAULT_FINETUNED_PATH.")
        p = Path(local_dir)
        if not p.exists():
            raise FileNotFoundError(f"AfriBERTa folder not found: {local_dir}")

        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True)
        pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, truncation=True)
        id2label = getattr(mdl.config, "id2label", None) or ID2LABEL_FALLBACK
        if isinstance(id2label, dict):
            id2label_map = {int(k): v for k, v in id2label.items()}
        else:
            id2label_map = ID2LABEL_FALLBACK
        return pipe, id2label_map, local_dir
    else:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
        pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, truncation=True)
        id2label = getattr(mdl.config, "id2label", None) or ID2LABEL_FALLBACK
        id2label_map = {int(k): v for k, v in (id2label.items() if isinstance(id2label, dict) else ID2LABEL_FALLBACK.items())}
        return pipe, id2label_map, BASE_MODEL

def run_sentiment(pipe, id2label_map, texts: List[str]):
    outs = pipe(texts, batch_size=16)
    results = []
    for o in outs:
        item = o[0] if isinstance(o, list) else o
        raw_label = item.get("label")
        std_label = _standardize_label(raw_label, id2label_map)
        score = float(item["score"])
        results.append({"label": std_label, "score": score, "confidence": confidence_badge(score)})
    return results

# =============================
# 3) SETUP UI (MODERNIZED)
# =============================
if not st.session_state.get("in_run", False) and not st.session_state.get("ready", False):
    
    # Custom Styled Header
    st.markdown('<p class="big-font">SHA YouTube Sentiment Analyzer 🇰🇪</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-font">Analyzing Public Opinion on Social Health Authority (SHA) in Kenya.</p>', unsafe_allow_html=True)
    
    st.divider()

    if not YOUTUBE_API_KEY:
        st.error("⚠️ **Missing API Key:** Add your YouTube API key in `.streamlit/secrets.toml`")
        st.stop()

    with st.sidebar:
        with st.expander("💡 **How the Analysis Works**", expanded=False):
             st.info(
                "1. **Fetch**: We pull YouTube comments from SHA-related videos.\n"
                "2. **Frame**: AI detects the 'media frame' from the video title (e.g., Economic, Governance).\n"
                "3. **Sentiment**: AI analyzes comments using the fine-tuned **AfriBERTa** model to gauge public feeling (Positive, Neutral, Negative)."
            )
        st.divider()
        st.caption("System: Local AfriBERTa • Offline Mode")

    mode = st.radio(
        "**Choose Analysis Mode:**", 
        ["🔗 Analyze One Link", "🔎 Explore by Date Range"], 
        horizontal=True,
        help="Select a mode to analyze public sentiment."
    )

    prev_mode = st.session_state.get("prev_mode")
    if prev_mode != mode:
        _reset_run()
        st.session_state["prev_mode"] = mode

    # ---------- Mode A: Link ----------
    if mode == "🔗 Analyze One Link":
        st.markdown("#### **Analyze a Single Video**")
        url = st.text_input("🔗 **Paste YouTube link (SHA-related):**", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        
        c_btn, c_info = st.columns([1, 2])
        with c_btn:
            if st.button("🚀 Analyze Link", type="primary", use_container_width=True):
                vid = parse_video_id_from_url(url)
                if not vid:
                    st.error("❌ Could not read a valid video ID. Please check the URL.")
                    st.stop()

                with st.spinner("Fetching video metadata..."):
                    meta = fetch_video_metadata(vid)
                
                if not meta:
                    st.error("❌ Video not found or unavailable.")
                    st.stop()

                title = meta["snippet"]["title"]
                desc = meta["snippet"].get("description", "")
                channel_title = meta["snippet"].get("channelTitle", "")

                if not is_sha_related(f"{title}\n{desc}"):
                    st.warning("🚫 The video title or description does not appear to be SHA-related. Analysis stopped.")
                    st.stop()

                with st.spinner(f"Fetching last **{LINK_COMMENTS_LIMIT}** comments..."):
                    comments = fetch_comments(vid, LINK_COMMENTS_LIMIT)
                
                df = pd.DataFrame(comments)
                if df.empty:
                    st.info("No comments found.")
                    st.stop()

                try:
                    pipe, id2label_map, model_path = load_pipeline()
                except Exception as e:
                    st.error(f"❌ Could not load local AfriBERTa model. Check your path or files.\n\nDetails: {e}")
                    st.stop()

                with st.spinner("Running AfriBERTa sentiment analysis..."):
                    sent = run_sentiment(pipe, id2label_map, df["text"].astype(str).tolist())
                
                df = pd.concat([df, pd.DataFrame(sent)], axis=1)
                df = ensure_review_cols(df)

                st.session_state.update({
                    "in_run": True,
                    "ready": True,
                    "mode": "single",
                    "video_meta": {"video_id": vid, "title": title, "channelTitle": channel_title},
                    "table_df": df,
                    "model_path": model_path,
                })
                st.rerun()
        with c_info:
            st.info(f"We cap the analysis at the top **{LINK_COMMENTS_LIMIT}** comments for performance and API quota management.")


    # ---------- Mode B: Date range ----------
    if mode == "🔎 Explore by Date Range":
        st.markdown("#### **Search & Analyze Multiple Videos**")
        
        col1, col2 = st.columns(2)
        with col1:
            today = date.today()
            start_default = today - timedelta(days=30)
            dr = st.date_input(
                "📅 **Pick Date Range (UTC)**",
                value=(start_default, today),
                help="Find SHA-related videos published in this window.",
                key="date_range_picker"
            )
        with col2:
            outlet_selected = st.multiselect("📺 **Filter Channels**", OUTLETS, default=OUTLETS)

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            max_videos = st.number_input("🎬 **Max Videos to Analyze**", min_value=1, max_value=40, value=5, step=1)
        with col4:
            per_video_comments = st.number_input("💬 **Max Comments per Video**", min_value=20, max_value=1200, value=200, step=20)

        st.markdown("---")
        if st.button("🔎 Search & Analyze", type="primary", use_container_width=True):
            start_d, end_d = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (start_default, today))
            start_dt = datetime.combine(start_d, datetime.min.time(), tzinfo=timezone.utc)
            end_dt = datetime.combine(end_d, datetime.max.time(), tzinfo=timezone.utc)

            with st.status("Processing YouTube Data...", expanded=True) as status:
                status.write("Searching for videos...")
                vids = search_sha_videos(start_dt, end_dt, max_videos=int(max_videos))
                if outlet_selected:
                    vids = [v for v in vids if v["channelTitle"] in outlet_selected]

                if not vids:
                    status.update(label="No videos found matching criteria!", state="error", icon="🚫")
                    st.info("No SHA-related videos found for this window/filter.")
                    st.stop()

                status.write(f"Found **{len(vids)}** videos. Fetching **{per_video_comments}** comments per video...")
                all_rows, failures = [], []
                for v in vids:
                    try:
                        rows = fetch_comments(v["video_id"], int(per_video_comments))
                        for r in rows:
                            r["video_title"] = v["title"]
                            r["channelTitle"] = v["channelTitle"]
                            r["video_publishedAt"] = v["publishedAt"]
                        all_rows.extend(rows)
                        time.sleep(0.25) # Respectful delay
                    except Exception as e:
                        failures.append((v["video_id"], str(e)))

                if not all_rows:
                    status.update(label="No comments found in total.", state="error", icon="🚫")
                    st.stop()

                df = pd.DataFrame(all_rows)
                df["time_parsed"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = df[df["time_parsed"].notna()].copy()

                status.write(f"Analyzing **{len(df)}** comments for sentiment...")
                try:
                    pipe, id2label_map, model_path = load_pipeline()
                    sent = run_sentiment(pipe, id2label_map, df["text"].astype(str).tolist())
                    df = pd.concat([df, pd.DataFrame(sent)], axis=1)
                    df = ensure_review_cols(df)
                except Exception as e:
                    status.update(label="Model Error", state="error", icon="❌")
                    st.error(f"Model Error: {e}")
                    st.stop()
                
                status.update(label="Analysis Complete! 🎉", state="complete", icon="✅")

            st.session_state.update({
                "in_run": True,
                "ready": True,
                "mode": "range",
                "videos": vids,
                "table_df": df,
                "date_range": (start_d.isoformat(), end_d.isoformat()),
                "limit_n": int(per_video_comments),
                "model_path": model_path,
            })
            st.rerun()

# =============================
# 4) RESULTS DASHBOARD (MODIFIED KPIs and LAYOUT)
# =============================
if st.session_state.get("ready", False):
    df = st.session_state["table_df"]
    mode_used = st.session_state.get("mode", "single")

    # --- Header Info ---
    if mode_used == "single":
        meta = st.session_state["video_meta"]
        title = meta["title"]
        focus_key, matched = detect_title_focus_and_keywords(title)
        
        st.markdown(f"## {title}")
        st.markdown(f"**Channel:** *{meta.get('channelTitle', 'N/A')}* | **Video ID:** *{meta.get('video_id', 'N/A')}*")
        
        # Frame Analysis Banner - Enhanced Visual
        frame_name = frame_label_for_display(focus_key)
        st.markdown("---")
        st.info(f"**🧠 Media Frame Detected: {frame_name}**\n\nThis video's title suggests a focus that **{frame_meaning_text(focus_key)}**", icon="💡")
        st.markdown(f"**Keywords Matched:** *{', '.join(matched) if matched else 'None'}*")

    else:
        st.markdown("## 📈 Multi-Video Sentiment Trend")
        frame_name = "General Trend" # Default for multi-video
        start_d, end_d = st.session_state.get("date_range", ('N/A', 'N/A'))
        st.markdown(f"**Coverage:** Analyzing **{len(st.session_state.get('videos', []))}** videos ({len(df)} comments total) from **{start_d}** to **{end_d}**.")

    # --- KPIs ---
    
    # Calculate KPIs 
    counts = df["label"].value_counts(dropna=False)
    total = int(counts.sum()) if len(counts) else 0
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    pos = int(counts.get("positive", 0))
    
    # Helper for Delta Text
    def delta_text(n):
        current_pct = (n / total) if total else 0
        return f'<span style="color: #555555; font-size: 0.9em;">({current_pct:.1%})</span>'

    # --- TABS ---
    tab_dash, tab_data = st.tabs(["📊 Main Dashboard", "📝 Data & Review Table"])

    # === TAB 1: DASHBOARD ===
    with tab_dash:
        
        # 1. KPI CARDS - Use columns for visual balance and distinct boxes
        st.markdown("### 🎯 Key Performance Indicators")
        
        # Use columns to contain the layout and improve balance
        col_total, col_neg, col_neu, col_pos = st.columns([1, 1, 1, 1]) # Distribute space equally

        # Total Comments KPI
        with col_total:
            st.markdown(f"""
                <div class="kpi-box kpi-total">
                    <p class="kpi-label">Total Comments Analyzed</p>
                    <p class="kpi-value">{total}</p>
                </div>
            """, unsafe_allow_html=True)

        # Negative KPI
        with col_neg:
            st.markdown(f"""
                <div class="kpi-box kpi-negative">
                    <p class="kpi-label">Negative Sentiment</p>
                    <p class="kpi-value">{neg}</p>
                    <p class="kpi-percent">{delta_text(neg)} of total</p>
                </div>
            """, unsafe_allow_html=True)

        # Neutral KPI
        with col_neu:
            st.markdown(f"""
                <div class="kpi-box kpi-neutral">
                    <p class="kpi-label">Neutral Sentiment</p>
                    <p class="kpi-value">{neu}</p>
                    <p class="kpi-percent">{delta_text(neu)} of total</p>
                </div>
            """, unsafe_allow_html=True)

        # Positive KPI
        with col_pos:
            st.markdown(f"""
                <div class="kpi-box kpi-positive">
                    <p class="kpi-label">Positive Sentiment</p>
                    <p class="kpi-value">{pos}</p>
                    <p class="kpi-percent">{delta_text(pos)} of total</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # 2. VISUAL DISTRIBUTION
        st.markdown("### 📈 Sentiment Distribution")
        
        sentiment_df = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Count": [neg, neu, pos],
            # Use KPI_COLORS for bar background (Plotly's marker color)
            "Color": [KPI_COLORS["negative"], KPI_COLORS["neutral"], KPI_COLORS["positive"]] 
        })
        
        fig = px.bar(
            sentiment_df, 
            x="Sentiment", 
            y="Count", 
            color="Sentiment", 
            text_auto=True,
            # Use KPI colors for the bars
            color_discrete_map={"Negative": KPI_COLORS["negative"], "Neutral": KPI_COLORS["neutral"], "Positive": KPI_COLORS["positive"]}
        )
        
        # Customization for bar thickness (Plotly uses width/gap)
        fig.update_traces(
            marker_line_color='rgb(8,48,107)', # Add a slight border for visibility if needed
            marker_line_width=1.0,
            width=[0.5, 0.5, 0.5] # Make bars thinner (default is 1.0/0.9)
        )
        
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of Comments")
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. FRAME TREND (Only visible in Range mode)
        if mode_used == "range":
            st.markdown("---")
            st.markdown("### 🧠 Media Frame Coverage Trend (Video Titles)")
            
            video_df = pd.DataFrame(st.session_state.get("videos", []))
            if not video_df.empty:
                video_df["focus_key"] = video_df["title"].apply(lambda t: detect_title_focus_and_keywords(t)[0])
                video_df["frame_display"] = video_df["focus_key"].apply(frame_label_for_display)
                
                frame_counts = video_df["frame_display"].value_counts().reset_index()
                frame_counts.columns = ["Frame", "Video Count"]
                
                frame_fig = px.pie(
                    frame_counts, 
                    values='Video Count', 
                    names='Frame', 
                    title='Distribution of Video Media Frames'
                )
                st.plotly_chart(frame_fig, use_container_width=True)

        st.markdown("---")

        # 4. FEEDBACK SECTION
        st.markdown("### 🗣️ Rate and Improve the Model")
        
        if st.session_state.get("feedback_given", False):
            st.success("🙏 Thank you for your feedback! Your input helps improve our fine-tuned AfriBERTa model.")
        else:
            with st.form("feedback_form", clear_on_submit=True):
                rating = st.radio(
                    "**Was the AI analysis accurate/helpful?**", 
                    ["👍 Accurate/Helpful", "👎 Inaccurate/Needs Improvement"], 
                    horizontal=True,
                    index=None
                )
                comment = st.text_area("Add specific comments on misclassification or errors (Optional)")
                
                submitted = st.form_submit_button("Submit Feedback", type="primary")
                
                if submitted and rating:
                    # Prepare feedback data
                    feedback_data = {
                        "timestamp": [pd.Timestamp.utcnow().isoformat()],
                        "rating": [rating],
                        "comment": [comment],
                        "analysis_type": [mode_used],
                        "media_frame": [frame_name],
                        "total_comments": [total],
                        "kpi_neg": [neg],
                        "kpi_neu": [neu],
                        "kpi_pos": [pos],
                    }
                    
                    # Add context-specific info
                    if mode_used == "single":
                        feedback_data["video_id"] = [st.session_state["video_meta"].get("video_id")]
                        feedback_data["video_title"] = [st.session_state["video_meta"].get("title")]
                    else:
                        feedback_data["date_range"] = [str(st.session_state.get("date_range"))]
                    
                    df_feedback = pd.DataFrame.from_dict(feedback_data, orient="columns")
                    
                    # Save to CSV
                    out_path = "analysis_feedback.csv"
                    df_feedback.to_csv(
                        out_path, 
                        mode="a", 
                        index=False, 
                        header=not os.path.exists(out_path), 
                        encoding="utf-8"
                    )
                    
                    st.session_state["feedback_given"] = True
                    st.rerun()
                elif submitted and not rating:
                    st.warning("Please select a rating before submitting.")

        st.markdown("---")

        # 5. SHARE REPORT - Social Sharing Buttons
        st.markdown("### 📤 Share This Report")
        
        # Generate the hidden text payload
        share_text = (
            f"🇰🇪 SHA Sentiment Alert: Just analyzed {total} comments on "
            f"SHA/SHIF policy in Kenya.\n\n"
            f"📊 Overall Sentiment: {(pos/total):.1%} Positive vs {(neg/total):.1%} Negative.\n"
            f"🔍 Media Focus: {frame_name}.\n\n"
            f"#SHIF #SocialHealthAuthority #Kenya"
        )
        encoded_text = urllib.parse.quote(share_text)
        
        twitter_url = f"https.twitter.com/intent/tweet?text={encoded_text}"
        whatsapp_url = f"https.wa.me/?text={encoded_text}"
        email_url = f"mailto:?subject=SHA%20Sentiment%20Analysis&body={encoded_text}"
        
        b1, b2, b3 = st.columns(3)
        b1.link_button("🐦 Post on X (Twitter)", twitter_url, use_container_width=True, help="Share the key findings on X (Twitter).")
        b2.link_button("💬 Share on WhatsApp", whatsapp_url, use_container_width=True, help="Send the summary via WhatsApp.")
        b3.link_button("✉️ Send via Email", email_url, use_container_width=True, help="Send the summary via email.")


    # === TAB 2: DATA & REVIEW ===
    with tab_data:
        st.subheader("📝 Comment-Level Data Review & Export")

        # --- Explore one comment (Cleaned up) ---
        with st.expander("🔍 **Inspect Specific Comment Details**", expanded=False):
            if len(df) > 0:
                options = [f"{i}: {str(t)[:80].replace('\n',' ')}..." for i, t in enumerate(df["text"].astype(str))]
                pick = st.selectbox("Choose a comment to inspect:", options, index=0)
                row_idx = int(pick.split(":")[0])
                
                # Show clean preview
                hide = {"human_label", "reviewer", "action"}
                show_cols = [c for c in df.columns if c not in hide]
                st.dataframe(
                    df.loc[[row_idx], show_cols].T.rename(columns={row_idx: "Details"}), 
                    use_container_width=True,
                    height=350 # Fixed height for better look
                )
            else:
                st.info("No rows available.")

        st.markdown("---")
        
        # --- Main Table (Data Editor) ---
        st.markdown("#### **Full Dataset: Human Labeling Interface**")
        st.caption("Use this table to correct AI labels and log human review actions for model fine-tuning.")
        
        column_cfg = {
            "author": st.column_config.TextColumn("Author", disabled=True),
            "time": st.column_config.TextColumn("Time", disabled=True),
            "likes": st.column_config.NumberColumn("👍 Likes", disabled=True),
            "text": st.column_config.TextColumn("Comment Text", disabled=True, width="large"),
            "label": st.column_config.TextColumn("🤖 AI Label", disabled=True),
            "score": st.column_config.NumberColumn("AI Prob.", disabled=True, format="%.2f", help="AI Confidence Score (0.00 to 1.00)"),
            "confidence": st.column_config.TextColumn("Conf. Level", disabled=True),
            "time_parsed": st.column_config.DatetimeColumn("Published (UTC)", disabled=True),
            "human_label": st.column_config.SelectboxColumn(
                "🧑 Human Label",
                options=["", "negative", "neutral", "positive"],
                help="Manually set the correct sentiment label here for re-training data."
            ),
            "reviewer": st.column_config.TextColumn("Reviewer ID", help="Enter your initials or ID."),
            "action": st.column_config.SelectboxColumn("Action", options=["", "Approved", "Not approved", "Discarded"], help="Log the outcome of the review."),
        }
        
        if mode_used == "range":
            column_cfg["video_title"] = st.column_config.TextColumn("Video Title", disabled=True, width="medium")
            column_cfg["channelTitle"] = st.column_config.TextColumn("Channel", disabled=True)

        editable = st.data_editor(
            df,
            key="comments_editor",
            use_container_width=True,
            num_rows="fixed",
            column_config=column_cfg,
            height=400
        )
        st.session_state["table_df"] = editable

        # --- Action Bar ---
        st.divider()
        c_save, c_export, c_spacer = st.columns([1, 1, 2])
        
        with c_save:
            if st.button("💾 Save Human Labels", use_container_width=True):
                try:
                    changed = editable[
                        (editable["human_label"].astype(str) != "") |
                        (editable["action"].astype(str) != "")
                    ].copy()
                except Exception:
                    changed = pd.DataFrame()
                
                if changed.empty:
                    st.toast("Nothing marked for saving yet.", icon="ℹ️")
                else:
                    changed["saved_at_utc"] = pd.Timestamp.utcnow().isoformat()
                    out_path = "labels_log.csv"
                    # Add 'video_id' if not present (only present in single-mode)
                    if 'video_id' not in changed.columns and 'video_id' in df.columns:
                        changed = changed.merge(df[['video_id']].drop_duplicates(), how='left', left_index=True, right_index=True)

                    changed.to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path), encoding="utf-8")
                    st.toast(f"Saved **{len(changed)}** labeled/reviewed rows!", icon="✅")
        
        with c_export:
            csv = editable.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Dataset CSV",
                data=csv,
                file_name='sha_sentiment_results_full.csv',
                mime='text/csv',
                use_container_width=True
            )

    # Footer Controls
    st.markdown("---")
    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("🔄 **Start New Analysis**", type="secondary", use_container_width=True):
            _reset_run()
            st.rerun()