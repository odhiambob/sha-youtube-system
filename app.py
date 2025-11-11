# app.py
# SHA YouTube Sentiment — AfriBERTa (Fine-Tuned, Local)
# - Main table: text + label (sentiment)
# - Explore one comment: hides human_label/reviewer/action in preview ONLY
# - Full table: includes editable human_label, reviewer, action
# - Two modes: Analyze One Link / Explore by Date Range
# - Loads fine-tuned AfriBERTa locally (no internet required for model)

import os
import re
import time
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple

import requests
import pandas as pd
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

# =============================
# 0) CONFIG
# =============================
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""))

USE_FINETUNED = True
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
# Session helpers (results-only flow)
# =============================
def _reset_run():
    for k in [
        "ready", "table_df", "videos", "video_meta", "date_range",
        "limit_n", "mode", "model_path", "in_run", "prev_mode",
        "show_adv_table"
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
        "citizen welfare": "Citizen Welfare",
        "economic": "Economic",
        "governance": "Governance",
        "implementation": "Implementation",
        "other": "Other / Unclear",
    }
    return mapping.get(key, "Other / Unclear")

def frame_meaning_text(key: str) -> str:
    if key == "citizen welfare":
        return "focus on people’s care experience — access, quality, equity, and benefits."
    if key == "economic":
        return "focus on affordability and money — costs, charges, levies, and funding."
    if key == "governance":
        return "focus on integrity and oversight — investigations, audits, legal processes, and public trust."
    if key == "implementation":
        return "focus on day-to-day rollout — registration, systems, deadlines, capacity, and readiness."
    return "no clear emphasis on cost, trust, rollout, or everyday patient experience."

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

# ====== NEW: ensure review columns exist for full table ======
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
# 3) SETUP UI
# =============================
if not st.session_state.get("in_run", False) and not st.session_state.get("ready", False):
    st.title("SHA YouTube Sentiment — AfriBERTa (Fine-Tuned, Local)")
    st.caption("Paste a link or pick dates. Results show sentiment plus a clear explanation of the title’s focus.")

    if not YOUTUBE_API_KEY:
        st.warning("⚠️ Add your YouTube API key in .streamlit/secrets.toml")
        st.stop()

    with st.expander("Quick tip", expanded=True):
        st.write("**Option 1:** Paste a SHA-related YouTube link and click **Analyze**.  \n"
                 "**Option 2:** Choose a date range (and channels) to explore multiple videos at once.")

    mode = st.radio("Choose how you want to analyze", ["Analyze One Link", "Explore by Date Range"], horizontal=True)

    prev_mode = st.session_state.get("prev_mode")
    if prev_mode != mode:
        for k in ["ready", "table_df", "videos", "video_meta", "date_range", "limit_n", "mode"]:
            st.session_state.pop(k, None)
        st.session_state["prev_mode"] = mode

    # ---------- Mode A: Link ----------
    if mode == "Analyze One Link":
        url = st.text_input("Paste YouTube link (SHA-related):")
        if st.button("Analyze"):
            vid = parse_video_id_from_url(url)
            if not vid:
                st.error("❌ Could not read a valid video ID.")
                st.stop()

            meta = fetch_video_metadata(vid)
            if not meta:
                st.error("❌ Video not found or unavailable.")
                st.stop()

            title = meta["snippet"]["title"]
            desc = meta["snippet"].get("description", "")
            channel_title = meta["snippet"].get("channelTitle", "")

            if not is_sha_related(f"{title}\n{desc}"):
                st.warning("🚫 Not SHA content. This tool is limited to SHA-related videos.")
                st.stop()

            comments = fetch_comments(vid, LINK_COMMENTS_LIMIT)
            df = pd.DataFrame(comments)
            if df.empty:
                st.info("No comments found.")
                st.stop()

            try:
                pipe, id2label_map, model_path = load_pipeline()
            except Exception as e:
                st.error(f"❌ Could not load local AfriBERTa model.\n\n{e}")
                st.stop()

            sent = run_sentiment(pipe, id2label_map, df["text"].astype(str).tolist())
            df = pd.concat([df, pd.DataFrame(sent)], axis=1)

            # ensure review fields exist for full table
            df = ensure_review_cols(df)

            st.session_state.update({
                "in_run": True,
                "ready": True,
                "mode": "single",
                "video_meta": {"video_id": vid, "title": title, "channelTitle": channel_title},
                "table_df": df,           # includes human_label/reviewer/action
                "model_path": model_path,
            })
            st.rerun()

    # ---------- Mode B: Date range ----------
    if mode == "Explore by Date Range":
        today = date.today()
        start_default = today - timedelta(days=30)
        dr = st.date_input(
            "Pick date range (UTC publish date)",
            value=(start_default, today),
            help="Find SHA-related videos published in this window, then fetch & analyze comments.",
        )
        with st.expander("Filter by channel (optional)"):
            outlet_selected = st.multiselect("Channels", OUTLETS, default=OUTLETS)

        max_videos = st.number_input("Max videos", min_value=1, max_value=40, value=5, step=1)
        per_video_comments = st.number_input("Max comments per video", min_value=20, max_value=1200, value=200, step=20)

        if st.button("Search & Analyze"):
            start_d, end_d = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (start_default, today))
            start_dt = datetime.combine(start_d, datetime.min.time(), tzinfo=timezone.utc)
            end_dt = datetime.combine(end_d, datetime.max.time(), tzinfo=timezone.utc)

            vids = search_sha_videos(start_dt, end_dt, max_videos=int(max_videos))
            if outlet_selected:
                vids = [v for v in vids if v["channelTitle"] in outlet_selected]

            if not vids:
                st.info("No SHA-related videos found for this window/filter.")
                st.stop()

            all_rows, failures = [], []
            for v in vids:
                try:
                    rows = fetch_comments(v["video_id"], int(per_video_comments))
                    for r in rows:
                        r["video_title"] = v["title"]
                        r["channelTitle"] = v["channelTitle"]
                        r["video_publishedAt"] = v["publishedAt"]
                        all_rows.append(r)
                    time.sleep(0.25)  # gentle pacing
                except Exception as e:
                    failures.append((v["video_id"], str(e)))

            if failures:
                st.warning(f"Skipped {len(failures)} video(s) due to API timeouts/errors).")
                with st.expander("Show skipped videos"):
                    for vid, err in failures:
                        st.write(f"- {vid}: {err}")

            if not all_rows:
                st.info("No comments retrieved for the selected videos.")
                st.stop()

            df = pd.DataFrame(all_rows)
            df["time_parsed"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df[df["time_parsed"].notna()].copy()

            try:
                pipe, id2label_map, model_path = load_pipeline()
            except Exception as e:
                st.error(f"❌ Could not load local AfriBERTa model.\n\n{e}")
                st.stop()

            sent = run_sentiment(pipe, id2label_map, df["text"].astype(str).tolist())
            df = pd.concat([df, pd.DataFrame(sent)], axis=1)

            # ensure review fields exist for full table
            df = ensure_review_cols(df)

            st.session_state.update({
                "in_run": True,
                "ready": True,
                "mode": "range",
                "videos": vids,
                "table_df": df,    # includes human_label/reviewer/action
                "date_range": (start_d.isoformat(), end_d.isoformat()),
                "limit_n": int(per_video_comments),
                "model_path": model_path,
            })
            st.rerun()

# =============================
# 4) RESULTS (text + sentiment only)
# =============================
if st.session_state.get("ready", False):
    df = st.session_state["table_df"]
    mode_used = st.session_state.get("mode", "single")

    if mode_used == "single":
        meta = st.session_state["video_meta"]
        title = meta["title"]
        focus_key, matched = detect_title_focus_and_keywords(title)

        st.write(f"**Title:** {title}")
        st.write(f"**Channel:** {meta.get('channelTitle', '')}")

        frame_name = frame_label_for_display(focus_key)
        st.markdown(f"**Frame detected:** {frame_name}")
        st.markdown(
            f"This media house is looking at this issue through a **{frame_name}** frame, "
            f"which means {frame_meaning_text(focus_key)}"
        )
        if matched:
            st.caption(f"Noticed keywords: {', '.join(matched)}")
        else:
            st.caption("No strong frame keywords detected.")
        st.caption(f"Internal comment cap: {LINK_COMMENTS_LIMIT}")
    else:
        st.subheader("Videos analyzed")
        vids = st.session_state.get("videos", [])
        for v in vids:
            st.write(f"- **{v['title']}** — {v['channelTitle']} — published {v['publishedAt']} (UTC)")
        dr = st.session_state.get("date_range", None)
        ln = st.session_state.get("limit_n", None)
        st.caption(f"Window (UTC): {dr[0]} → {dr[1]} • Max comments/video: {ln}")

    # KPIs
    counts = df["label"].value_counts(dropna=False)
    total = int(counts.sum()) if len(counts) else 0
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    pos = int(counts.get("positive", 0))

    def pct(n): return f"{(n/total):.0%}" if total > 0 else "0%"

    c1, c2, c3 = st.columns(3)
    c1.metric("Negative", f"{neg}", pct(neg))
    c2.metric("Neutral",  f"{neu}", pct(neu))
    c3.metric("Positive", f"{pos}", pct(pos))

    # ===== Main table: ONLY text + label (read-only) =====
    st.subheader("Comments")
    view_cols = ["text", "label"]
    missing = [c for c in view_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing columns in data: {missing}")
        st.stop()

    display_df = df[view_cols].copy()
    st.dataframe(display_df, use_container_width=True)

    # Export just the two-column view
    if st.button("Export comments (text + label) to CSV"):
        export_path = "analysis_export_text_label.csv"
        display_df.to_csv(export_path, index=False, encoding="utf-8")
        st.success(f"Exported to {export_path}")

    # --- Explore one comment (details on demand; HIDE review fields only here) ---
    with st.expander("Explore one comment (details on demand)"):
        if len(df) > 0:
            options = [f"{i}: {str(t)[:80].replace('\n',' ')}" for i, t in enumerate(df["text"].astype(str))]
            pick = st.selectbox("Pick a row to inspect", options, index=0)
            row_idx = int(pick.split(":")[0])

            # Hide only the review columns in this preview; keep df intact for full table
            hide = {"human_label", "reviewer", "action"}
            show_cols = [c for c in df.columns if c not in hide]
            row_view = df.loc[row_idx, show_cols]

            st.write("**Full record (review fields hidden)**")
            st.dataframe(row_view.to_frame(name="Value"))
        else:
            st.info("No rows available to explore.")

    # ===== Full table with review fields (unchanged) =====
    show_full = st.toggle("Show full table (review & all columns)", value=st.session_state.get("show_adv_table", False))
    st.session_state["show_adv_table"] = show_full

    if st.session_state.get("show_adv_table", False):
        st.subheader("Comments (full view: model result + your review)")

        column_cfg = {
            "author": st.column_config.TextColumn("author", disabled=True),
            "time": st.column_config.TextColumn("time", disabled=True),
            "likes": st.column_config.NumberColumn("likes", disabled=True),
            "text": st.column_config.TextColumn("text", disabled=True),
            "label": st.column_config.TextColumn("model_label", disabled=True),
            "score": st.column_config.NumberColumn("probability", disabled=True, format="%.4f"),
            "confidence": st.column_config.TextColumn("confidence", disabled=True),
            "time_parsed": st.column_config.DatetimeColumn("publishedAt (UTC)", disabled=True),
            "human_label": st.column_config.SelectboxColumn(
                "Your label",
                options=["", "negative", "neutral", "positive"],
                help="If the model is wrong, pick the correct label."
            ),
            "reviewer": st.column_config.TextColumn("Reviewer"),
            "action": st.column_config.SelectboxColumn("Action", options=["", "Approved", "Not approved"]),
        }
        if mode_used == "range":
            column_cfg["video_id"] = st.column_config.TextColumn("video_id", disabled=True)
            column_cfg["video_title"] = st.column_config.TextColumn("video_title", disabled=True)
            column_cfg["channelTitle"] = st.column_config.TextColumn("channel", disabled=True)
            column_cfg["video_publishedAt"] = st.column_config.TextColumn("video_publishedAt", disabled=True)

        editable = st.data_editor(
            df,
            key="comments_editor",
            use_container_width=True,
            num_rows="fixed",
            column_config=column_cfg,
        )
        st.session_state["table_df"] = editable
        st.caption(
            "Mark the model’s output as **Approved** or **Not approved**. "
            "If incorrect, also select your label above."
        )

        csave, cexport = st.columns([1, 1])
        with csave:
            if st.button("Save collaboration log"):
                try:
                    # if columns exist, filter rows with any input in human_label or action
                    changed = editable[
                        (editable["human_label"].astype(str) != "") |
                        (editable["action"].astype(str) != "")
                    ].copy()
                except Exception:
                    changed = pd.DataFrame()
                if changed.empty:
                    st.info("Nothing to save yet.")
                else:
                    changed["saved_at_utc"] = pd.Timestamp.utcnow().isoformat()
                    out_path = "labels_log.csv"
                    changed.to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path), encoding="utf-8")
                    st.success(f"Saved {len(changed)} row(s) to {out_path}")
        with cexport:
            if st.button("Export full results (CSV)"):
                export_path = "analysis_export.csv"
                editable.to_csv(export_path, index=False, encoding="utf-8")
                st.success(f"Exported to {export_path}")

    # Footer controls
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("New analysis"):
            _reset_run()
            st.rerun()
    with col_b:
        with st.expander("Diagnostics & Notes"):
            st.markdown("""
- Date filtering only in **Explore by Date Range**; link mode uses an internal cap for speed and quota safety.
- YouTube calls use retries + backoff and longer timeouts for slow networks.
- The model loads from local disk (`local_files_only=True`) using the resolved AfriBERTa path.
            """)
            st.write("**USE_FINETUNED:**", USE_FINETUNED)
            st.write("**Resolved FINETUNED_PATH:**", resolve_ft_path())
            st.write("**Current mode:**", st.session_state.get("mode"))
