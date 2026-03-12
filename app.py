"""
VideoSeek AI — Streamlit Application
Intelligent video context search engine powered by Whisper, CLIP, FAISS & Twelve Labs.
"""

import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np

import config
from utils import format_timestamp, youtube_embed_url, cleanup_temp_files

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoSeek AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
    }

    /* ── Header ────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #7C3AED 0%, #EC4899 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* ── Result Cards ──────────────────────────────────── */
    .result-card {
        background: linear-gradient(145deg, rgba(30,30,60,0.8), rgba(26,26,46,0.95));
        border: 1px solid rgba(124,58,237,0.25);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .result-card:hover {
        border-color: rgba(124,58,237,0.6);
        box-shadow: 0 8px 32px rgba(124,58,237,0.15);
        transform: translateY(-2px);
    }
    .result-timestamp {
        display: inline-block;
        background: linear-gradient(135deg, #7C3AED, #6D28D9);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .result-score {
        display: inline-block;
        background: rgba(236,72,153,0.15);
        color: #F472B6;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-left: 8px;
    }
    .result-text {
        color: #CBD5E1;
        margin-top: 0.75rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* ── Sidebar ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%);
        border-right: 1px solid rgba(124,58,237,0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #C4B5FD;
    }

    /* ── Status badges ─────────────────────────────────── */
    .status-ready {
        background: rgba(34,197,94,0.15);
        color: #4ADE80;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-processing {
        background: rgba(245,158,11,0.15);
        color: #FBBF24;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* ── Tab styling ───────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* ── Metrics ───────────────────────────────────────── */
    .metric-card {
        background: rgba(30,30,60,0.6);
        border: 1px solid rgba(124,58,237,0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #C4B5FD;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Hide Streamlit branding ───────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "video_path": None,
        "video_source": None,       # "upload" or "youtube"
        "youtube_url": None,
        "processed": False,
        "processing": False,
        "transcript_segments": [],
        "frame_store": None,         # VectorStore for CLIP
        "transcript_store": None,    # VectorStore for transcripts
        "frames_data": [],           # (timestamp, PIL.Image) list
        "twelve_labs_index_id": None,
        "twelve_labs_video_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 VideoSeek AI</h1>
        <p>Find any moment in any video — powered by Whisper, CLIP & FAISS</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Video Input
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📹 Video Input")

    input_method = st.radio(
        "Choose input method",
        ["Upload File", "YouTube URL"],
        horizontal=True,
        label_visibility="collapsed",
    )

    video_ready = False

    if input_method == "Upload File":
        uploaded = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Max 500 MB",
        )
        if uploaded:
            # Save to temp
            tmp_path = os.path.join(config.TEMP_DIR, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.video_path = tmp_path
            st.session_state.video_source = "upload"
            video_ready = True
            st.success(f"✅ Loaded: `{uploaded.name}`")

    else:
        yt_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        if yt_url:
            st.session_state.youtube_url = yt_url
            st.session_state.video_source = "youtube"
            video_ready = True
            st.success("✅ YouTube URL ready")

    st.markdown("---")

    # ── Process Button ────────────────────────────────────────────────────
    process_btn = st.button(
        "🚀 Process Video",
        use_container_width=True,
        disabled=not video_ready or st.session_state.processing,
        type="primary",
    )

    # ── Status ────────────────────────────────────────────────────────────
    if st.session_state.processed:
        st.markdown('<div class="status-ready">● Ready to Search</div>', unsafe_allow_html=True)
    elif st.session_state.processing:
        st.markdown('<div class="status-processing">● Processing…</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ⚙️ Settings")

    frame_fps = st.slider(
        "Frame extraction FPS",
        min_value=0.25,
        max_value=5.0,
        value=config.FRAME_EXTRACT_FPS,
        step=0.25,
        help="Higher = more frames = better visual search but slower processing.",
    )

    top_k = st.slider(
        "Results to show",
        min_value=1,
        max_value=25,
        value=config.TOP_K_RESULTS,
        step=1,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:#64748B; font-size:0.75rem;">
            VideoSeek AI v1.0<br>
            Whisper · CLIP · FAISS · Twelve Labs
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Processing Pipeline
# ─────────────────────────────────────────────────────────────────────────────
if process_btn and video_ready:
    st.session_state.processing = True
    st.session_state.processed = False

    progress = st.progress(0, text="Initialising pipeline…")

    try:
        # Step 1 — Download / prepare video
        progress.progress(5, text="📥 Preparing video…")
        if st.session_state.video_source == "youtube":
            from video_processor import download_youtube
            with st.spinner("Downloading from YouTube…"):
                st.session_state.video_path = download_youtube(
                    st.session_state.youtube_url
                )
            progress.progress(15, text="✅ Video downloaded")
        else:
            progress.progress(15, text="✅ Video file loaded")

        video_path = st.session_state.video_path

        # Step 2 — Extract audio
        progress.progress(20, text="🎵 Extracting audio…")
        from video_processor import extract_audio
        audio_path = extract_audio(video_path)
        progress.progress(25, text="✅ Audio extracted")

        # Step 3 — Transcribe
        progress.progress(30, text="🗣️ Transcribing with Whisper…")
        from transcriber import transcribe, embed_segments
        with st.spinner("Running Whisper transcription (this may take a minute)…"):
            segments = transcribe(audio_path)
        st.session_state.transcript_segments = segments
        progress.progress(50, text=f"✅ Transcribed {len(segments)} segments")

        # Step 4 — Embed transcript segments
        progress.progress(55, text="📝 Embedding transcript segments…")
        transcript_embeddings, transcript_meta = embed_segments(segments)
        progress.progress(60, text="✅ Transcript embeddings ready")

        # Step 5 — Extract frames
        progress.progress(62, text=f"🎞️ Extracting frames at {frame_fps} FPS…")
        from video_processor import extract_frames
        with st.spinner("Extracting video frames…"):
            frames = extract_frames(video_path, fps=frame_fps)
        st.session_state.frames_data = frames
        progress.progress(70, text=f"✅ Extracted {len(frames)} frames")

        # Step 6 — Embed frames with CLIP
        progress.progress(72, text="🖼️ Embedding frames with CLIP…")
        from frame_embedder import embed_frames
        with st.spinner("Running CLIP inference on frames…"):
            frame_embeddings, frame_meta = embed_frames(frames)
        progress.progress(85, text="✅ Frame embeddings ready")

        # Step 7 — Build FAISS indices
        progress.progress(88, text="🗂️ Building FAISS indices…")
        from vector_store import VectorStore

        # Transcript store
        if len(transcript_embeddings) > 0:
            t_store = VectorStore(dimension=transcript_embeddings.shape[1])
            t_store.add(transcript_embeddings, transcript_meta)
            st.session_state.transcript_store = t_store

        # Frame store
        if len(frame_embeddings) > 0:
            f_store = VectorStore(dimension=frame_embeddings.shape[1])
            f_store.add(frame_embeddings, frame_meta)
            st.session_state.frame_store = f_store

        progress.progress(95, text="✅ FAISS indices built")

        # Step 8 — Twelve Labs (if configured)
        try:
            from twelve_labs_client import TwelveLabsClient
            tl_client = TwelveLabsClient()
            if tl_client.is_available:
                progress.progress(96, text="☁️ Uploading to Twelve Labs…")
                index_id = tl_client.get_or_create_index()
                video_id = tl_client.upload_video(index_id, video_path)
                st.session_state.twelve_labs_index_id = index_id
                st.session_state.twelve_labs_video_id = video_id
                progress.progress(99, text="✅ Twelve Labs indexing complete")
        except Exception:
            pass  # Twelve Labs is optional

        progress.progress(100, text="🎉 Processing complete!")
        st.session_state.processed = True
        st.session_state.processing = False

        # Cleanup audio temp file
        try:
            os.remove(audio_path)
        except OSError:
            pass

        st.rerun()

    except Exception as e:
        st.session_state.processing = False
        st.error(f"❌ Processing failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Main Area — Search & Results
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.processed:
    # ── Metrics row ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.transcript_segments)}</div>
                <div class="metric-label">Transcript Segments</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.frames_data)}</div>
                <div class="metric-label">Extracted Frames</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        t_size = st.session_state.transcript_store.size if st.session_state.transcript_store else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{t_size}</div>
                <div class="metric-label">Text Vectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        f_size = st.session_state.frame_store.size if st.session_state.frame_store else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{f_size}</div>
                <div class="metric-label">Visual Vectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_names = ["🖼️ Visual Search (CLIP)", "📝 Transcript Search", "📜 Full Transcript"]

    # Conditionally add Twelve Labs tab
    tl_available = (
        st.session_state.twelve_labs_index_id is not None
        and st.session_state.twelve_labs_video_id is not None
    )
    if tl_available:
        tab_names.append("☁️ Twelve Labs Search")

    tabs = st.tabs(tab_names)

    # ── Tab 1: Visual (CLIP) Search ──────────────────────────────────────
    with tabs[0]:
        st.markdown("### 🖼️ Visual Search")
        st.markdown("*Describe what you see — find the exact frame.*")

        visual_query = st.text_input(
            "Search by visual description",
            placeholder="e.g. a person standing near a whiteboard",
            key="visual_query",
        )
        visual_search_btn = st.button("Search Visually", key="visual_search_btn", type="primary")

        if visual_search_btn and visual_query and st.session_state.frame_store:
            from frame_embedder import embed_text_query

            query_emb = embed_text_query(visual_query)
            results = st.session_state.frame_store.search(query_emb, top_k=top_k)

            if not results:
                st.info("No results found. Try a different query.")
            else:
                st.markdown(f"**Found {len(results)} results:**")
                for i, r in enumerate(results):
                    ts = r.get("timestamp", 0)
                    score = r.get("score", 0)

                    # Find the closest frame image
                    frame_img = None
                    for fts, fimg in st.session_state.frames_data:
                        if abs(fts - ts) < 1.0:
                            frame_img = fimg
                            break

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <span class="result-timestamp">⏱ {format_timestamp(ts)}</span>
                            <span class="result-score">Score: {score:.3f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    col_img, col_player = st.columns([1, 2])
                    with col_img:
                        if frame_img:
                            st.image(frame_img, caption=f"Frame at {format_timestamp(ts)}", use_container_width=True)

                    with col_player:
                        if st.session_state.video_source == "youtube":
                            embed_url = youtube_embed_url(
                                st.session_state.youtube_url,
                                start_seconds=int(ts),
                            )
                            st.markdown(
                                f'<iframe width="100%" height="250" src="{embed_url}" '
                                f'frameborder="0" allowfullscreen></iframe>',
                                unsafe_allow_html=True,
                            )
                        elif st.session_state.video_path:
                            st.video(st.session_state.video_path, start_time=int(ts))

    # ── Tab 2: Transcript Search ─────────────────────────────────────────
    with tabs[1]:
        st.markdown("### 📝 Transcript Search")
        st.markdown("*Search through what was said in the video.*")

        text_query = st.text_input(
            "Search transcripts",
            placeholder="e.g. discusses machine learning challenges",
            key="text_query",
        )
        text_search_btn = st.button("Search Transcripts", key="text_search_btn", type="primary")

        if text_search_btn and text_query and st.session_state.transcript_store:
            from transcriber import embed_query_text

            query_emb = embed_query_text(text_query)
            results = st.session_state.transcript_store.search(query_emb, top_k=top_k)

            if not results:
                st.info("No results found. Try a different query.")
            else:
                st.markdown(f"**Found {len(results)} results:**")
                for i, r in enumerate(results):
                    start = r.get("start", 0)
                    end = r.get("end", 0)
                    score = r.get("score", 0)
                    text = r.get("text", "")

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <span class="result-timestamp">⏱ {format_timestamp(start)} → {format_timestamp(end)}</span>
                            <span class="result-score">Score: {score:.3f}</span>
                            <div class="result-text">"{text}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.session_state.video_source == "youtube":
                        embed_url = youtube_embed_url(
                            st.session_state.youtube_url,
                            start_seconds=int(start),
                        )
                        st.markdown(
                            f'<iframe width="100%" height="200" src="{embed_url}" '
                            f'frameborder="0" allowfullscreen></iframe>',
                            unsafe_allow_html=True,
                        )
                    elif st.session_state.video_path:
                        st.video(st.session_state.video_path, start_time=int(start))

    # ── Tab 3: Full Transcript ───────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 📜 Full Transcript")
        if st.session_state.transcript_segments:
            for seg in st.session_state.transcript_segments:
                ts_label = f"{format_timestamp(seg['start'])} → {format_timestamp(seg['end'])}"
                st.markdown(
                    f"""
                    <div class="result-card" style="padding: 0.75rem 1rem;">
                        <span class="result-timestamp" style="font-size:0.75rem;">{ts_label}</span>
                        <div class="result-text" style="margin-top:0.4rem;">{seg['text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No transcript segments available.")

    # ── Tab 4: Twelve Labs (conditional) ─────────────────────────────────
    if tl_available and len(tabs) > 3:
        with tabs[3]:
            st.markdown("### ☁️ Twelve Labs Multimodal Search")
            st.markdown("*Powered by Twelve Labs Marengo for cross-modal video understanding.*")

            tl_query = st.text_input(
                "Multimodal search query",
                placeholder="e.g. someone explaining a diagram on screen",
                key="tl_query",
            )

            tl_options = st.multiselect(
                "Search modalities",
                ["visual", "conversation", "text_in_video", "logo"],
                default=["visual", "conversation", "text_in_video"],
                key="tl_options",
            )

            tl_search_btn = st.button("Search with Twelve Labs", key="tl_search_btn", type="primary")

            if tl_search_btn and tl_query:
                try:
                    from twelve_labs_client import TwelveLabsClient
                    tl_client = TwelveLabsClient()
                    results = tl_client.search(
                        st.session_state.twelve_labs_index_id,
                        tl_query,
                        search_options=tl_options,
                        top_k=top_k,
                    )

                    if not results:
                        st.info("No results found via Twelve Labs.")
                    else:
                        st.markdown(f"**Found {len(results)} results:**")
                        for r in results:
                            start = r.get("start", 0)
                            end = r.get("end", 0)
                            score = r.get("score", 0)

                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <span class="result-timestamp">⏱ {format_timestamp(start)} → {format_timestamp(end)}</span>
                                    <span class="result-score">Score: {score:.3f}</span>
                                    <span class="result-score" style="background:rgba(124,58,237,0.15); color:#C4B5FD;">☁️ Twelve Labs</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            if st.session_state.video_source == "youtube":
                                embed_url = youtube_embed_url(
                                    st.session_state.youtube_url,
                                    start_seconds=int(start),
                                )
                                st.markdown(
                                    f'<iframe width="100%" height="200" src="{embed_url}" '
                                    f'frameborder="0" allowfullscreen></iframe>',
                                    unsafe_allow_html=True,
                                )
                            elif st.session_state.video_path:
                                st.video(st.session_state.video_path, start_time=int(start))
                except Exception as e:
                    st.error(f"Twelve Labs search error: {e}")

else:
    # ── Welcome / empty state ─────────────────────────────────────────────
    st.markdown("")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(
            """
            <div style="text-align:center; padding: 3rem 0;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">🎬</div>
                <h2 style="color:#C4B5FD; margin-bottom:0.5rem;">Welcome to VideoSeek AI</h2>
                <p style="color:#94A3B8; font-size:1.1rem; max-width:500px; margin:0 auto;">
                    Upload a video or paste a YouTube URL in the sidebar to get started.
                    Our AI will transcribe, analyze frames, and build a searchable index
                    so you can find any moment instantly.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Feature cards
        features = [
            ("🗣️", "Whisper Transcription", "Auto-transcribe spoken words with OpenAI's Whisper model"),
            ("🖼️", "CLIP Visual Search", "Find frames by describing what you see in natural language"),
            ("⚡", "FAISS Vector Search", "Blazing-fast similarity search across thousands of embeddings"),
            ("☁️", "Twelve Labs", "Optional multimodal search via Twelve Labs Marengo API"),
        ]

        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(features):
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-bottom:1rem; text-align:left; padding:1.25rem;">
                        <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
                        <div style="color:#E2E8F0; font-weight:600; margin-bottom:0.25rem;">{title}</div>
                        <div style="color:#94A3B8; font-size:0.85rem;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
