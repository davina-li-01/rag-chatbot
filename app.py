"""Reel Agent: point-and-click app. Launch with the 'Reel Agent' icon on your Desktop."""
import glob
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from reel_agent import llm
from reel_agent.config import AUDIO_EXTS, ROOT, VIDEO_EXTS
from reel_agent.graph import build

PROJECTS = ROOT / "projects"
PROJECTS.mkdir(exist_ok=True)
st.set_page_config(page_title="Reel Agent", page_icon="🎬", layout="wide")


# ---------- helpers ----------
@st.cache_resource
def graph_for(project: str):
    return build(project)


def cfg(project: str):
    return {"configurable": {"thread_id": Path(project).name}}


def open_path(p, reveal=False):
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(p)] if reveal else ["open", str(p)])


def premiere_app():
    hits = sorted(glob.glob("/Applications/Adobe Premiere Pro*/Adobe Premiere Pro*.app"))
    return hits[-1] if hits else None


def files_in(folder: Path, exts):
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in exts) if folder.exists() else []


def save_uploads(uploads, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for u in uploads or []:
        target = dest / u.name
        if not target.exists() or target.stat().st_size != u.size:
            target.write_bytes(u.getbuffer())


def run(project, payload):
    """Run the agents until they finish, pause for review, or fail."""
    graph = graph_for(project)
    logs = st.session_state.setdefault("logs", {}).setdefault(project, [])
    st.session_state.get("errors", {}).pop(project, None)
    with st.status("Agents working…", expanded=True) as status:
        try:
            for update in graph.stream(payload, cfg(project), stream_mode="updates"):
                for node, data in update.items():
                    if node != "__interrupt__" and data:
                        for line in data.get("log", []):
                            logs.append(line)
                            st.write(line)
            status.update(label="Done for now", state="complete", expanded=False)
        except Exception as e:  # progress up to the last finished step is saved
            st.session_state.setdefault("errors", {})[project] = f"{type(e).__name__}: {e}"
            status.update(label="Stopped with an error", state="error")
    st.rerun()


def resume(project, answer):
    run(project, Command(resume=answer))


def thumb(project_dir: Path, source: str, t: float) -> str:
    """One small frame from a clip, cached so the review screen stays fast."""
    d = project_dir / ".agent" / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{Path(source).stem}_{t:.2f}.jpg"
    if not f.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", source, "-frames:v", "1",
                        "-vf", "scale=160:-2", str(f)], stdin=subprocess.DEVNULL)
    return str(f)


def render_caption_preview(project_dir: Path, values: dict, captions: list) -> str:
    """Quick low-res video of the cut + music + these captions, before approving."""
    from reel_agent.config import spec
    from reel_agent.tools.exporters import write_ass
    from reel_agent.tools.media import render_draft
    s = spec()
    out = project_dir / "out"
    out.mkdir(exist_ok=True)
    ass = write_ass(captions, str(out / "caption_preview.ass"), s["frame"], s["captions"])
    return render_draft(values["cut_list"], values["song"], [], ass, str(out / "caption_preview.mp4"),
                        360, 640, s["frame"]["fps"])


def explain(err: str):
    """Turn a raw error into (plain-English message, fix type)."""
    e = err.lower()
    if any(k in e for k in ("libsndfile", "format not recognised", "audio", ".m4a", ".mp3")) or "songs left" in e:
        return "🎵 **We couldn't use your song.** Upload a different song below (.mp3 or .wav work best).", "song"
    if "no video files" in e:
        return "🎥 **No footage found.** Add your clips below.", "footage"
    if "no usable footage" in e:
        return "🎥 **Your clips are too short or unusable.** Add a few longer clips below.", "footage"
    return "😬 **Something went wrong on our side.** Try Resume. If it happens again, copy the details below and send them to Claude.", "other"


# ---------- sidebar ----------
with st.sidebar:
    st.title("🎬 Reel Agent")

    with st.form("new", clear_on_submit=True):
        name = st.text_input("New video", placeholder="e.g. 2026-09-25 matcha day")
        if st.form_submit_button("Create", width="stretch") and name.strip():
            slug = "".join(c if c.isalnum() or c in "-_ " else "" for c in name).strip().replace(" ", "-")
            (PROJECTS / slug / "footage").mkdir(parents=True, exist_ok=True)
            (PROJECTS / slug / "music").mkdir(exist_ok=True)
            st.session_state["project"] = slug

    projects = sorted((p for p in PROJECTS.iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    names = [p.name for p in projects]
    if not names:
        st.info("No videos yet. Create one above, or try the sample.")
        if st.button("Make sample footage", width="stretch"):
            with st.spinner("Generating test clips…"):
                subprocess.run([sys.executable, str(ROOT / "make_sample.py")], cwd=ROOT)
            st.session_state["project"] = "sample"
            st.rerun()
        st.stop()
    current = st.session_state.get("project")
    idx = names.index(current) if current in names else 0
    project_name = st.radio("Your videos", names, index=idx)
    st.session_state["project"] = project_name

    st.divider()
    with st.expander("AI connections"):
        st.caption("Leave blank to keep that agent on placeholder (mock) output.")
        st.write(("✅ " if llm.has_gemini() else "⚪️ ") + "Gemini: watches footage")
        st.write(("✅ " if llm.has_claude() else "⚪️ ") + "Claude: writes captions, orders clips")
        g = st.text_input("Gemini API key", type="password")
        a = st.text_input("Anthropic API key", type="password")
        if st.button("Save keys"):
            envf = ROOT / ".env"
            lines = envf.read_text().splitlines() if envf.exists() else []
            for key, val in (("GEMINI_API_KEY", g), ("ANTHROPIC_API_KEY", a)):
                if val.strip():
                    lines = [l for l in lines if not l.startswith(key + "=")] + [f"{key}={val.strip()}"]
                    os.environ[key] = val.strip()
            envf.write_text("\n".join(lines) + "\n")
            st.rerun()

    if st.button("Quit Reel Agent", width="stretch"):
        st.write("Closed. You can close this tab.")
        os.kill(os.getpid(), signal.SIGTERM)


# ---------- main ----------
project = str((PROJECTS / project_name).resolve())
pdir = Path(project)
graph = graph_for(project)
snap = graph.get_state(cfg(project))
pending = [i.value for t in snap.tasks for i in t.interrupts]
started = bool(snap.values)

st.header(project_name.replace("-", " "))
err = st.session_state.get("errors", {}).get(project)
if err:
    msg, kind = explain(err)
    st.error(msg)
    if kind == "song":
        songs = files_in(pdir / "music", AUDIO_EXTS)
        up = st.file_uploader("Upload a different song", type=[e[1:] for e in AUDIO_EXTS], key=f"fix_song_{project_name}")
        if up and st.button("Use this song and continue", type="primary"):
            for old in songs:
                old.unlink()
            save_uploads([up], pdir / "music")
            run(project, None)
    elif kind == "footage":
        save_uploads(st.file_uploader("Add clips", type=[e[1:] for e in VIDEO_EXTS], accept_multiple_files=True,
                                      key=f"fix_foot_{project_name}"), pdir / "footage")
    with st.expander("Technical details"):
        st.code(err)

# --- 1. Not started: add footage + song, run ---
if not started:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1 · Footage")
        clips = files_in(pdir / "footage", VIDEO_EXTS)
        st.caption(f"{len(clips)} clip(s) added" + (": " + ", ".join(c.name for c in clips[:6]) if clips else ""))
        save_uploads(st.file_uploader("Drop clips here", type=[e[1:] for e in VIDEO_EXTS],
                                      accept_multiple_files=True, key=f"v_{project_name}"), pdir / "footage")
        if st.button("…or open the footage folder in Finder"):
            (pdir / "footage").mkdir(parents=True, exist_ok=True)
            open_path(pdir / "footage")
    with c2:
        st.subheader("2 · Song")
        songs = files_in(pdir / "music", AUDIO_EXTS)
        st.caption(songs[0].name if songs else "None yet (falls back to library/music)")
        save_uploads(st.file_uploader("Drop your song here", type=[e[1:] for e in AUDIO_EXTS],
                                      accept_multiple_files=True, key=f"a_{project_name}"), pdir / "music")

    st.subheader("3 · Edit")
    auto = st.toggle("Fully autonomous (skip my caption + final reviews)", value=False)
    if st.button("▶  Make my reel", type="primary", disabled=not files_in(pdir / "footage", VIDEO_EXTS)):
        run(project, {"project_dir": project, "auto_approve": auto, "revision": 0, "log": []})

# --- 2. Paused at caption review ---
elif pending and pending[0]["gate"] == "captions":
    st.subheader("✍️ Review captions")
    st.caption("Each row is one caption and the clips it plays over. Edit the text right in the boxes. "
               "Left = setup, right = context/punchline.")
    caps = pending[0]["captions"]
    cuts = snap.values.get("cut_list", [])
    moments = {m["id"]: m for m in snap.values.get("clip_log", [])}
    ver = len(snap.values.get("log", []))
    edited = []
    for i, c in enumerate(caps):
        seg = [k for k in cuts if c["rec_in_s"] - 1e-3 <= k["rec_in_s"] < c["rec_out_s"] - 1e-3]
        picks = seg[:: max(1, len(seg) // 4)][:4]
        with st.container(border=True):
            strip, left, right = st.columns([3, 2, 2], vertical_alignment="center")
            with strip:
                st.image([thumb(pdir, k["source"], (k["src_in_s"] + k["src_out_s"]) / 2) for k in picks], width=72)
                acts = list(dict.fromkeys(moments[k["moment_id"]]["action"] for k in seg if k["moment_id"] in moments))
                st.caption(f"**{c['rec_in_s']:.1f}–{c['rec_out_s']:.1f}s** · {len(seg)} shots · " + "; ".join(acts[:2]))
            with left:
                l = st.text_input("Left", c["left"], key=f"L{i}_{ver}")
            with right:
                r = st.text_input("Right", c["right"], key=f"R{i}_{ver}")
            edited.append({**c, "left": l, "right": r})

    pv, _ = st.columns([1, 2])
    with pv:
        if st.button("▶ Preview video with these captions", width="stretch"):
            with st.spinner("Rendering a quick preview (about 20 seconds)…"):
                st.session_state[f"preview_{project_name}"] = render_caption_preview(pdir, snap.values, edited)
    prev = st.session_state.get(f"preview_{project_name}")
    if prev and Path(prev).exists():
        st.video(prev, width=320)

    st.divider()
    b1, b2 = st.columns([1, 2])
    with b1:
        if st.button("✅ Approve captions", type="primary", width="stretch"):
            st.session_state.pop(f"preview_{project_name}", None)
            changed = any(n["left"] != c["left"] or n["right"] != c["right"] for n, c in zip(edited, caps))
            resume(project, {"action": "edit", "captions": edited} if changed else {"action": "approve"})
    with b2:
        note = st.text_input("Or ask for a rewrite", placeholder="e.g. funnier, less literal")
        if st.button("🔁 Rewrite with this note") and note:
            st.session_state.pop(f"preview_{project_name}", None)
            resume(project, {"action": "redo", "note": note})

# --- 3. Paused at final review ---
elif pending and pending[0]["gate"] == "final":
    p = pending[0]
    st.subheader("🎞 Final review")
    v, info = st.columns([1, 1])
    with v:
        st.video(p["draft"])
    with info:
        m = p["critic"]["metrics"]
        a, b = st.columns(2)
        a.metric("Runtime", f"{m['runtime_s']}s", help="target ~19s")
        b.metric("Cuts / sec", m["cuts_per_s"], help="target 2.4–3.2")
        a.metric("Captions on cuts", f"{m['captions_on_cut_pct']}%", help="target ≥85%")
        b.metric("Cuts on beat", f"{m['cuts_on_beat_pct']}%", help="soft target")
        for f in p["critic"]["fails"]:
            st.warning(f)
        if not p["critic"]["fails"]:
            st.success("Passed every check in your spec.")
        if st.button("✅ Approve", type="primary", width="stretch"):
            resume(project, {"action": "approve"})
        st.divider()
        to = st.selectbox("Or send it back to…", ["planner", "commentary", "music"],
                          format_func={"planner": "Edit Planner (pacing / clip order)",
                                       "commentary": "Commentary Writer (captions)",
                                       "music": "Music (try a different song)"}.get)
        note = st.text_input("Note", placeholder="e.g. open on the matcha pour")
        if st.button("↩️ Send back"):
            resume(project, {"action": "revise", "to": to, "note": note})

# --- 4. Stopped mid-way (error or quit) ---
elif snap.next:
    st.info(f"Paused before: **{', '.join(snap.next)}**")
    if st.button("▶  Resume", type="primary"):
        run(project, None)

# --- 5. Finished ---
else:
    out = pdir / "out"
    st.success("Your reel is ready for Premiere.")
    v, info = st.columns([1, 1])
    drafts = sorted(out.glob("draft_v*.mp4"), key=lambda p: p.stat().st_mtime)
    with v:
        if drafts:
            st.video(str(drafts[-1]))
    with info:
        xml = snap.values.get("outputs", {}).get("xml")
        app = premiere_app()
        if xml and st.button("Open in Premiere Pro", type="primary", width="stretch"):
            subprocess.run(["open", "-a", app, xml] if app else ["open", "-R", xml])
        if st.button("Show files in Finder", width="stretch"):
            open_path(out)
        st.caption("In Premiere, if it asks for media, point it at this video's footage folder.")

if started:
    st.divider()
    with st.expander("Agent activity log"):
        for line in snap.values.get("log", []):
            st.text(line)
    if st.button("Start this video over", help="Clears saved progress and drafts for this video"):
        graph_for.clear()
        shutil.rmtree(pdir / ".agent", ignore_errors=True)
        shutil.rmtree(pdir / "out", ignore_errors=True)
        st.session_state.get("errors", {}).pop(project, None)
        st.rerun()
