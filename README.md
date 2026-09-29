<div align="center">

<img src="./assets/banner.png" alt="ChatClip Banner" width="100%" />

# ChatClip ✂️

### An AI Agent That Edits Videos Through Conversation

**Just say it. It's edited.**

`FastAPI` `LangGraph` `LangChain` `SenseVoice` `FFmpeg`

**English** · [简体中文](./README_zh.md)
</div>

---

## 💡 What is ChatClip?

ChatClip is an **AI video-editing agent**. You don't drag clips on a timeline or scrub through hours of footage — you describe what you want in natural language, and the agent handles the entire process:

**understanding the footage → locating the target content → planning the edit → executing cuts → delivering the final clips.**

<div align="center">
  <img src="./assets/showcase/conversational-highlight-editing-preview.gif" alt="ChatClip conversational highlight editing workflow" width="900" />
  <br />
  <sub><b>"Make a highlight from the best moments."</b> — ChatClip analyzes the footage, presents the event timeline, and delivers AI-edited versions.</sub>
</div>

Every agent action is **plan-gated**: the agent presents an auditable editing plan (with its understanding of your goal) before executing anything. Timeline drafts, subtitle reviews, covers, and formal exports each require your explicit confirmation — the agent never silently publishes or deletes media.

---

## ✨ Core Features

### 💬 Conversational Editing

From *"make a highlight of the best moments"* to *"cut out the part where they introduce pricing"* — natural language becomes editing actions. Refine results with follow-ups such as *"make it shorter"* or *"start from the part about pricing"*.

### 🎯 Four Core Editing Capabilities

* **⚡ Highlight Extraction** — automatically identify and extract the most valuable moments from long-form footage.
* **🧭 Topic-Based Editing** — locate and extract clips around a specific topic.
* **👤 Face-Matched Editing** — find a target person and extract the segments where they appear on screen.
* **🔊 Voiceprint-Based Editing** — identify a target speaker by voiceprint and extract the segments where they speak.

### 🛠️ Extensible Skill System

24 built-in **Skills** (SKILL.md policy documents) encode editing know-how — short-form hook direction, subtitle revision, cover art, social reframing, delivery QC. Skills are managed in the UI; new ones can be generated from a natural-language description.

---

## 🧠 Architecture

One FastAPI process hosts everything — there is **no separate agent service**:

```
static/ (vanilla JS SPA)  ──HTTP/SSE──▶  app/*_api.py (thin routes)
                                            │
        app/agent/  LangGraph orchestration │  app/llm/  LangChain model layer
        planner · compiler · human gates    │  provider factory · JSON client
        (interrupt / Command resume)        │
                                            │
        app/agent_tools/  33 media tool handlers (content · timeline ·
        subtitle · cover · delivery) on top of the app/main.py kernel
```

* **LangGraph state machine** drives plan → approval → step execution → replan. Plan approval, structured review confirmations, and background render/analysis operations are framework-level `interrupt()` pauses resumed by `Command(resume=...)`.
* **Deterministic compiler** — for built-in skills the step skeleton is compiled from facts (not invented by the model); the planning model contributes strategy only.
* **Safety semantics** — exports/deletes never run autonomously, identity/review steps gate on structured user confirmation, and approvals are bound to a content hash.

Full details: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) · agent internals: [`docs/agent-platform.md`](./docs/agent-platform.md)

---

## 🤖 Models

Three remote roles, pre-filled with the recommended domestic (CN) model line — **only two accounts needed**:

| Role | Recommended | Fallback candidate |
|---|---|---|
| Vision (VLM) | `qwen3-vl-max` @ Alibaba Bailian | `doubao-seed-2.0` @ Volcengine Ark |
| Editing planner | `deepseek-flash` (DeepSeek-V4.1-Flash) @ DeepSeek | `qwen3.8-max` @ Bailian |
| Agent (tool calling) | `qwen3.8-max` @ Alibaba Bailian | `glm-4.6` @ Zhipu BigModel |

* **Local multimodal stack · included** — SenseVoice ASR (optional whisper fallback), SigLIP/E5/CLAP embeddings, OCR, anonymous person recognition; weights download on first use. TalkNet active-speaker detection is an optional install.
* Configure via `.env` (see [`.env.example`](./.env.example)) or the in-app **Settings** page; the agent model must pass a real tool-calling probe before saving.

---

## 🚀 Quick Start

Prerequisites: x86-64 Linux/WSL2 · Python 3.10–3.11 · Node.js 22 (only for the local motion renderer) · FFmpeg/FFprobe with `libx264` + `drawtext` · ≥10 GiB free disk.

```bash
git clone https://github.com/LittleSongxx/ChatClip.git
cd ChatClip

python3 tools/setup.py --profile auto   # guided installer (CPU/GPU auto-detect)
cp .env.example .env                    # fill your API keys (CHATCLIP_* vars)
./start.sh                              # http://127.0.0.1:5180
```

Open the printed URL, paste your API keys over the pre-selected models in **Settings**, then upload a video and describe the edit you want.

---

## ⚙️ Deployment

### 🐳 Docker

```bash
cp .env.example .env   # fill model keys
docker compose up --build -d          # CPU
# GPU (NVIDIA Container Toolkit required):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Docker publishes to `127.0.0.1` only. For remote access set `CHATCLIP_BIND_ADDRESS=0.0.0.0` **together with** a strong `CHATCLIP_ACCESS_TOKEN`, and put an authenticated HTTPS reverse proxy in front.

### 🔧 Optional setup

* **NVIDIA GPU** — native: install `requirements-gpu.txt`; validate with `python3 tools/doctor.py --profile cuda`.
* **TalkNet active-speaker detection** — installed by the local installer; verify in the in-app capability panel.
* **Local model warm-up** — `python3 tools/prepare_recognition_models.py --data-root data`.
* **All environment variables** — [`docs/environment.example`](./docs/environment.example).

---

## ✅ Verification

```bash
python -m pytest -q                 # backend suite (1,300+ tests)
npm run test:frontend               # browser/contract tests
python3 tools/check_repository.py --mode deployment
python3 tools/doctor.py             # environment check
```

---

## 📄 License

This project is released under the [Non-Commercial Attribution License](./LICENSE).

<div align="center">

⭐ If you find ChatClip useful, please give it a star!

Made with ❤️ by [LittleSongxx](https://github.com/LittleSongxx)

</div>
