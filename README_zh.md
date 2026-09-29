<div align="center">

<img src="./assets/banner_zh.png" alt="ChatClip Banner" width="100%" />

# ChatClip ✂️

### 一个通过对话完成视频剪辑的 AI Agent

**只需要说出来，视频就剪好了。**

`FastAPI` `LangGraph` `LangChain` `SenseVoice` `FFmpeg`

[English](./README.md) · **简体中文**
</div>

---

## 💡 ChatClip 是什么？

ChatClip 是一个**AI 视频剪辑 Agent**。你不需要在时间线上拖动素材、也不需要逐帧翻看几个小时的素材——只需要用自然语言描述你想要的成片，剩下的交给它：

**理解素材 → 定位目标内容 → 规划剪辑 → 执行剪辑 → 交付成片。**

<div align="center">
  <img src="./assets/showcase/conversational-highlight-editing-preview.gif" alt="ChatClip 对话式高光剪辑工作流" width="900" />
  <br />
  <sub><b>“把最精彩的部分剪成高光集锦。”</b>——ChatClip 分析素材、给出事件时间线，并交付 AI 剪辑版本。</sub>
</div>

Agent 的每一个动作都是**计划门控**的：执行任何操作前，它会先给出一份可审计的剪辑计划（包括它对你目标的理解）。时间线草案、字幕校对、封面、正式导出都需要你的明确确认——Agent 绝不会悄悄发布或删除媒体。

---

## ✨ 核心特性

### 💬 对话式剪辑

从“把最精彩的部分剪成高光”到“把介绍定价的部分剪出来”——自然语言直接变成剪辑动作。还可以用“再短一点”“从讲定价的地方开始”这样的追问继续修改。

### 🎯 四大剪辑能力

* **⚡ 高光提取** — 自动从长视频中识别并提取最有价值的片段。
* **🧭 话题剪辑** — 围绕指定话题定位并提取相关片段。
* **👤 人脸匹配剪辑** — 找到目标人物，提取其出镜片段。
* **🔊 声纹剪辑** — 通过声纹识别目标说话人，提取其发言片段。

### 🛠️ 可扩展的技能系统

内置 24 个 **Skill**（SKILL.md 策略文档），沉淀剪辑方法论——短视频 Hook 编排、字幕返修、封面生成、社媒画幅转换、交付质检等。技能可在界面中管理，也支持用自然语言生成新技能。

---

## 🧠 架构

单个 FastAPI 进程承载一切——**没有独立的 Agent 子服务**：

```
static/（原生 JS 单页应用）──HTTP/SSE──▶  app/*_api.py（薄路由层）
                                              │
      app/agent/  LangGraph 编排域            │  app/llm/  LangChain 模型层
      规划器 · 编译器 · 人工门                │  Provider 工厂 · JSON 客户端
      （interrupt / Command 恢复）            │
                                              │
        app/agent_tools/  33 个媒体工具处理器（内容 · 时间线 ·
        字幕 · 封面 · 交付），构建于 app/main.py 媒体内核之上
```

* **LangGraph 状态机**驱动 计划 → 审批 → 步骤执行 → 重规划。计划审批、结构化审核确认、后台渲染/分析等待都是框架级 `interrupt()` 暂停点，由 `Command(resume=...)` 恢复。
* **确定性编译器** — 内置技能的步骤骨架由事实编译而来（不由模型发明），规划模型只贡献策略。
* **安全语义** — 导出/删除永不自动执行；身份/审核步骤必须经用户结构化确认；审批与内容哈希绑定。

完整说明：[`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) · Agent 内部细节：[`docs/agent-platform.md`](./docs/agent-platform.md)

---

## 🤖 模型

三个远程模型角色，已预填推荐的国产模型主线——**只需两个账号**：

| 角色 | 主推荐 | 降级候选 |
|---|---|---|
| 视觉（VLM） | `qwen3-vl-max` @ 阿里云百炼 | `doubao-seed-2.0` @ 火山方舟 |
| 剪辑规划 | `deepseek-flash`（DeepSeek-V4.1-Flash）@ DeepSeek | `qwen3.8-max` @ 百炼 |
| Agent（工具调用） | `qwen3.8-max` @ 阿里云百炼 | `glm-4.6` @ 智谱 BigModel |

* **本地多模态能力 · 已包含** — SenseVoice 语音识别（可选 whisper 降级）、SigLIP/E5/CLAP 向量、OCR、匿名人物识别；权重首次使用时下载。TalkNet 主动说话人检测为可选安装。
* 通过 `.env`（见 [`.env.example`](./.env.example)）或页面 **设置** 配置；Agent 模型保存前必须通过真实工具调用探测。

---

## 🚀 快速开始

环境要求：x86-64 Linux/WSL2 · Python 3.10–3.11 · Node.js 22（仅本地图文动效渲染需要）· 带 `libx264` + `drawtext` 的 FFmpeg/FFprobe · 磁盘 ≥10 GiB。

```bash
git clone https://github.com/LittleSongxx/ChatClip.git
cd ChatClip

python3 tools/setup.py --profile auto   # 统一安装器（自动选择 CPU/GPU）
cp .env.example .env                    # 填入你的 API Key（CHATCLIP_* 变量）
./start.sh                              # http://127.0.0.1:5180
```

打开终端输出的地址，在 **设置** 中把 API Key 填入预选模型，然后上传视频、描述你的剪辑要求。

---

## ⚙️ 部署

### 🐳 Docker

```bash
cp .env.example .env   # 填入模型 Key
docker compose up --build -d          # CPU
# GPU（需 NVIDIA Container Toolkit）：
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Docker 默认只发布到 `127.0.0.1`。远程访问时需设置 `CHATCLIP_BIND_ADDRESS=0.0.0.0` **并同时**配置强 `CHATCLIP_ACCESS_TOKEN`，前置带身份验证的 HTTPS 反向代理。

### 🔧 可选配置

* **NVIDIA GPU** — 本机安装 `requirements-gpu.txt`，用 `python3 tools/doctor.py --profile cuda` 验证。
* **TalkNet 主动说话人检测** — 本机安装器默认安装，在页面能力面板查看状态。
* **本地模型预热** — `python3 tools/prepare_recognition_models.py --data-root data`。
* **全部环境变量** — [`docs/environment.example`](./docs/environment.example)。

---

## ✅ 验证

```bash
python -m pytest -q                 # 后端测试（1300+ 项）
npm run test:frontend               # 浏览器/契约测试
python3 tools/check_repository.py --mode deployment
python3 tools/doctor.py             # 环境体检
```

---

## 📄 许可

本项目基于[非商业署名许可](./LICENSE)发布。

<div align="center">

⭐ 如果 ChatClip 对你有用，欢迎点一个 Star！

Made with ❤️ by [LittleSongxx](https://github.com/LittleSongxx)

</div>
