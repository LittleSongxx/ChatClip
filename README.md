<div align="center">

<img src="./assets/banner_zh.png" alt="ChatClip" width="860" />

# ChatClip

**一个计划门控的 AI 剪辑 Agent：把一句话变成一条剪好的视频。**

你说出想要的成片，ChatClip 看完素材、拟出可执行计划、等你确认，
然后驱动每一刀剪辑、字幕、封面与导出——全程都在你掌控的护栏内。

[快速开始](#-快速开始) · [工作原理](#-工作原理) · [架构](#-架构) · [模型](#-模型)

`Python 3.10+` `LangGraph` `LangChain` `FastAPI` `SenseVoice` `FFmpeg`

[![License: NC-AL](https://img.shields.io/badge/license-Non--Commercial%20Attribution-blue)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-1342%20passing-brightgreen)](#-验证)

**简体中文** · [English](./README_EN.md)

</div>

---

## 为什么是 ChatClip

时间线剪辑软件让*你*干机械活。ChatClip 把分工倒过来：你只当导演，
Agent 来当剪辑团队。

- **动口不动手**——“剪一段 60 秒高光”“把讲定价的部分剪出来”“只保留
  主持人说话的片段”。Agent 负责找到对的时刻并组装成片。
- **没有你的签字什么都不执行**——每个任务先给出一份可读的计划（Agent
  理解了什么、会调用哪些工具、按什么顺序）。时间线草案、字幕校对、
  封面、正式导出各自停在明确的确认门；导出与删除永不自动执行。
- **证据优先，不靠感觉**——多模态检索（语音转写、视觉向量、OCR、说话
  人分离）让每一个提案片段都能回溯到源画面；没有匹配时诚实说无结果。
- **本地优先的媒体栈**——语音识别、向量检索、识别与渲染都在你的机器
  上跑；云端只有三个可选模型角色（视觉 / 规划 / 工具调用）。

### 系统架构

<div align="center">
  <img src="./assets/diagrams/architecture.svg" alt="ChatClip 系统架构图" width="900" />
</div>

### Agent 工作流

<div align="center">
  <img src="./assets/diagrams/workflow.svg" alt="计划门控工作流图" width="900" />
</div>

## 工作原理

流程总览见上方 **Agent 工作流图**，逐步说明：

1. **上传并描述**——丢进一个视频，打出你想要的剪辑要求。
2. **理解**——确定性解析器从你的句子里抽出时长目标、素材范围、锚点
   和交付物；技能路由选出剪辑方法论（内置 24 个 Skill：高光、话题、
   人脸匹配、声纹、短视频 Hook、封面、社媒画幅、交付质检……）。
3. **规划**——Agent 编译出可审计的步骤计划。内置技能的骨架由事实
   确定性编译，LLM 只贡献策略，绝不发明步骤。
4. **审批**——计划（绑定素材状态的哈希）等你确认。阅读期间素材变了？
   审批直接拒绝，重新规划。
5. **护栏内执行**——33 个媒体工具按序单步执行；长渲染停靠在持久操作
   日志上；每个计划有执行时限；token 用量、重规划与错误分类全程记录。
6. **审阅与迭代**——先出带水印的审核样片，正式导出必须明确确认。
   “再短一点”“从讲定价开始”等追问直接在时间线上原位返修。

## 架构

完整图示见上方 **系统架构图** 与 **Agent 工作流图**。要点：

- 单个 FastAPI 进程承载全部能力，没有任何旁路 Agent 服务
- 计划活在带检查点的 LangGraph 状态机里：`interrupt()` 为审批、审核
  确认与后台渲染暂停，`Command(resume=…)` 恢复——重启后依然有效
- 实体写入与审计事件同事务提交；凭据永不进入图检查点
- 深入阅读：[`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)

## 模型

| 角色 | 主推荐 | 降级候选 |
|---|---|---|
| 视觉（VLM） | `qwen3-vl-max` @ 阿里云百炼 | `doubao-seed-2.0` @ 火山方舟 |
| 剪辑规划 | `deepseek-flash` @ DeepSeek | `qwen3.8-max` @ 百炼 |
| Agent 工具调用 | `qwen3.8-max` @ 阿里云百炼 | `glm-4.6` @ 智谱 BigModel |

两个账号即可跑满主线。其余——SenseVoice 语音识别、SigLIP/E5/CLAP 向量、
OCR、人物识别、可选 TalkNet 主动说话人检测——全部本地运行。

## 快速开始

**环境要求**——x86-64 Linux/WSL2 · Python 3.10–3.11 · Node 22（仅动效
渲染需要）· 带 `libx264` + `drawtext` 的 FFmpeg · 磁盘 ≥10 GiB。

```bash
git clone https://github.com/LittleSongxx/ChatClip.git
cd ChatClip

python3 tools/setup.py --profile auto    # 引导式安装，自动选择 CPU/GPU
cp .env.example .env                     # 填入你的 API Key（CHATCLIP_* 变量）
./start.sh                               # → http://127.0.0.1:5180
```

在 **设置** 中把 Key 填入预选模型（Agent 模型需通过真实工具调用探测），
上传视频，描述你的剪辑要求。

<details>
<summary><b>Docker</b></summary>

```bash
cp .env.example .env      # 填入模型 Key
docker compose up --build -d                       # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d   # GPU
```
</details>

<details>
<summary><b>配置</b></summary>

所有变量使用 `CHATCLIP_*` / `VISION_*` / `LLM_*` / `AGENT_*` 前缀——
见 [`.env.example`](./.env.example) 与完整参考
[`docs/environment.example`](./docs/environment.example)。重点：

- `CHATCLIP_AGENT_PLAN_DEADLINE_SECONDS`——单计划执行时限
- `CHATCLIP_AGENT_MAX_MESSAGES_PER_10MIN`——每工作区消息限流
- `CHATCLIP_ACTIVE_SPEAKER_MODE`——可选 TalkNet 集成
</details>

## 验证

```bash
python -m pytest -q                # 1342 项后端测试
npm run test:frontend              # 306 项浏览器/契约测试
npm run test:agent-scenarios       # Agent 门禁（场景 + 硬化 + 预算）
python3 tools/doctor.py            # 环境体检
```

## 目录结构

```
app/agent/        LangGraph 编排（图、规划器、编译器、人工门）
app/agent_tools/  33 个媒体工具处理器（按域拆分）
app/llm/          LangChain 模型层（Provider、客户端）
app/main.py       媒体内核（任务、分析、渲染、质检）
static/           原生 JS 剪辑工作台
skills/           24 个剪辑 Skill（SKILL.md 策略文档）
tools/            安装器、体检、验证器、基准
tests/            1342 项测试（含浏览器套件）
```

## 路线图

- [ ] 真实视频评测集与公开基准报告
- [ ] 说话人/人脸身份跨会话持久化
- [ ] 更多交付目标（平台专属包）
- [ ] 多素材项目（合并多个上传）

## 许可

基于[非商业署名许可](./LICENSE)发布。

<div align="center">

**如果 ChatClip 替你省下了在时间线上逐帧拖拽的一个下午，一个 ⭐ 就是回报。**

</div>
