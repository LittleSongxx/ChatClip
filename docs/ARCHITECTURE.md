# ChatClip 系统架构（LangGraph 版）

本文描述当前代码的真实架构，供维护者与编码 Agent 使用。安装部署请读
[`AI_INSTALL.md`](AI_INSTALL.md)，完整环境变量见 [`environment.example`](environment.example)。

## 总览

ChatClip 是单进程中文 AI 视频剪辑助手：FastAPI 服务同时承载 Web 界面、
媒体内核与 LangGraph Agent。2026-09 重构后，原 Node/Pi Agent 子服务已被
移除，Agent 编排全部在 Python 进程内完成。

```
┌────────────────────────── uvicorn app.main:app (默认 127.0.0.1:5180) ──────────────────────────┐
│                                                                                                 │
│  static/ (vanilla JS SPA + 4 个 Preact 岛)                                                      │
│      │ HTTP/SSE（/api/agent/*、/api/jobs/*、/api/settings/* …）                                 │
│      ▼                                                                                          │
│  API 层 app/*_api.py（薄路由，按域拆分）                                                          │
│      ▼                                                                                          │
│  ┌──────── app.agent（LangGraph 编排域）────────┐   ┌──────── app/llm（LangChain 模型层）──────┐ │
│  │ AgentPlatform + StateGraph + SqliteSaver    │──▶│ create_chat_model（角色×provider 工厂）  │ │
│  │ planner（强制工具调用）/ compiler（确定性编译）│   │ LangChainJsonClient（VLM/JSON 客户端） │ │
│  │ 33 工具目录 + 人工门（interrupt）             │   └────────────────────────────────────────┘ │
│  └──────────────┬───────────────────────────────┘                                             │
│                 │ configure_tool_dispatcher（DI 缝）                                            │
│                 ▼                                                                               │
│  媒体内核 app/main.py：分析管线 pipeline.py、content_search、speech、recognition、               │
│  渲染 media.py、封面 cover_art、TalkNet、字幕、QC；SQLite 任务/队列/输出                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 目录结构（核心）

| 路径 | 职责 |
|---|---|
| `app/main.py` | 媒体内核：任务生命周期、分析/渲染调度（媒体工具实现见 `app/agent_tools/`） |
| `app/agent_tools/` | 33 个媒体工具处理器（context/content/timeline/subtitle/cover/delivery + registry），经 `dispatch_agent_tool` 薄路由调用 |
| `app/agent/` | **Agent 编排域**（详见下节） |
| `app/llm/` | **模型层**：provider 注册表与预设、LangChain 模型工厂、JSON/多模态客户端适配 |
| `app/vision_settings.py` | 三角色配置存储（视觉/规划/Agent），泛型 `ModelRoleStore` |
| `app/pipeline.py` 等 | 高光分析、内容检索、语音、识别、渲染等媒体能力 |
| `static/` `frontend/` | 前端（vanilla JS 主体 + Vite 构建的 4 个 Preact 小部件） |
| `skills/` | 24 个 SKILL.md 策略文档（编辑方法论，非可执行代码） |
| `tools/` | 安装/启动/诊断/校验脚本 |

## app/agent/ 模块

| 模块 | 职责 |
|---|---|
| `catalog.py` | 33 个媒体工具的 JSON Schema 目录与 sideEffect 分级（read/analysis/preview/identity/review/export/delete） |
| `brief.py` | 中文目标的确定性 brief 抽取（时长/范围/锚点/交付物/身份目标）与用户可读理解摘要 |
| `skills.py` | SKILL_PROFILES 注册表（skill → 工作流 kind + 授权工具）、路由资格、技能链组合 |
| `compiler.py` | 托管技能的确定性计划编译器 + 计划规范化/校验（DAG、参数 schema、时间线审批门、步骤签名） |
| `planner.py` | LangChain 强制工具调用会话（submit_plan/select_skill/submit_skill/能力探针），流式 emit |
| `graph.py` | LangGraph StateGraph 装配：planner → compile → 审批门 → 执行循环 → 动作/操作门 → 重规划 → finish |
| `platform.py` | `AgentPlatform`：AgentStore（SQLite）、checkpointer、审批/取消/确认/重试 API、恢复逻辑 |
| `prompts.py` | Planner/Skill Creator/Router/Probe 系统提示词 |

## Agent 生命周期（LangGraph 状态机）

每个计划一个图线程（checkpoint 存于 `data/agent/graph.sqlite3`），人工门
全部是 `interrupt()` 暂停点，API 确认/后台 Future 完成后以
`Command(resume=...)` 恢复：

1. **规划**：`prepare_plan_request` 预留工作区 → 确定性 brief → 规则梯路由
   技能（LLM 兜底）→ planner 节点以 `bind_tools([submit_plan], tool_choice)`
   获取计划（流式 token/思考事件经 custom stream 转 SSE）。
   托管技能的 LLM 步骤会被丢弃，由 `compiler` 生成步骤骨架。
2. **审批门**：`interrupt(planHash)` 暂停；`POST /plans/{id}/confirm` 校验
   哈希后恢复执行。
3. **执行循环**：按依赖顺序单步执行；`identity/review` 副作用与导出在
   分步模式下进入**动作门** `interrupt`，由审核面板结构化确认后恢复；
   返回 Future 的长任务（检索/渲染）进入**操作门** `interrupt`，Future 回调
   经单线程驱动器投递、跨线程阻塞在每计划锁上直至图挂起后以终态恢复
   （避免回调与仍在运行的派发节点竞态）。
4. **重规划**：必需步骤失败时最多 2 次；依赖签名复用已完成步骤，minor 修订
   继续执行，material 修订回到审批门。
5. **恢复**：进程重启后 `recover_completed_operations` 结算游离操作，
   `_kick_plan` 续跑未完成计划。

安全语义（保持不变）：export/delete 永不自动执行（`export_editing_draft`
除外）；时间线必须先提案后确认；planHash 绑定审批（含工具目录指纹与提示词
版本）；确认值与任务/步骤绑定（receipt 含出现代）。

可靠性设施：后台操作登记 durable journal（CAS：planId+stepId+operationId+
planRevision；迟到结果记 `late` 并阻断）；单计划 wall-clock deadline 与每
工作区消息限流；Run 级 trace（模型指纹/token/时长/重规划/错误分类）；
AgentStore 实体+事件同事务原子写；计划 API 携带 `statusView` 四态投影
（planStatus/artifactStatus/qualityStatus/userActionStatus）。

## 模型层（app/llm）与推荐主线

三个远程模型角色，配置存储于 `data/{vision,llm,agent}-settings.json`：

| 角色 | 用途 | 主推荐 | 降级候选 |
|---|---|---|---|
| VISION | 多模态高光分析（图/视频输入） | qwen3-vl-max @ 阿里云百炼 | doubao-seed-2.0 @ 火山方舟 |
| LLM | 剪辑规划/结构化 JSON | deepseek-flash（DeepSeek-V4.1-Flash）@ DeepSeek | qwen3.8-max @ 百炼 |
| AGENT | 强制工具调用（submit_plan 等） | qwen3.8-max @ 阿里云百炼 | glm-4.6 @ 智谱 BigModel |

主线只需两个账号（百炼 + DeepSeek）。provider 工厂支持
bailian/deepseek/bigmodel/ark/openai/openai_compatible/anthropic*，模型列表
通过各平台 `/models` 运行时发现。本地模型栈（国产开源）不变：
SenseVoice ASR（降级 faster-whisper）、CAM++ 声纹、SigLIP/WeMM 视觉检索、
E5/CLAP 嵌入、GroundingDINO/YuNet/SFace/YOLOX/YouTuReID、PaddleOCR、
TalkNet（可选安装）。

**功能主线**（默认启用组合）：上传 → SenseVoice 转写（标点/分离默认关）→
多模态识别索引（SigLIP+OCR）→ `chatclip-highlight-director` 智能高光技能 →
计划确认 → 时间线 → 自动字幕 → 水印审核预览 → 导出。其余技能（内容检索/
人物/说话人/封面/动效/横竖屏/交付）与模块（声纹、WeMM、TalkNet、whisper）
按需启用。

## 数据与部署

- 数据根 `data/`：jobs/analysis/render SQLite、uploads、outputs、kept、
  cache、models（本地模型缓存）、agent/（AgentStore + graph.sqlite3 + 技能）、
  三份模型配置 JSON（0600）。
- 单实例 flock 锁（`data/.workspace-worker.lock`）；分析/渲染为持久任务队列
  （租约 + 心跳），渲染任务支持同任务并发预览。
- 部署：本机 `bash start.sh`（uvicorn 前台）；Docker 单服务
  `docker compose up`（CPU）或叠加 `docker-compose.gpu.yml`（cu121）。GPU
  profile 提供 SenseVoice/识别栈的 CUDA 加速。
- 已移除：Node agent-service、Pi 运行时、npm 插件沙箱、restart.sh 守护路径、
  `HIGHLIGHT_*`/`ARK_*` 遗留环境变量（统一 `CHATCLIP_*`）。

## 测试与校验

- `pytest`（tests/，后端全覆盖）+ `node --test`（tests/frontend，含浏览器冒烟）
- `python3 tools/check_repository.py --mode source|deployment`：提交边界与
  配置覆盖（config.py 引用的每个环境变量必须在示例中可查）
- `tools/validate_agent_scenarios.py` / `validate_workflows.py`：场景级校验
- `tools/doctor.py`：环境体检（visual/cpu/cuda profile）
