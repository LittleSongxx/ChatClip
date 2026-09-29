# ChatClip Agent Platform

ChatClip 的 Agent 编排运行在 FastAPI 进程内的 LangGraph 状态机上，媒体内核
保持 Python 单体。浏览器只与 FastAPI 通信；系统整体结构见
[ARCHITECTURE.md](ARCHITECTURE.md)。

> 2026-09 重构说明：原 Node/Pi agent-service（单提示强制工具调用 + npm 插件
> 沙箱）已被移除。规划、技能路由、Skill 生成与能力探针由
> `app/agent/planner.py` 以 LangChain `bind_tools` 强制调用实现；人工审批、
> 结构化确认与后台操作等待由 `interrupt()`/`Command(resume)` 承载；插件系统
> 不再提供，扩展机制是 SKILL.md 技能（由 `SKILL_PROFILES` 托管工作流）。

## 进程与所有权

- 单进程单实例：加载数据前获取 `data/.workspace-worker.lock` 排他锁；第二个
  实例显式报错。队列所有权使用租约与心跳；恢复不会窃取活跃租约，仅在持有
  进程锁后释放被遗弃的租约。关闭时保持锁直到后台写入器停止。
- Agent 状态（workspaces/skills/plans/runs/events）在 `data/agent/agent.sqlite3`；
  图控制态（interrupt 暂停点）在 `data/agent/graph.sqlite3`（SqliteSaver）。
  业务状态以 AgentStore 为准，checkpoint 只承载图的恢复位置。

## 模型与凭据

- Agent 模型通过 `AGENT_*` 或页面“设置”配置；保存时执行真实的 Tool Calling
  探针（`planner.probe`），只返回 JSON 文本的模型会被拒绝。
- 推荐主线：视觉 qwen3-vl-max（百炼）、规划 deepseek-flash（DeepSeek）、
  Agent qwen3.8-max（百炼）。详见 [ARCHITECTURE.md](ARCHITECTURE.md)。
- 浏览器访问令牌（`CHATCLIP_ACCESS_TOKEN`）与模型 API Key 互相独立；密钥
  不进入日志、文档或前端。

## 输出交付与持久化

正式导出请求携带 `outputFilename` 与随输出返回的不透明 `outputRevision`。
确认会冻结所选编辑及其字幕设置；相同导出请求返回同一持久 `operationId`。
输出 `capabilities` 是 keep/download/edit 动作的权威来源。SQLite 是任务的
权威存储（revision 检查 + 事务提交），JSON 仅为可重建备份。完整交付语义见
[transactional delivery](transactional-delivery.md)。

## Execution contract

- 执行器按依赖顺序单步运行；`identity`/`review` 步骤与导出在分步模式下经
  `interrupt()` 请求结构化确认，确认值必须携带匹配的 jobId/stepId。
- 返回 Future 的媒体操作进入操作门；Future 完成回调以终态恢复图线程，
  服务重启后由 `recover_completed_operations` 结算并续跑。
- 必需步骤失败触发最多 2 次重规划：依赖签名复用已完成步骤；minor 修订自动
  继续，material 修订要求重新审批（新 planHash）。
