# E2E 任务成功率评测口径

`benchmarks/e2e-tasks.jsonl` 固定 25 条中文剪辑目标，跑真实 HTTP 服务、
真实模型、真实分析与渲染（`npm run benchmark:e2e`，门禁 `benchmark:e2e:check`）。

## 判据（确定性，无 LLM judge）

- **成功** = 计划终态 `preview_ready`/`completed` + 本计划产物全部通过
  ffmpeg 全解码（`-xerror`）+ 任务声明的时长（±容差）与画幅断言 + 源有音轨
  则产物必须保留音轨。多版本成片时任一版本满足断言即通过。
- **不成功**：终态 `failed`/`no_result`/超时取消、断言失败、或 autonomous
  模式出现 `action_required`（记为 humanIntervention，属人工介入而非成功）。
- `expectedProfile` 只做路由诊断记录（profileMatch），**不改变**任务成败；
  路由准确性由 workflow-intent 基准单独评测。
- 分母 = 全部排期任务；依赖未成功的链式任务记 `blocked` 保留在报告中，
  不从分母删除。

## 汇总指标

taskSuccessRate、avgReplanCount、avgPlanningTokens（规划 LLM token）、
humanInterventionCount、wallClockSeconds。

## 样本与复现

- 源视频 4 段（Wikimedia Commons，普通话，CC BY / CC BY-SA，总时长约 32
  分钟）：见 `benchmarks/media/SOURCES.md`；媒体文件不入库。
- 上传走 `POST /api/jobs`（每 IP 每小时 10 次限流）；`--jobs-file` 缓存
  源视频→job 映射，重跑复用已上传 job。
- 每任务独立 workspace，autonomous_review 模式，服务端 30 条消息/10 分钟
  限流对单 workspace 不构成约束。
- 失败样本（failedSteps、assertionFailures、interventionSteps）完整保留在
  `test-results/e2e-benchmark-report.json`，禁止为提高成功率删除任务；
  新增任务须同步更新本文件与 SOURCES.md。

## 已知边界

- 内容主题相关性（"讨论喜剧艺术的部分"是否剪对了内容）不在本判据内，
  由 interview-content-search 质量门禁与人工抽审覆盖。
- 首轮为报告模式（不设 `--minimum-success-rate` 结论）；取得基线后再定
  门禁阈值，避免无依据阈值。
