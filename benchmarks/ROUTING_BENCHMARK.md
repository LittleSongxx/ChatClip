# 意图路由准确率评测口径

`benchmarks/workflow-intent-v2.jsonl` 固定 100 例中文（含少量英文）剪辑
指令，5 类均衡（highlight / content_search / person_edit / speaker_edit /
clarification 各 20）。三模式对照：

| 模式 | 被测对象 | 运行方式 |
|---|---|---|
| rules | 纯确定性规则梯 `route_editing_instruction` | 离线，无需服务与模型 |
| llm | 规划模型原始分类（normalize 前，读 `rawWorkflowKind`） | HTTP 打真实服务 |
| hybrid | 模型为主 + 规则强冲突守卫（线上实际路径） | HTTP 打真实服务 |

## 判据

- **exact** = 判定类别 == 标注类别；clarification 视为判定类别之一
  （期望 clarification 的样例必须 needsConfirmation）。
- **unsafe** = 判定为四工作流之一且与标注不一致（错误自动路由）；
  clarification 永不算 unsafe。通过线：exactRate ≥ 阈值 且 unsafe = 0。
- `--minimum-exact-rate` 默认 0.95 沿用旧 45 例门禁；100 例口语化扩充后
  的实际门禁以首轮三模式基线为准再定。

## 已有基线（2026-09-29，rules 离线）

exactRate 0.47、unsafe 7（speaker/person 误路由为 content_search 的样例
在纯规则下不安全）。该数字即"纯规则不足、需混合路由"的对照基线；
llm / hybrid 基线待服务配置模型后补记。

## 边界

- 旧 45 例语料与 `tests/test_intent_router.py` 回归门禁保持不动；本基准
  只用 v2 语料。
- 语料可选字段 `expectedKind` 记录 skill 级真值（shortform/multi-topic
  等），供后续 skill 级路由评测与 E2E 诊断使用，不参与 intent 计分。
