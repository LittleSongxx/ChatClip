"""Content-inspection and identity-selection Agent tool handlers."""
from __future__ import annotations

from typing import Any

from .context import ToolContext


def inspect_workspace(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    required_state = str(arguments.get("requiredState") or "")
    if required_state:
        reason_code = str(arguments.get("preconditionCode") or "missing_precondition")
        message = str(arguments.get("preconditionMessage") or "当前任务不满足所选编辑能力的前置条件")
        has_timeline = bool(ctx.snapshot.get("activeEditSessionId"))
        has_output = bool(ctx.snapshot.get("outputs"))
        has_external_cover_asset = bool(ctx.snapshot.get("externalCoverAsset"))
        satisfied = (
            (required_state == "active_timeline" and has_timeline)
            or (required_state == "current_output" and has_output)
            or (required_state == "external_cover_asset" and has_external_cover_asset)
        )
        if required_state == "compatible_request" or not satisfied:
            next_step = "先在当前任务生成并确认成片" if required_state == "current_output" else "先在当前任务建立并确认时间线"
            if required_state == "compatible_request":
                next_step = "修改目标或改用适合源素材检索与剪辑的 Skill"
            elif required_state == "external_cover_asset":
                next_step = "修改封面要求，改用本次视频中的画面"
            return ctx.no_result(
                reason_code, f"前置条件未满足：{message}",
                artifact_kind="precondition_missing",
                suggestions=[next_step, "确认当前工作区绑定的是本次任务", "不要复用其他任务产物"],
            )
    public = _main.public_job(ctx.snapshot)
    return {
        "artifact": {
            "kind": "workspace_snapshot", "jobId": ctx.job_id,
            "status": public.get("status"), "stage": public.get("stage"),
            "duration": public.get("duration"), "workflowKind": _main.workflow_kind_for_job(ctx.snapshot),
            "candidateCount": len(public.get("candidates") or []),
            "outputCount": len(public.get("outputs") or []),
        },
    }


def validate_task_provenance(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    _main.public_job(ctx.snapshot)
    editing = _main.agent_planning_context(ctx.job_id).get("editing") if ctx.job_id else {}
    issues: list[dict[str, Any]] = []
    if str(ctx.workspace.get("jobId") or "") != ctx.job_id:
        issues.append({"severity": "error", "code": "workspace_job_mismatch", "message": "工作区绑定任务与当前任务不一致。"})
    source_hash = str(ctx.snapshot.get("sourceHash") or ctx.snapshot.get("sourceAssetId") or "")
    for cover in ctx.snapshot.get("coverVersions") or []:
        if isinstance(cover, dict) and str(cover.get("sourceAssetId") or "") and str(cover.get("sourceAssetId") or "") != source_hash:
            issues.append({"severity": "warning", "code": "cover_source_mismatch", "message": "存在来源不匹配的封面版本。"})
            break
    if ctx.snapshot.get("agentHandoffJobId") and not str(ctx.workspace.get("sourceJobId") or ""):
        issues.append({"severity": "warning", "code": "handoff_without_source_workspace", "message": "任务曾发生同源交接，需确认当前工作区已切到新任务。"})
    return {
        "artifact": {
            "kind": "task_provenance_report", "jobId": ctx.job_id,
            "sourceHash": source_hash, "sourceScope": str((ctx.snapshot.get("request") or {}).get("sourceScopeKind") or "all"),
            "workspaceJobId": str(ctx.workspace.get("jobId") or ""),
            "hasOutputs": bool((editing or {}).get("hasOutputs")),
            "hasActiveSession": bool((editing or {}).get("hasActiveSession")),
            "issues": issues, "passed": not any(item["severity"] == "error" for item in issues),
        },
    }


def diagnose_edit_failure(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    public = _main.public_job(ctx.snapshot)
    plan_id = str(ctx.workspace.get("activePlanId") or "")
    plan = _main.agent_platform.store.get("plans", plan_id) if plan_id else None
    failed_steps = [
        {"id": str(item.get("id") or ""), "title": str(item.get("title") or ""), "tool": str(item.get("tool") or ""), "error": str(item.get("error") or "")}
        for item in (plan or {}).get("steps") or [] if str(item.get("status") or "") == "failed"
    ]
    no_result_steps = [
        item for item in (plan or {}).get("steps") or []
        if isinstance(item.get("result"), dict) and str(item["result"].get("terminalStatus") or "") == "no_result"
    ]
    causes: list[dict[str, Any]] = []
    if failed_steps:
        causes.append({"code": "failed_steps", "message": "Agent 计划存在失败步骤。", "items": failed_steps})
    if no_result_steps:
        causes.append({"code": "no_result", "message": "上游检索或时间线约束未得到可用结果。"})
    if not (public.get("outputs") or public.get("agentPreviewOutputs") or public.get("agentReviewPreviews")):
        causes.append({"code": "no_outputs", "message": "当前任务没有可播放的审核样片或成片。"})
    if public.get("status") in {"running", "queued", "cancelling"}:
        causes.append({"code": "background_running", "message": "当前任务仍有后台操作在执行。"})
    return {
        "artifact": {
            "kind": "edit_diagnostics", "jobId": ctx.job_id,
            "focus": str(arguments.get("focus") or "")[:240],
            "status": public.get("status"), "stage": public.get("stage"),
            "causes": causes or [{"code": "no_obvious_failure", "message": "未发现明确失败状态，请检查具体预览或导出对象。"}],
            "nextSteps": ["刷新当前任务状态", "查看失败步骤技术详情", "重新生成审核样片"] if causes else ["继续当前审核流程"],
        },
    }


def select_multi_topic_evidence(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        search = current.get("contentSearch") if current and isinstance(current.get("contentSearch"), dict) else {}
        candidates = [
            item for item in search.get("candidates") or []
            if isinstance(item, dict) and item.get("reviewStatus") != "rejected"
            and float(item.get("end") or 0) > float(item.get("start") or 0)
        ]
        reliable = [
            item for item in candidates
            if item.get("selected") is True
            or str(item.get("confidenceTier") or "") == "reliable"
            or (not str(item.get("confidenceTier") or "") and not bool(item.get("requiresReview")))
        ]
        reliable_ids = {str(item.get("id") or "") for item in reliable if str(item.get("id") or "")}
        spec = _main._content_assembly_spec(search, allowed_match_ids=reliable_ids)
        if not current or not str(search.get("id") or ""):
            return ctx.no_result("no_search", "当前任务还没有可用于多主题选择的内容检索结果。", query=str(arguments.get("query") or ""))
        if len(spec.get("predicateIds") or []) < 2:
            return ctx.no_result("not_multi_topic", "当前检索没有拆出多个必需主题，无法按多主题配额编排。", query=str(arguments.get("query") or ""), candidate_count=len(candidates), reliable_count=len(reliable))
        if spec.get("missingPredicates"):
            return ctx.no_result(
                "missing_required_categories",
                "自动编排缺少必要类别的可靠候选：" + "、".join(str(value) for value in spec["missingPredicates"]),
                query=str(arguments.get("query") or search.get("instruction") or ""),
                candidate_count=len(candidates), reliable_count=len(reliable),
            )
        selected_ids = [str(value) for value in spec.get("selectedMatchIds") or [] if str(value) in reliable_ids]
        if not selected_ids:
            return ctx.no_result("no_match", "多主题检索没有生成可用于自动编排的可靠候选。", query=str(arguments.get("query") or ""), candidate_count=len(candidates), reliable_count=len(reliable))
        search["reviewDraft"] = {
            "schemaVersion": "content-review-draft-v1",
            "searchId": str(search["id"]),
            "selectedMatchIds": selected_ids,
            "orderedMatchIds": selected_ids,
            "source": "agent_multi_topic_selection",
            "updatedAt": _main.now_iso(),
        }
        search["defaultSelectedIds"] = selected_ids
        search["agentAssemblySpec"] = {**_main.copy.deepcopy(spec), "searchId": str(search["id"])}
        current.update({
            "status": _main.AWAITING_AGENT_PLAN, "stage": "agent_plan_running",
            "actionRequired": None, "currentAction": "已按多主题选定内容候选",
            "detail": f"已采用 {len(selected_ids)} 个多主题可靠候选，正在建立时间线。",
            "progressMode": "indeterminate", "etaSeconds": None, "etaMode": "unavailable",
            "updatedAt": _main.now_iso(),
        })
        _main.save_job(current)
    return {
        "artifact": {
            "kind": "multi_topic_evidence_selection", "jobId": ctx.job_id,
            "searchId": str(search["id"]), "matchIds": selected_ids,
            "coverage": _main.copy.deepcopy(spec.get("coverage") or []),
            "message": f"我已为 {len(spec.get('predicateIds') or [])} 个主题选择可靠候选。",
        },
    }


def select_identity(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    if ctx.autonomous:
        search = ctx.snapshot.get("contentSearch") if isinstance(ctx.snapshot.get("contentSearch"), dict) else {}
        if ctx.tool_name == "select_people":
            target = (
                (ctx.snapshot.get("request") or {}).get("contentSearchPersonTarget")
                or ctx.snapshot.get("contentSearchPersonTarget")
                or (search.get("intent") or {}).get("personTarget")
                or {}
            )
            selected = [str(value) for value in target.get("personIds") or [] if str(value)]
            label = "人物"
            selection_source = "persisted_reliable_evidence"
            description = str(arguments.get("description") or "").strip()
            if not selected and description:
                index = _main._load_content_person_index(ctx.snapshot)
                catalog = _main._content_person_catalog(ctx.snapshot, index)
                try:
                    matched, diagnostics = _main._match_person_catalog_by_visual_description(
                        ctx.job_id, ctx.snapshot, f'agent_person_{_main.uuid.uuid4().hex[:12]}',
                        description, catalog,
                        _main.cancel_events.setdefault(ctx.job_id, _main.threading.Event()),
                        person_tracks=[
                            item for item in index.get("personTracks") or []
                            if isinstance(item, dict)
                        ],
                        require_reliable=True,
                    )
                except Exception as error:
                    return ctx.no_result(
                        "identity_verification_unavailable",
                        f"已完成人物识别，但暂时无法可靠核定“{description}”：{str(error)[:180]}",
                        candidate_count=len(catalog), reliable_count=0,
                    )
                reliable_ids = {
                    str(value) for value in diagnostics.get("reliablePersonIds") or []
                    if str(value)
                }
                selected_people = [
                    item for item in matched
                    if str(item.get("id") or "") in reliable_ids
                ]
                if selected_people:
                    selected = [str(item.get("id") or "") for item in selected_people]
                    selection_source = "visual_description_consensus"
                    _main.select_content_person_target(
                        ctx.job_id,
                        _main.PersonTargetRequest(
                            personIds=selected, matchMode="any", activity="appearance",
                        ),
                        display_text=f"根据外观描述选择人物：{description}",
                    )
                else:
                    possible_count = len(diagnostics.get("uncertainPersonIds") or [])
                    return ctx.no_result(
                        "ambiguous_identity" if possible_count else "no_match",
                        (
                            f"已识别画面人物，但“{description}”只有 {possible_count} 个待核对候选，"
                            "没有足够可靠证据可自动选定。"
                            if possible_count else
                            f"已检查画面人物，没有找到可靠符合“{description}”的对象。"
                        ),
                        candidate_count=len(catalog), reliable_count=0,
                    )
        else:
            intent = search.get("intent") if isinstance(search.get("intent"), dict) else {}
            selected = list(dict.fromkeys(
                str(value) for value in [
                    *(search.get("selectedSpeakerRefs") or []),
                    *(intent.get("speakerRefs") or []),
                ] if str(value)
            ))
            label = "说话人"
            selection_source = "persisted_reliable_evidence"
            requested_label = str(arguments.get("label") or "").strip().casefold()
            if not selected and requested_label:
                matching_voices = [
                    voice for voice in _main._public_current_voice_catalog(ctx.snapshot)
                    if requested_label in {
                        str(voice.get("label") or "").strip().casefold(),
                        str(voice.get("speakerRef") or "").strip().casefold(),
                    }
                ]
                if len(matching_voices) == 1:
                    selected = [str(matching_voices[0].get("speakerRef") or "")]
                    selection_source = "explicit_anonymous_label"
            # Speaker discovery deliberately precedes this step.  Do not
            # discard that fresh evidence merely because no selection had
            # existed before the plan started.  A single, high-confidence
            # non-mixed candidate is safe to adopt; anything less certain
            # must become a review checkpoint, never a misleading
            # "no content" terminal result.
            if not selected:
                reliable_voices = []
                for voice in _main._public_current_voice_catalog(ctx.snapshot):
                    speaker_ref = str(voice.get("speakerRef") or "").strip()
                    narration = voice.get("narration") if isinstance(voice.get("narration"), dict) else {}
                    quality = voice.get("quality") if isinstance(voice.get("quality"), dict) else {}
                    if (
                        speaker_ref
                        and not bool(voice.get("requiresReview"))
                        and not bool(quality.get("suspectedMixed"))
                        and str(narration.get("status") or "") in {"candidate", "confirmed"}
                        and float(narration.get("score") or 0.0) >= 0.80
                    ):
                        reliable_voices.append((speaker_ref, narration))
                if len(reliable_voices) == 1:
                    selected = [reliable_voices[0][0]]
                    selection_source = "voice_timeline_heuristics"
        if selected:
            adopted_match_ids: list[str] = []
            if ctx.tool_name == "select_people":
                # Selecting an appearance target synchronously materializes
                # a content-search result from the cached person tracks.
                # That helper normally exposes a human review checkpoint;
                # autonomous review has already resolved the identity, so
                # adopt only its reliable candidates here and immediately
                # return ownership of the job state to the Agent plan.
                with _main.jobs_lock:
                    current = _main.jobs.get(ctx.job_id)
                    current_search = (
                        current.get("contentSearch")
                        if current and isinstance(current.get("contentSearch"), dict)
                        else {}
                    )
                    reliable_candidates = [
                        item for item in current_search.get("candidates") or []
                        if isinstance(item, dict)
                        and item.get("reviewStatus") != "rejected"
                        and float(item.get("end") or 0) > float(item.get("start") or 0)
                        and (
                            item.get("selected") is True
                            or str(item.get("confidenceTier") or "") == "reliable"
                            or (
                                not str(item.get("confidenceTier") or "")
                                and not bool(item.get("requiresReview"))
                            )
                        )
                    ]
                    adopted_match_ids = list(dict.fromkeys(
                        str(item.get("id") or "") for item in reliable_candidates
                        if str(item.get("id") or "")
                    ))[:64]
                    if current and str(current_search.get("id") or "") and adopted_match_ids:
                        current_search["reviewDraft"] = {
                            "schemaVersion": "content-review-draft-v1",
                            "searchId": str(current_search["id"]),
                            "selectedMatchIds": adopted_match_ids,
                            "orderedMatchIds": adopted_match_ids,
                            "source": "agent_autonomous_identity_selection",
                            "updatedAt": _main.now_iso(),
                        }
                        current_search["defaultSelectedIds"] = adopted_match_ids
                        current.update({
                            "status": _main.AWAITING_AGENT_PLAN,
                            "stage": "agent_plan_running",
                            "actionRequired": None,
                            "currentAction": "已自动采用人物出镜范围",
                            "detail": (
                                f"已采用 {len(adopted_match_ids)} 个可靠出镜片段，"
                                "正在建立可审阅时间线。"
                            ),
                            "progressMode": "indeterminate",
                            "etaSeconds": None,
                            "etaMode": "unavailable",
                            "updatedAt": _main.now_iso(),
                        })
                        _main.save_job(current)
            if ctx.tool_name == "select_speakers" and selection_source in {
                "voice_timeline_heuristics", "explicit_anonymous_label",
            }:
                # Persist the automatic identity decision through the
                # same path as the review panel.  The following semantic
                # search then inherits the selected speaker scope.
                _main.select_current_voices(
                    ctx.job_id,
                    _main.CurrentVoiceSelectionRequest(
                        speakerRefs=selected,
                        mode=str(arguments.get("mode") or "include"),
                        query="",
                    ),
                )
            return {"artifact": {
                "kind": "identity_selection", "identityKind": ctx.tool_name,
                "selectedIds": selected, "source": selection_source,
                **({"matchIds": adopted_match_ids} if adopted_match_ids else {}),
                "message": (
                    f"我已根据多帧可见外观证据采用符合“{str(arguments.get('description') or '')}”的{label}。"
                    if selection_source == "visual_description_consensus"
                    else
                    f"我已采用用户指定的{str(arguments.get('label') or label)}。"
                    if selection_source == "explicit_anonymous_label"
                    else f"我已根据刚完成的说话人证据自动采用可靠{label}范围。"
                    if selection_source == "voice_timeline_heuristics"
                    else f"我已采用现有可靠{label}范围。"
                ),
            }}
        return {
            "actionRequired": True,
            "action": "identity_selection",
            "message": (
                f"已完成{label}识别，但存在多个候选或证据不足。"
                f"请在{label}面板试听或核对后选择要保留的对象，再继续。"
            ),
        }
    return {
        "actionRequired": True,
        "action": "identity_selection",
        "message": "请在人物或说话人审核面板中完成结构化选择。",
    }


def review_content_evidence(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    if ctx.autonomous:
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            search = current.get("contentSearch") if current and isinstance(current.get("contentSearch"), dict) else {}
            candidates = [
                item for item in search.get("candidates") or []
                if isinstance(item, dict) and item.get("reviewStatus") != "rejected"
                and float(item.get("end") or 0) > float(item.get("start") or 0)
            ]
            allowed = {str(item.get("id") or "") for item in candidates if str(item.get("id") or "")}
            if any(item.get("matchType") == "voice_identity" for item in candidates):
                contract = _main.build_contract(search)
                review = search.get("reviewDraft") or {}
                reviewed_ids = set(review.get("orderedMatchIds") or review.get("selectedMatchIds") or [])
                selected_voice = [item for item in candidates if item.get("id") in reviewed_ids]
                if not selected_voice or any(not _main.verification_current(item, contract) for item in selected_voice):
                    return {"actionRequired": True, "action": "content_evidence_review",
                            "message": "参考声纹仅提供疑似匹配。请试听并确认要保留的片段，再继续生成样片。"}
            reliable_candidates = [
                item for item in candidates
                if (
                    item.get("selected") is True
                    or str(item.get("confidenceTier") or "") == "reliable"
                    or (
                        not str(item.get("confidenceTier") or "")
                        and not bool(item.get("requiresReview"))
                    )
                )
            ]
            reliable_ids = {
                str(item.get("id") or "") for item in reliable_candidates
                if str(item.get("id") or "")
            }
            # Required branch coverage is evaluated against the exact
            # reliable set that autonomous mode is allowed to consume.
            # A merely "possible" match must not make a required category
            # look covered and then disappear from the resulting timeline.
            assembly_spec = _main._content_assembly_spec(
                search, allowed_match_ids=reliable_ids,
            )
            existing = search.get("reviewDraft") if isinstance(search.get("reviewDraft"), dict) else {}
            confirmed_ids = [
                str(value) for value in search.get("confirmedMatchIds") or []
                if str(value) in allowed
            ]
            existing_ids = [
                str(value)
                for value in (
                    existing.get("orderedMatchIds")
                    or existing.get("selectedMatchIds")
                    or []
                )
                if str(value) in allowed
            ]
            existing_source = str(existing.get("source") or "")
            # Search workers may preselect a single representative match
            # for UI focus. In autonomous review that must not become the
            # whole edit when the user asked for “所有/全部” content. Only
            # user-confirmed selections constrain the timeline; otherwise
            # consume the full reliable set.
            manual_review_sources = {"user", "manual", "content_review_user", "user_review"}
            selected_ids = confirmed_ids or (
                existing_ids if existing_source in manual_review_sources else []
            )
            selection_policy = str(arguments.get("selectionPolicy") or "all_reliable")
            if selection_policy == "unique_or_review":
                # UI focus/preselection alone does not establish a reliable
                # semantic starting point.
                reliable_candidates = [item for item in candidates if (
                    str(item.get("confidenceTier") or "") == "reliable"
                    or (not item.get("confidenceTier") and not item.get("requiresReview"))
                )]
                reliable_ids = {str(item["id"]) for item in reliable_candidates if item.get("id")}
                reviewed_ids = [value for value in selected_ids if value in reliable_ids]
                if len(reviewed_ids) == 1:
                    selected_ids = reviewed_ids
                elif len(reliable_ids) == 1:
                    selected_ids = list(reliable_ids)
                elif len(reliable_ids) > 1:
                    return {
                        "actionRequired": True,
                        "action": "structured_review",
                        "message": (
                            f"找到 {len(reliable_ids)} 个可靠的起剪位置。"
                            "请在候选面板预览并仅保留一个作为新的开始位置。"
                        ),
                        "candidateCount": len(candidates),
                        "reliableCandidateCount": len(reliable_candidates),
                    }
                else:
                    return ctx.no_result(
                        "no_match",
                        "没有找到可靠的起剪位置，未修改当前时间线。",
                        query=str(arguments.get("query") or search.get("instruction") or ""),
                        candidate_count=len(candidates), reliable_count=0,
                    )
            if selection_policy != "unique_or_review" and len(assembly_spec.get("predicateIds") or []) >= 2:
                if assembly_spec.get("missingPredicates"):
                    return ctx.no_result(
                        "missing_required_categories",
                        "自动编排缺少必要类别的可靠候选："
                        + "、".join(str(value) for value in assembly_spec["missingPredicates"]),
                        query=str(arguments.get("query") or search.get("instruction") or ""),
                        candidate_count=len(candidates), reliable_count=len(reliable_candidates),
                    )
                selected_ids = [
                    str(value) for value in assembly_spec.get("selectedMatchIds") or []
                    if str(value) in reliable_ids
                ]
            if not selected_ids:
                selected_ids = [
                    str(item.get("id") or "") for item in candidates
                    if str(item.get("id") or "") in reliable_ids
                ]
            selected_ids = list(dict.fromkeys(value for value in selected_ids if value))[:64]
            if not current or not str(search.get("id") or "") or not selected_ids:
                return ctx.no_result(
                    "no_match",
                    "内容检索没有生成可用于自动编排的可靠候选。",
                    query=str(arguments.get("query") or search.get("instruction") or ""),
                    candidate_count=len(candidates), reliable_count=len(reliable_candidates),
                )
            selected_lookup = {
                str(item.get("id") or ""): item for item in candidates
                if str(item.get("id") or "") in set(selected_ids)
            }
            coverage_seconds = sum(
                max(0.0, float(item.get("end") or 0) - float(item.get("start") or 0))
                for item in selected_lookup.values()
            )
            search["reviewDraft"] = {
                "schemaVersion": "content-review-draft-v1",
                "searchId": str(search["id"]),
                "selectedMatchIds": selected_ids,
                "orderedMatchIds": selected_ids,
                "source": "agent_autonomous_review",
                "updatedAt": _main.now_iso(),
            }
            search["defaultSelectedIds"] = selected_ids
            if len(assembly_spec.get("predicateIds") or []) >= 2:
                search["agentAssemblySpec"] = _main.copy.deepcopy(assembly_spec)
            # The search worker deliberately stops at a review checkpoint.
            # Autonomous review has resolved it, so the outer job must
            # return to Agent-owned execution instead of advertising a
            # user action while the timeline is already being applied.
            current.update({
                "status": _main.AWAITING_AGENT_PLAN,
                "stage": "agent_plan_running",
                "actionRequired": None,
                "currentAction": "已自动选定内容候选",
                "detail": f"已采用 {len(selected_ids)} 个可靠候选，正在建立可审阅时间线。",
                "progressMode": "indeterminate",
                "etaSeconds": None,
                "etaMode": "unavailable",
            })
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
        return {
            "artifact": {
                "kind": "content_evidence_selection", "jobId": ctx.job_id,
                "searchId": str(search["id"]), "matchIds": selected_ids,
                "candidateCount": len(candidates),
                "reliableCandidateCount": len(reliable_candidates),
                "coverageSeconds": round(coverage_seconds, 3),
                "message": f"我已自动采用 {len(selected_ids)} 个有效内容候选。",
            },
        }
    raise RuntimeError(f"未注册的 Agent 核心工具：{ctx.tool_name}")


def content_workflow(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    target_workflow = {
        "analyze_highlights": "highlight",
        "search_content": "content_search",
        "discover_people": "person_edit",
        "discover_speakers": "speaker_edit",
    }.get(ctx.tool_name)
    speaker_scoped_search = bool(
        ctx.tool_name == "search_content"
        and isinstance(ctx.snapshot.get("contentSearch"), dict)
        and _main._is_voice_candidate_search(ctx.snapshot.get("contentSearch") or {})
        and any(
            str(value) for value in ((ctx.snapshot.get("contentSearch") or {}).get("intent") or {}).get("speakerRefs") or []
        )
    )
    inherited_scope = (ctx.snapshot.get("request") or {}).get("sourceScope") or {}
    scope_kind = str(arguments.get("sourceScopeKind") or inherited_scope.get("kind") or (ctx.snapshot.get("request") or {}).get("sourceScopeKind") or "all")
    scope_start = arguments.get("sourceScopeStart", inherited_scope.get("start"))
    scope_end = arguments.get("sourceScopeEnd", inherited_scope.get("end"))
    scope_changed = "sourceScopeKind" in arguments and (
        scope_kind != str(inherited_scope.get("kind") or "all")
        or scope_start != inherited_scope.get("start") or scope_end != inherited_scope.get("end")
    )
    if target_workflow and (scope_changed or (_main.workflow_kind_for_job(ctx.snapshot) != target_workflow and not speaker_scoped_search)):
        instruction = str(
            arguments.get("query") or arguments.get("instruction") or {
                "highlight": "从整个源视频生成高光",
                "content_search": "查找与目标匹配的内容",
                "person_edit": "提取所选画面人物的所有出镜片段",
                "speaker_edit": "识别本视频中的说话人",
            }[target_workflow]
        )[:500]
        handoff = _main.create_same_source_task_job(
            ctx.job_id,
            _main.SameSourceTaskRequest(
                workflowKind=target_workflow, instruction=instruction,
                targetSeconds=arguments.get("targetSeconds"),
                expectedSpeakerCount=arguments.get("expectedSpeakers"),
                sourceScopeKind=scope_kind,
                sourceScopeStart=scope_start,
                sourceScopeEnd=scope_end,
                # The Agent plan owns proposal, application and review render.
                # Starting the legacy auto-composer here creates a second
                # concurrent pipeline that can keep rendering after the plan
                # is already marked complete and can surface an obsolete
                # ``review_highlights`` confirmation gate.
                autoCompose=False,
            ),
        )
        child = handoff.get("job") if isinstance(handoff.get("job"), dict) else {}
        child_id = str(child.get("id") or "")
        if not child_id:
            raise RuntimeError("同源 Agent 任务创建失败")
        # Moving to a same-source task is a durable Agent handoff.  Keep the
        # workspace id and active plan available on the child before its
        # background analysis may finish, so a reload cannot strand the user
        # on the source task.
        _main.agent_platform.bind_workspace_to_job(
            workspace_id=str(ctx.workspace["id"]), job_id=child_id, source_job_id=ctx.job_id,
        )
        with _main.jobs_lock:
            source = _main.jobs.get(ctx.job_id)
            if source and str(source.get("status") or "") == _main.AWAITING_AGENT_PLAN:
                source_agent = _main.copy.deepcopy(source.get("agent")) if isinstance(source.get("agent"), dict) else {}
                source_agent.update({
                    "workspaceStatus": "handed_off",
                    "status": "handed_off",
                    "currentStepId": "",
                    "currentStepTitle": "",
                    "currentStepTool": "",
                    "currentStepStatus": "",
                })
                source.update({
                    "status": "completed", "stage": "agent_handed_off",
                    "progress": 1.0, "stageProgress": 1.0,
                    "detail": "已转入同源子任务继续执行",
                    "currentAction": "已交接给 Agent 子任务",
                    "progressMode": "completed", "etaMode": "completed",
                    "etaSeconds": None, "updatedAt": _main.now_iso(),
                    "agent": source_agent,
                    "agentHandoffJobId": child_id,
                    "agentHandoffWorkflowKind": target_workflow,
                })
                _main.save_job(source)
            future = _main.analysis_futures.get(child_id)
            current_child = _main.public_job(_main.jobs[child_id]) if _main.jobs.get(child_id) else child
        if target_workflow == "speaker_edit" and future is None:
            future = _main.submit_workflow_analysis(child_id, "speaker_discovery", {
                "expectedSpeakers": arguments.get("expectedSpeakers"),
            })
        result = {
            "operationId": f'{child_id}:{ctx.tool_name}', "accepted": True,
            "jobId": child_id, "job": current_child, "handoff": handoff.get("handoff"),
        }
        if future is not None:
            result["future"] = future
            result["cancel"] = lambda: _main.cancel_job(child_id)
        return result
    if ctx.tool_name == "search_content" and isinstance(ctx.snapshot.get("contentSearch"), dict) and ctx.snapshot["contentSearch"].get("id"):
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise RuntimeError("内容搜索缺少 query")
        if str(ctx.snapshot.get("status") or "") in {"running", "queued", "cancelling"}:
            raise RuntimeError("当前素材已有后台操作，不能重复提交")
        existing_search = ctx.snapshot.get("contentSearch") if isinstance(ctx.snapshot.get("contentSearch"), dict) else {}
        existing_intent = existing_search.get("intent") if isinstance(existing_search.get("intent"), dict) else {}
        speaker_refs = [str(value) for value in existing_intent.get("speakerRefs") or [] if str(value)]
        selection_mode = str(existing_intent.get("voiceSelectionMode") or "include")
        if _main._is_voice_candidate_search(existing_search) and speaker_refs:
            # Continue inside the selected speaker scope.  Routing the next
            # semantic query through generic chat could silently create a new
            # content-exploration task and lose the identity constraint.
            result_job = _main._apply_current_speakers_search(
                ctx.job_id, speaker_refs, query, selection_mode,
            )
            with _main.jobs_lock:
                current = _main.jobs.get(ctx.job_id)
                if current:
                    current.update({
                        "status": _main.AWAITING_AGENT_PLAN,
                        "stage": "agent_plan_running",
                        "actionRequired": None,
                        "currentAction": "已在目标说话人范围内筛选内容",
                        "detail": "已按目标说话人和语义要求筛选发言候选。",
                        "progressMode": "indeterminate",
                        "etaSeconds": None,
                        "etaMode": "unavailable",
                        "updatedAt": _main.now_iso(),
                    })
                    _main.save_job(current)
            return {
                "artifact": {
                    "kind": "speaker_scoped_content_search", "jobId": ctx.job_id,
                    "speakerRefs": speaker_refs, "query": query,
                    "candidateCount": len((result_job.get("contentSearch") or {}).get("candidates") or []),
                    "message": "我已在选定说话人的发言中完成语义筛选。",
                },
            }
        queued = _main.queue_content_followup(ctx.job_id, query, _main.ChatRequest(text=query))
        with _main.jobs_lock:
            future = _main.analysis_futures.get(ctx.job_id)
        result = {
            "operationId": f'{ctx.job_id}:content_followup_search',
            "operation": "content_followup_search", "accepted": True,
            "job": queued.get("job"),
            "artifact": {
                "kind": "content_search_result", "jobId": ctx.job_id,
                "query": query[:500],
                "message": "内容候选将在当前任务中展示；本步骤不会生成视频样片。",
            },
        }
        if future is not None:
            result["future"] = future
            result["cancel"] = lambda: _main.cancel_job(ctx.job_id)
        return result
    operation = {
        "analyze_highlights": "highlight_analysis",
        "discover_people": "person_discovery",
        "discover_speakers": "speaker_discovery",
    }.get(ctx.tool_name)
    if ctx.tool_name == "search_content":
        operation = "content_initial_search"
    if not operation:
        raise RuntimeError(f'未注册的 Agent 核心工具：{ctx.tool_name}')
    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        if not current:
            raise RuntimeError("素材任务不存在")
        if str(current.get("status") or "") in {"running", "queued", "cancelling"}:
            raise RuntimeError("当前素材已有后台操作，不能重复提交")
        job_request = current.setdefault("request", {})
        if ctx.tool_name == "analyze_highlights":
            # Agent-owned plans have explicit timeline and review-render
            # steps. Keep the legacy analyzer evidence-only to avoid launching
            # a duplicate background auto-composition pipeline.
            current["autoCompose"] = False
            instruction = str(arguments.get("instruction") or arguments.get("focus") or "").strip()
            if instruction:
                job_request["theme"] = instruction[:500]
            if arguments.get("targetSeconds") is not None:
                target_seconds = max(4.0, float(arguments["targetSeconds"]))
                job_request["targetSeconds"] = target_seconds
                current["targetSeconds"] = target_seconds
            current["brief"] = _main._confirmed_brief_from_request(job_request)
            current["editingIntent"] = _main.compile_editing_intent(current["brief"], job_request)
        elif ctx.tool_name == "search_content":
            query = str(arguments.get("query") or "").strip()
            if not query:
                raise RuntimeError("内容搜索缺少 query")
            job_request["contentInstruction"] = query[:500]
            job_request["theme"] = query[:500]
            if isinstance(current.get("brief"), dict):
                current["brief"]["narrativeGoal"] = query[:500]
                current["brief"]["focus"] = [query[:500]]
        elif ctx.tool_name == "discover_speakers" and arguments.get("expectedSpeakers") is not None:
            job_request["expectedSpeakerCount"] = max(0, min(32, int(arguments["expectedSpeakers"])))
        current["updatedAt"] = _main.now_iso()
        _main.save_job(current)
    future = _main.submit_workflow_analysis(ctx.job_id, operation, _main.copy.deepcopy(arguments))
    return {
        "operationId": f'{ctx.job_id}:{operation}', "operation": operation,
        "accepted": True, "future": future, "cancel": lambda: _main.cancel_job(ctx.job_id),
        **({
            "artifact": {
                "kind": "content_search_result", "jobId": ctx.job_id,
                "query": str(arguments.get("query") or "")[:500],
                "message": "内容候选将在当前任务中展示；本步骤不会生成视频样片。",
            },
        } if ctx.tool_name == "search_content" else {}),
    }


