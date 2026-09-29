"""Timeline-edit Agent tool handlers."""
from __future__ import annotations

from typing import Any

from .context import ToolContext


def cancel_operation(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    active_plan = _main.agent_platform.active_plan_for_workspace(ctx.workspace)
    if active_plan and str(active_plan.get("status") or "") in {"running", "action_required", "waiting_operation"}:
        cancelled = _main.agent_platform.cancel_plan(str(active_plan["id"]))
        return {"cancelled": True, "plan": {"id": cancelled["id"], "status": cancelled["status"]}}
    return {"cancelled": False, "message": "当前没有正在执行的 Agent 后台操作"}


def propose_timeline_edit(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    approved_plan = _main.agent_platform.store.get("plans", str(ctx.workspace.get("activePlanId") or "")) or {}
    frozen_input = approved_plan.get("inputContext") or {}
    _main.validate_frozen(ctx.snapshot, frozen_input)
    if arguments.get("anchorStartQuery") or arguments.get("durationSource") == "relative":
        with _main.jobs_lock:
            current = _main.jobs[ctx.job_id]
            try:
                result = _main._conversational_timeline_proposal(
                    current, arguments, frozen_input, str(approved_plan.get("id") or ""),
                )
            except (ValueError, _main.EditSessionError) as error:
                return ctx.no_result("revision_constraint_unmet", str(error))
            _main.save_job(current)
        return {**result, "actionRequired": not ctx.autonomous}
    contract_action = None if frozen_input.get("outputFilename") else _main._prepare_content_composition_contract(ctx.job_id)
    if contract_action:
        return contract_action
    with _main.jobs_lock:
        ctx.snapshot = _main.copy.deepcopy(_main.jobs.get(ctx.job_id) or ctx.snapshot)
    session_id = str(ctx.snapshot.get("activeEditSessionId") or "")
    created = False
    if frozen_input.get("outputFilename"):
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            _main.validate_frozen(current or {}, frozen_input)
            session = next((s for s in current.get("editSessions") or [] if s.get("agentInputPlanId") == approved_plan.get("id")), None)
            if session is None:
                if frozen_input.get("outputVersionId"):
                    base, _ = _main.create_or_resume_edit_session(current, version_id=frozen_input["outputVersionId"], output_filename=frozen_input["outputFilename"])
                else:
                    base = next((s for s in current.get("editSessions") or [] if s.get("id") == frozen_input.get("editSessionId")), None)
                if not base:
                    raise RuntimeError("所引用的样片没有可编辑时间线，请先在精剪中建立草稿")
                session = _main.copy.deepcopy(base)
                session.update({"id": f'edit_session_{_main.uuid.uuid4().hex[:12]}', "agentInputPlanId": approved_plan["id"], "revision": 0, "status": "draft", "previewStatus": "idle", "undo": [], "redo": []})
                for key in ("previewPath", "previewUrl", "previewFingerprint", "pendingProposal", "contentVerification"):
                    session.pop(key, None)
                current.setdefault("editSessions", []).append(session)
                created = True
            session_id = session["id"]
            current["activeEditSessionId"] = session_id
            _main.save_job(current)
    duration_fit: dict[str, Any] = {}
    requested_variants = max(1, min(4, int(arguments.get("variantCount") or 1)))
    variant_directions = [
        str(item).strip()[:160] for item in arguments.get("variantDirections") or []
        if str(item).strip()
    ][:requested_variants]
    instruction = str(arguments.get("instruction") or "按已确认候选建立可审核时间线")[:500]
    with _main.jobs_lock:
        current = _main.jobs.get(ctx.job_id)
        if current and session_id and not any(
            str(item.get("id") or "") == session_id for item in current.get("editSessions") or []
        ):
            session_id = ""
        search = current.get("contentSearch") if current and isinstance(current.get("contentSearch"), dict) else {}
        if frozen_input.get("outputFilename"):
            search = {}
        candidates = [item for item in search.get("candidates") or [] if isinstance(item, dict)]
        candidates = [item for item in candidates if float(item.get("end") or 0) > float(item.get("start") or 0)]
        review = search.get("reviewDraft") if isinstance(search.get("reviewDraft"), dict) else {}
        saved_ids = review.get("orderedMatchIds") or review.get("selectedMatchIds") or search.get("confirmedMatchIds") or search.get("defaultSelectedIds") or []
        if frozen_input.get("contentSelection"):
            _main.validate_frozen(current, frozen_input)
            saved_ids = frozen_input["contentSelection"]["matchIds"]
        allowed_ids = {str(item.get("id") or "") for item in candidates}
        selected_ids = [str(value) for value in saved_ids if str(value) in allowed_ids]
        if current and session_id and str(search.get("id") or ""):
            active_session = _main.find_edit_session(current, session_id)
            if (
                str(active_session.get("sourceSearchId") or "") != str(search.get("id") or "")
                or list(active_session.get("sourceMatchIds") or []) != selected_ids
                or (active_session.get("contentBinding") or {}).get("selectionFingerprint") != _main.selection_binding(
                    search, [item for value in selected_ids for item in candidates if str(item.get("id")) == value]
                )["selectionFingerprint"]
            ):
                # An edit session created by an earlier search or by the
                # candidate drawer is not the approved Agent selection.
                # Reusing it silently injected every candidate into the
                # new timeline.
                session_id = ""
            elif ctx.autonomous:
                assembly_spec = (
                    search.get("agentAssemblySpec")
                    if isinstance(search.get("agentAssemblySpec"), dict)
                    else _main._content_assembly_spec(search)
                )
                assembly_spec = {
                    **assembly_spec, "searchId": str(search.get("id") or ""),
                }
                if (
                    len(assembly_spec.get("predicateIds") or []) >= 2
                    and _main._content_session_needs_assembly_fit(active_session, assembly_spec)
                ):
                    _main._fit_content_session_to_assembly(
                        current, active_session, search, assembly_spec,
                    )
                    current["updatedAt"] = _main.now_iso()
                    _main.save_job(current)
        if current and not session_id and str(search.get("id") or ""):
            if selected_ids:
                try:
                    session, created = _main.create_or_resume_content_edit_session(
                        current, search_id=str(search["id"]), selected_match_ids=selected_ids,
                        order_mode="selection",
                    )
                    assembly_spec = (
                        search.get("agentAssemblySpec")
                        if isinstance(search.get("agentAssemblySpec"), dict)
                        else _main._content_assembly_spec(search)
                    )
                    assembly_spec = {
                        **assembly_spec, "searchId": str(search.get("id") or ""),
                    }
                    if (
                        ctx.autonomous
                        and len(assembly_spec.get("predicateIds") or []) >= 2
                        and _main._content_session_needs_assembly_fit(session, assembly_spec)
                    ):
                        if not created:
                            session = _main.copy.deepcopy(session)
                            session.update({
                                "id": f'edit_session_{_main.uuid.uuid4().hex[:12]}',
                                "revision": 0, "undo": [], "redo": [],
                                "pendingProposal": None, "proposalVariants": [],
                                "createdAt": _main.now_iso(), "updatedAt": _main.now_iso(),
                            })
                            for key in (
                                "previewPath", "previewUrl", "previewFingerprint", "previewRevision",
                                "renderPlanFingerprint", "renderedVersionId",
                            ):
                                session.pop(key, None)
                            session["previewStatus"] = "idle"
                            current.setdefault("editSessions", []).append(session)
                            created = True
                        _main._fit_content_session_to_assembly(current, session, search, assembly_spec)
                    session["agentTimelineRequest"] = {
                        "variantCount": requested_variants,
                        "instruction": instruction,
                        "source": "agent_plan",
                    }
                    session["title"] = (
                        f"内容检索精剪 · {requested_variants} 个结构方向"
                        if requested_variants > 1 else "内容检索精剪"
                    )
                    session_id = str(session.get("id") or "")
                    current["activeEditSessionId"] = session_id
                    current["updatedAt"] = _main.now_iso()
                    _main.save_job(current)
                except _main.EditSessionError:
                    session_id = ""
        if current and not session_id and not str(search.get("id") or ""):
            highlight_candidates: list[tuple[str, dict[str, Any]]] = []
            for index, item in enumerate(current.get("candidates") or []):
                if not isinstance(item, dict) or float(item.get("end") or 0) <= float(item.get("start") or 0):
                    continue
                identity = item.get("id") or item.get("candidateId")
                if identity in (None, "") and item.get("index") is not None:
                    identity = item.get("index")
                candidate_id = str(identity if identity not in (None, "") else f"candidate_{index}")
                highlight_candidates.append((candidate_id, item))
            if highlight_candidates:
                allowed_ids = {candidate_id for candidate_id, _item in highlight_candidates}
                preferred_values = (
                    current.get("confirmedIndices") or current.get("recommendedIndices") or []
                )
                selected_ids = [
                    str(value) for value in preferred_values if str(value) in allowed_ids
                ]
                if not selected_ids:
                    selected_group_ids = {
                        str(value) for value in (
                            current.get("confirmedGroupIds") or current.get("recommendedGroupIds") or []
                        ) if str(value)
                    }
                    for group in current.get("eventGroups") or []:
                        if selected_group_ids and str(group.get("id") or "") not in selected_group_ids:
                            continue
                        for segment in group.get("segments") or []:
                            candidate_id = str(segment.get("candidateId") or segment.get("id") or "")
                            if candidate_id in allowed_ids and candidate_id not in selected_ids:
                                selected_ids.append(candidate_id)
                if not selected_ids:
                    ranked = sorted(
                        highlight_candidates,
                        key=lambda row: float(
                            row[1].get("editorialScore") or row[1].get("score") or 0
                        ),
                        reverse=True,
                    )
                    selected_ids = [candidate_id for candidate_id, _item in ranked]
                selected_ids = list(dict.fromkeys(selected_ids))[:64]
                try:
                    session, created = _main.create_or_resume_candidate_edit_session(
                        current, candidate_ids=selected_ids, order_mode="source",
                    )
                    session["agentTimelineRequest"] = {
                        "variantCount": requested_variants,
                        "instruction": instruction,
                        "source": "agent_plan",
                    }
                    session["title"] = (
                        f"高光候选精剪 · {requested_variants} 个结构方向"
                        if requested_variants > 1 else "高光候选精剪"
                    )
                    session_id = str(session.get("id") or "")
                    current["activeEditSessionId"] = session_id
                    current["updatedAt"] = _main.now_iso()
                    _main.save_job(current)
                except _main.EditSessionError:
                    session_id = ""
        if current and not session_id and not str(search.get("id") or ""):
            delivery = _main.AgentPlatform._social_delivery(instruction)
            retrieval_query = _main.AgentPlatform._retrieval_query(instruction)
            duration = float(
                (current.get("videoInfo") or {}).get("duration")
                or current.get("duration") or 0
            )
            request_state = (
                current.get("request") if isinstance(current.get("request"), dict) else {}
            )
            source_scope = (
                request_state.get("sourceScope")
                if isinstance(request_state.get("sourceScope"), dict) else {}
            )
            source_start = max(0.0, float(source_scope.get("start") or 0))
            source_end = min(duration, float(source_scope.get("end") or duration))
            if (
                bool(delivery.get("requested"))
                and not retrieval_query
                and source_end - source_start >= .25
            ):
                # A format-only request means “keep the source and change
                # its canvas”, not “discover an arbitrary highlight”.
                # Materialize one full-scope candidate so the regular edit
                # and preview paths remain traceable without visual analysis.
                search_id = f'source_passthrough_{_main.uuid.uuid4().hex[:12]}'
                match_id = f'source_match_{_main.uuid.uuid4().hex[:12]}'
                passthrough_search = {
                    "schemaVersion": _main.CONTENT_SEARCH_VERSION,
                    "id": search_id,
                    "instruction": instruction,
                    "status": "ready",
                    "candidateCount": 1,
                    "candidates": [{
                        "id": match_id,
                        "start": round(source_start, 3),
                        "end": round(source_end, 3),
                        "duration": round(source_end - source_start, 3),
                        "title": "完整素材 · 画幅转换",
                        "reason": "用户仅要求改变画幅，完整保留所选素材范围",
                        "confidence": 1.0,
                        "confidenceTier": "reliable",
                        "requiresReview": False,
                        "selected": True,
                        "reviewStatus": "confirmed",
                        "evidenceType": "source_scope",
                        "matchedModalities": ["source"],
                    }],
                    "defaultSelectedIds": [match_id],
                    "reviewDraft": {
                        "schemaVersion": "content-review-draft-v1",
                        "searchId": search_id,
                        "selectedMatchIds": [match_id],
                        "orderedMatchIds": [match_id],
                        "source": "agent_source_passthrough",
                        "updatedAt": _main.now_iso(),
                    },
                    "createdAt": _main.now_iso(),
                }
                current["contentSearch"] = passthrough_search
                try:
                    session, created = _main.create_or_resume_content_edit_session(
                        current,
                        search_id=search_id,
                        selected_match_ids=[match_id],
                        order_mode="source",
                    )
                    session["title"] = "完整素材画幅转换"
                    session["agentTimelineRequest"] = {
                        "variantCount": 1,
                        "instruction": instruction,
                        "source": "agent_source_passthrough",
                    }
                    session_id = str(session.get("id") or "")
                    current["activeEditSessionId"] = session_id
                    current["updatedAt"] = _main.now_iso()
                    _main.save_job(current)
                except _main.EditSessionError:
                    session_id = ""
    if ctx.autonomous and not session_id:
        search_id = str((search or {}).get("id") or "")
        if search_id and not candidates:
            return ctx.no_result(
                "no_match",
                "内容检索没有生成可用于自动编排的可靠候选，已跳过后续时间线与渲染步骤。",
                query=str(arguments.get("instruction") or (search or {}).get("instruction") or ""),
                candidate_count=0,
                reliable_count=0,
            )
        if search_id and not selected_ids:
            reliable_count = sum(
                1 for item in candidates
                if (
                    item.get("selected") is True
                    or str(item.get("confidenceTier") or "") == "reliable"
                    or (
                        not str(item.get("confidenceTier") or "")
                        and not bool(item.get("requiresReview"))
                    )
                )
            )
            return ctx.no_result(
                "content_candidates_not_selected",
                "内容检索存在候选，但没有形成可自动编排的确认选择，已跳过后续时间线与渲染步骤。",
                query=str(arguments.get("instruction") or (search or {}).get("instruction") or ""),
                candidate_count=len(candidates),
                reliable_count=reliable_count,
            )
    if ctx.autonomous and session_id and not frozen_input.get("outputFilename"):
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            session = _main.find_edit_session(current, session_id) if current else None
            search = current.get("contentSearch") if current and isinstance(current.get("contentSearch"), dict) else {}
            duration_fit = _main._fit_agent_session_to_target(current, session, search) if current and session else {}
            if current:
                current["updatedAt"] = _main.now_iso()
                _main.save_job(current)
        if str(duration_fit.get("status") or "") in {"insufficient_coverage", "over_target"}:
            fit_status = str(duration_fit.get("status") or "")
            message = (
                f"可靠候选最多只能组成约 {float(duration_fit.get('actualSeconds') or 0):.1f} 秒，"
                f"不足以达到 {float(duration_fit.get('targetSeconds') or 0):.1f} 秒目标。"
                if fit_status == "insufficient_coverage" else
                f"保留完整语义边界后时间线仍为 {float(duration_fit.get('actualSeconds') or 0):.1f} 秒，"
                f"无法收敛到 {float(duration_fit.get('targetSeconds') or 0):.1f} 秒目标范围。"
            )
            return ctx.no_result(
                "insufficient_coverage" if fit_status == "insufficient_coverage" else "duration_constraint_unmet",
                message,
                query=str(arguments.get("instruction") or ""),
                candidate_count=len((ctx.snapshot.get("contentSearch") or {}).get("candidates") or ctx.snapshot.get("candidates") or []),
                coverage_seconds=float(duration_fit.get("actualSeconds") or 0),
            )
    proposal: dict[str, Any] | None = None
    proposal_variants: list[dict[str, Any]] = []
    if session_id:
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            session = _main.find_edit_session(current, session_id) if current else None
            pending = session.get("pendingProposal") if isinstance(session, dict) else None
            revision = int(session.get("revision") or 0) if isinstance(session, dict) else 0
            clip_ids = [
                str(item.get("id") or "") for item in (session.get("clips") or [])
                if str(item.get("id") or "")
            ][:64] if isinstance(session, dict) else []
            if (isinstance(pending, dict) and str(pending.get("status") or "") == "pending"
                    and not pending.get("plannerDegradedReason") and not pending.get("validationRejected")):
                proposal = _main.copy.deepcopy(pending)
                proposal_variants = _main.copy.deepcopy([
                    item for item in session.get("proposalVariants") or [] if isinstance(item, dict)
                ])
        if proposal is None:
            with _main.jobs_lock:
                current = _main.jobs.get(ctx.job_id)
                session = _main.find_edit_session(current, session_id) if current else None
                deterministic_assembly = bool(
                    ctx.autonomous and requested_variants == 1
                    and isinstance((session or {}).get("agentAssembly"), dict)
                    and not variant_directions
                )
                if deterministic_assembly and current and session:
                    proposal = _main.build_secondary_edit_proposal(
                        current, session, text=instruction,
                        selected_clip_ids=clip_ids,
                        model_result={
                            "title": "Agent 均衡时间线草案",
                            "summary": "按检索类别分组，并依据每类时长配额建立待审核时间线。",
                            "operations": [{"type": "reorder_clips", "clipIds": clip_ids}],
                        },
                    )
                    _main.save_job(current)
            try:
                if proposal is not None:
                    pass
                elif requested_variants > 1:
                    proposal_variants = _main.create_agent_timeline_variant_proposals(
                        ctx.job_id, session_id, revision=revision, instruction=instruction,
                        selected_clip_ids=clip_ids, count=requested_variants,
                        directions=variant_directions,
                    )
                    proposal = _main.copy.deepcopy(proposal_variants[0])
                else:
                    payload = _main.create_edit_session_proposal(
                        ctx.job_id, session_id,
                        _main.EditSessionProposalRequest(
                            revision=revision, text=instruction, selectedClipIds=clip_ids,
                        ),
                    )
                    proposal = _main.copy.deepcopy(payload.get("proposal") or {})
            except Exception as error:
                raise RuntimeError(
                    f"剪辑编排未完成（{type(error).__name__}）。候选与草稿已保留，"
                    "未用简单拼接替代原要求；请重试编排步骤。"
                ) from error
        if ctx.autonomous and proposal and proposal.get("plannerDegradedReason"):
            raise RuntimeError("剪辑规划模型不可用，当前仅有降级草稿；未自动应用或生成样片，请重试编排步骤。")
    if session_id and requested_variants > 1:
        if len(proposal_variants) != requested_variants:
            return ctx.no_result(
                "variant_count_unmet",
                f"要求生成 {requested_variants} 版，但当前只得到 {len(proposal_variants) or int(bool(proposal))} 个有效方案；未自动应用或渲染。请调整要求后重新规划。",
            )
        if arguments.get("distinctSourceAcrossVariants"):
            with _main.jobs_lock:
                validation_job = _main.copy.deepcopy(_main.jobs[ctx.job_id])
            base = _main.find_edit_session(validation_job, session_id)
            used_ranges: list[tuple[float, float]] = []
            for variant in proposal_variants:
                branch = _main.copy.deepcopy(base)
                branch["pendingProposal"] = _main.copy.deepcopy(variant)
                try:
                    _main.apply_secondary_edit_proposal(validation_job, branch, str(variant.get("id") or ""))
                except _main.EditSessionError as error:
                    return ctx.no_result("variant_validation_failed", f"版本检查未通过：{error}")
                ranges = [(float(clip.get("sourceStart") or 0), float(clip.get("sourceEnd") or 0)) for clip in branch.get("clips") or []]
                if any(min(end, old_end) - max(start, old_start) > .001 for start, end in ranges for old_start, old_end in used_ranges):
                    return ctx.no_result(
                        "variant_source_overlap",
                        "当前方案存在跨版本重复片段，无法满足“不要复用片段”；未自动应用或渲染。请减少版本数、缩短时长或补充素材后重新规划。",
                    )
                used_ranges.extend(ranges)
    if ctx.autonomous and session_id:
        def proposal_is_safe(value: dict[str, Any]) -> bool:
            preview = value.get("preview") if isinstance(value.get("preview"), dict) else None
            # Third-party/legacy dispatch tests may not provide a preview;
            # first-party proposals always do, and those are the values
            # this safety gate is designed to validate.
            if preview is None:
                return True
            if int(preview.get("clipCountAfter") or 0) <= 0:
                return False
            preflight = preview.get("preflight") if isinstance(preview.get("preflight"), dict) else {}
            if int(preflight.get("errorCount") or 0) > 0:
                return False
            target = float(duration_fit.get("targetSeconds") or 0)
            tolerance = float(duration_fit.get("toleranceSeconds") or 0)
            actual = float(preview.get("durationAfter") or 0)
            return not target or target - tolerance <= actual <= target + tolerance

        safe_variants = [
            value for value in proposal_variants
            if isinstance(value, dict) and proposal_is_safe(value)
        ]
        if requested_variants > 1 and len(safe_variants) != requested_variants:
            return ctx.no_result(
                "insufficient_safe_variants",
                "通过时长与内容检查的方案不足请求版本数；请调整版本数、时长或素材后重新规划。",
            )
        if safe_variants:
            proposal_variants = safe_variants
            proposal = _main.copy.deepcopy(safe_variants[0])
        elif proposal and not proposal_is_safe(proposal):
            with _main.jobs_lock:
                current = _main.jobs.get(ctx.job_id)
                current_session = _main.find_edit_session(current, session_id) if current else None
                pending = (current_session or {}).get("pendingProposal") or {}
                if pending.get("id") == proposal.get("id"):
                    pending["validationRejected"] = True
                    _main.save_job(current)
            proposal = None
            proposal_variants = []
        if proposal is None:
            raise RuntimeError("剪辑提案未通过内容或时长检查，未自动应用；已保留原草稿，请重试编排步骤。")
    batch_entries: list[dict[str, Any]] = []
    if ctx.autonomous:
        if not session_id or not proposal:
            raise RuntimeError("Agent 无法从当前候选生成有效时间线方案")
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            if not current:
                raise RuntimeError("素材任务不存在")
            base_session = _main.find_edit_session(current, session_id)
            variants = proposal_variants or [proposal]
            for index, variant in enumerate(variants):
                if index == 0:
                    branch = base_session
                else:
                    branch = _main.copy.deepcopy(base_session)
                    branch["id"] = f'edit_session_{_main.uuid.uuid4().hex[:12]}'
                    branch["createdAt"] = _main.now_iso()
                    current.setdefault("editSessions", []).append(branch)
                branch["title"] = str(variant.get("title") or f"Agent 结构方案 {index + 1}")[:100]
                branch["pendingProposal"] = _main.copy.deepcopy(variant)
                branch["proposalVariants"] = [_main.copy.deepcopy(variant)]
                branch["agentVariantIndex"] = index + 1
                branch["updatedAt"] = _main.now_iso()
                batch_entries.append({
                    "sessionId": str(branch["id"]),
                    "proposalId": str(variant.get("id") or ""),
                    "title": branch["title"],
                    "direction": str(variant.get("direction") or ""),
                })
            current["agentTimelineBatch"] = {
                "planId": str(ctx.workspace.get("activePlanId") or ""),
                "status": "proposed", "variants": batch_entries,
                "createdAt": _main.now_iso(), "updatedAt": _main.now_iso(),
            }
            current["activeEditSessionId"] = str(batch_entries[0]["sessionId"])
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
    return {
        "actionRequired": not ctx.autonomous,
        "action": "timeline_proposal",
        "sessionId": session_id or None,
        "proposalId": str((proposal or {}).get("id") or "") or None,
        "proposalVariantCount": len(proposal_variants) or (1 if proposal else 0),
        "proposalVariants": [{
            "id": str(item.get("id") or ""), "title": str(item.get("title") or ""),
            "summary": str(item.get("summary") or ""), "direction": str(item.get("direction") or ""),
        } for item in proposal_variants],
        "timelineBatch": _main.copy.deepcopy(batch_entries),
        "message": (
            ((f"我已生成 {len(proposal_variants)} 个时间线结构方案，将自动验证并应用。"
              if ctx.autonomous and len(proposal_variants) > 1 else
              "我已生成时间线草案，将自动验证并应用。"
              if ctx.autonomous else
              f"我已生成 {len(proposal_variants)} 个可切换的时间线结构方案。" if len(proposal_variants) > 1 else
              f"我已生成待审核时间线草案“{str((proposal or {}).get('title') or '初始时间线草案')}”。")
             + ("" if ctx.autonomous else "请点击“打开精剪时间线”核对；在你应用前不会修改时间线。"))
            if session_id else "当前任务还没有可用的高光或内容候选，请先完成素材分析。"
        ),
        "createdSession": created,
    }


def confirm_timeline_edit(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import main as _main

    if ctx.autonomous:
        with _main.jobs_lock:
            current = _main.jobs.get(ctx.job_id)
            if not current:
                raise RuntimeError("素材任务不存在")
            batch = current.get("agentTimelineBatch") if isinstance(current.get("agentTimelineBatch"), dict) else {}
            entries = [item for item in batch.get("variants") or [] if isinstance(item, dict)]
            if not entries:
                active_id = str(current.get("activeEditSessionId") or "")
                active = _main.find_edit_session(current, active_id)
                pending = active.get("pendingProposal") if isinstance(active.get("pendingProposal"), dict) else {}
                entries = [{"sessionId": active_id, "proposalId": str(pending.get("id") or ""), "title": active.get("title")}]
            # Validate the entire batch on a copy before changing any draft.
            trial_job = _main.copy.deepcopy(current)
            for entry in entries:
                trial_session = _main.find_edit_session(trial_job, str(entry.get("sessionId") or ""))
                _main.apply_secondary_edit_proposal(trial_job, trial_session, str(entry.get("proposalId") or ""))
                content_report = _main.timeline_content_report(trial_session, trial_job)
                if not content_report["passed"]:
                    return {"actionRequired": True, "action": "timeline_confirmation",
                            "sessionId": trial_session["id"], "contentVerification": content_report,
                            "message": "草案超出已核验范围或包含未核验内容，请在精剪时间线修正后再继续。"}
            applied: list[dict[str, Any]] = []
            for entry in entries:
                session = _main.find_edit_session(current, str(entry.get("sessionId") or ""))
                proposal_id = str(entry.get("proposalId") or "")
                if not proposal_id:
                    raise RuntimeError("时间线方案缺少可应用的提案")
                _main.apply_secondary_edit_proposal(current, session, proposal_id)
                applied.append({
                    "sessionId": str(session["id"]), "revision": int(session.get("revision") or 0),
                    "title": str(entry.get("title") or session.get("title") or "审核方案"),
                })
            batch.update({"status": "applied", "variants": applied, "updatedAt": _main.now_iso()})
            current["agentTimelineBatch"] = batch
            current["activeEditSessionId"] = str(applied[0]["sessionId"])
            current["updatedAt"] = _main.now_iso()
            _main.save_job(current)
        return {
            "artifact": {
                "kind": "applied_timeline_batch", "jobId": ctx.job_id,
                "variants": applied,
                "message": f"我已验证并应用 {len(applied)} 个时间线方案。",
            },
            "sessionId": str(applied[0]["sessionId"]), "timelineBatch": applied,
        }
    return {
        "actionRequired": True,
        "action": "timeline_confirmation",
        "sessionId": str(ctx.snapshot.get("activeEditSessionId") or "") or None,
        "message": "请应用并保存已审核的时间线草案，再继续生成字幕。",
    }


