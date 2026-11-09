"""LLM council: a tool the Author calls to get quick feedback from several
reviewer personas before finalizing a draft. Implemented as parallel
one-shot query() calls with different system prompts, fanned out with
asyncio.gather and combined into a single tool result.
"""

import asyncio
from typing import Any

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    tool,
    create_sdk_mcp_server,
    ToolAnnotations,
)

PERSONAS = {
    "skeptical_mathematician": (
        "You are a skeptical mathematician reviewing a cardiology "
        "manuscript excerpt. Look only for logical/derivation gaps or "
        "unjustified steps. Be terse: a short bullet list, max 5 items."
    ),
    "clarity_editor": (
        "You are a scientific clarity editor. Look only for ambiguous, "
        "overlong, or unclear sentences. Be terse: a short bullet list, "
        "max 5 items."
    ),
    "domain_expert": (
        "You are a cardiology domain expert specializing in LV "
        "remodeling after myocardial infarction. Look only for claims "
        "that are imprecise, outdated, or inconsistent with the current "
        "literature. Be terse: a short bullet list, max 5 items."
    ),
    "domain_expert_machine_learning": (
        "You are a machine learning methodologist reviewing an excerpt from a "
        "paper on multi-method feature selection with clinical interpretation "
        "in cardiology. Check specifically for:\n"
        "- Data leakage: feature selection performed before train/test split "
        "or cross-validation fold separation, rather than within each fold\n"
        "- Vague or missing description of which feature selection methods "
        "were combined and how their results were aggregated or reconciled "
        "when they disagree\n"
        "- Missing or inappropriate validation strategy for the claims made "
        "(e.g. no cross-validation, no external/held-out cohort, small "
        "sample size relative to feature count)\n"
        "- Clinical interpretability claims that overstate what the "
        "selected features/model actually support (e.g. claiming causal "
        "importance from a purely predictive or correlational method)\n"
        "- Outdated or imprecise ML terminology, or inconsistency with "
        "current feature-selection/interpretability literature (e.g. SHAP, "
        "permutation importance, embedded vs. filter vs. wrapper methods)\n"
        "- Missing baseline comparisons or performance metrics inappropriate "
        "for the clinical task (e.g. accuracy alone on an imbalanced "
        "outcome, no calibration or discrimination metrics)\n"
        "Be terse: a short bullet list, max 5 items, each citing the "
        "specific sentence or claim it applies to."
    ),
}


async def _ask_persona(persona_name: str, system_prompt: str, draft: str) -> str:
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=3,
        model="sonnet",
        effort="medium",
        tools=[],  # text-only reviewers, no file/bash/tool access needed
    )
    result_text = ""
    async for message in query(prompt=draft, options=options):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            result_text = message.result
    return f"### {persona_name}\n{result_text}"


@tool(
    "consult_council",
    "Get quick feedback on the current draft from a council of reviewer "
    "personas (skeptical mathematician, clarity editor, domain expert), "
    "run in parallel. Use this before submitting a draft to the critic, "
    "especially after a significant revision.",
    {"draft": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def consult_council(args: dict[str, Any]) -> dict[str, Any]:
    draft = args["draft"]
    responses = await asyncio.gather(
        *[_ask_persona(name, prompt, draft) for name, prompt in PERSONAS.items()]
    )
    return {"content": [{"type": "text", "text": "\n\n".join(responses)}]}


council_server = create_sdk_mcp_server(
    name="council",
    version="1.0.0",
    tools=[consult_council],
)
