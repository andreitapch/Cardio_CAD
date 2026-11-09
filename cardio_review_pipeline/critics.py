"""Two critics sharing one rubric, differing only in session handling:

- StatefulCritic reuses the same SDK session across rounds (via `resume`),
  so it remembers what it flagged previously and can check whether the
  Author actually addressed it.
- FreshCritic starts a brand-new session every call (no `resume`), so it
  reviews with no memory of prior rounds -- a periodic unbiased re-audit.
"""

from dataclasses import dataclass

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from rubric import build_critic_rubric
from code_reference import code_reference_server
from bib_reference import bib_reference_server
from review_notes import review_notes_server
from citation_discovery import citation_discovery_server

TOOLS = {
    "mcp_servers": {
        "code_reference": code_reference_server,
        "bib_reference": bib_reference_server,
        "review_notes": review_notes_server,
        "citation_discovery": citation_discovery_server,
    },
    "allowed_tools": [
        "mcp__code_reference__search_codebase",
        "mcp__code_reference__read_code_file",
        "mcp__bib_reference__read_bib_entry",
        "mcp__bib_reference__search_bib",
        "mcp__review_notes__add_review_note",
        "mcp__citation_discovery__search_candidate_papers",
    ],
}


@dataclass
class CriticVerdict:
    accepted: bool
    feedback: str


def _parse_verdict(result_text: str) -> CriticVerdict:
    accepted = result_text.strip().upper().startswith("ACCEPT") or (
        "ACCEPT" in result_text.splitlines()[0].upper()
        if result_text.splitlines()
        else False
    )
    return CriticVerdict(accepted=accepted, feedback=result_text)


async def _run_critic_query(options: ClaudeAgentOptions, draft: str) -> tuple[str, str | None]:
    """Runs a critic query and returns (result_text, session_id).
    On any SDK/process error (e.g. hitting the turn budget while still
    mid tool-call), returns a synthetic reject instead of raising, so one
    bad round doesn't crash the whole pipeline run."""
    try:
        result_text = ""
        session_id = None
        async for message in query(prompt=draft, options=options):
            if isinstance(message, ResultMessage) and message.subtype == "success":
                result_text = message.result
                session_id = message.session_id
        if not result_text:
            result_text = (
                "REJECT\n1. Critic did not return a usable result "
                "(empty response) -- treating as reject; re-run this round."
            )
        return result_text, session_id
    except Exception as e:
        return (
            f"REJECT\n1. Critic call failed with an error and could not "
            f"complete review: {e}. This is an infrastructure failure, "
            f"not a real review -- re-run this round.",
            None,
        )


class StatefulCritic:
    """Same-session critic. Call repeatedly across rounds; it keeps context
    of everything it has previously flagged."""

    def __init__(self, section: str):
        self.section = section
        self._session_id: str | None = None

    async def review(self, draft: str) -> CriticVerdict:
        options = ClaudeAgentOptions(
            system_prompt=build_critic_rubric(self.section),
            max_turns=14,  # room for code/bib/review-notes tool calls + final answer
            resume=self._session_id,  # None on first call -> starts fresh, then persists
            model="sonnet",
            effort="medium",
            tools=[],  # no built-in file/bash tools; only the MCP tools below
            **TOOLS,
        )
        result_text, session_id = await _run_critic_query(options, draft)
        if session_id:
            self._session_id = session_id
        return _parse_verdict(result_text)


async def fresh_critic_review(draft: str, section: str) -> CriticVerdict:
    """New-session critic. No `resume` -- always starts with zero memory
    of prior rounds."""
    options = ClaudeAgentOptions(
        system_prompt=build_critic_rubric(section),
        max_turns=14,  # room for code/bib/review-notes tool calls + final answer
        model="sonnet",
        effort="medium",
        tools=[],  # no built-in file/bash tools; only the MCP tools below
        **TOOLS,
    )
    result_text, _ = await _run_critic_query(options, draft)
    return _parse_verdict(result_text)
