"""The Author agent: drafts or revises a manuscript section, with access
to the LLM council and compute node tools. Runs in a single fresh session
each round — it doesn't need session memory of its own, because critic
feedback is passed back to it explicitly in the prompt each time.
"""

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from compute_node import compute_server
from council import council_server
from code_reference import code_reference_server
from bib_reference import bib_reference_server
from review_notes import review_notes_server
from citation_discovery import citation_discovery_server
from paper_context import load_paper_overview, load_review_notes_for_section

_AUTHOR_BASE_PROMPT = """
You are the Author for a section of an academic cardiology manuscript on
LV remodeling after myocardial infarction (LaTeX/BibTeX). Your job is to
draft or revise the given section. If the given section is thin or
incomplete, write it out fully rather than lightly editing -- draft
whatever is missing, using the paper overview and any cross-section
notes below for context.

Hard rules -- never violate these:
- NEVER fabricate a bibliography entry or invent a citation key that
  doesn't exist. If a claim needs a citation and nothing in the existing
  .bib fits, call search_candidate_papers to find a REAL candidate from
  Semantic Scholar. If it finds one, write
  "%% SUGGESTED CITATION: <title> (<year>) -- <DOI> -- verify and add to
  .bib" inline. If it finds nothing, write "%% NEEDS CITATION: <what
  claim needs support>" instead. Either way, NEVER \\cite a key that
  doesn't already exist in the .bib -- search results are suggestions
  for the user to review and add themselves, not citable keys.
- NEVER invent results, numbers, or statistics that aren't already
  established elsewhere (the draft's own history, a compute-node
  verification, or an existing citation). If a Results-type claim is
  needed but the underlying data isn't available to you, write
  "%% RESULTS PENDING: <what's needed>" inline instead of fabricating a
  plausible-sounding number.

Before finalizing your draft:
- If it contains any derivation, formula, or numeric claim, call
  verify_algebraic_equality to check the key algebraic steps.
- If you write or keep a \\cite{key}, especially a new one you're adding,
  call read_bib_entry to confirm the key actually exists in the
  bibliography and its entry plausibly supports the claim.
- If you are writing or revising a Methods claim about the multi-method
  feature selection pipeline (which methods were used, how results were
  combined, validation strategy, etc.), use search_codebase and
  read_code_file to verify the claim against the actual implementation
  before stating it. Only pull in what's relevant to the specific claim
  you're writing -- don't dump unrelated code into the draft.
- If, while writing, you notice this section depends on something a
  DIFFERENT section needs to establish (e.g. this Introduction previews a
  result that Results must actually report), call add_review_note with
  that section's name and a concise note.
- Consider calling consult_council for a quick read from reviewer personas,
  especially after a substantial revision.

Tool-call budget for this draft (combined across all tools above): at
most 12 calls total. Prioritize verifying the most important/novel claims
first. If you reach the budget before every claim is checked, finalize
the draft anyway, marking any remaining unverified claim inline with
"%% UNVERIFIED: <what wasn't checked>" rather than continuing to call
tools -- a finished draft with honest gaps beats an unfinished one.

Output only the revised LaTeX section text, with no commentary outside it.
""".strip()


def _build_author_prompt(section: str) -> str:
    """Read fresh on every call so edits to paper_overview.md or new
    review notes take effect on the very next round."""
    return (
        f"{_AUTHOR_BASE_PROMPT}\n\n"
        f"--- Paper overview (for context; you are writing '{section}') ---\n"
        f"{load_paper_overview()}\n\n"
        f"--- Existing notes addressed to '{section}' from other sections ---\n"
        f"{load_review_notes_for_section(section)}"
    )


def _build_author_options(section: str) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=_build_author_prompt(section),
        mcp_servers={
            "compute_node": compute_server,
            "council": council_server,
            "code_reference": code_reference_server,
            "bib_reference": bib_reference_server,
            "review_notes": review_notes_server,
            "citation_discovery": citation_discovery_server,
        },
        tools=[],  # no built-in file/bash tools; only the MCP tools below
        allowed_tools=[
            "mcp__compute_node__verify_algebraic_equality",
            "mcp__council__consult_council",
            "mcp__code_reference__search_codebase",
            "mcp__code_reference__read_code_file",
            "mcp__bib_reference__read_bib_entry",
            "mcp__bib_reference__search_bib",
            "mcp__review_notes__add_review_note",
            "mcp__citation_discovery__search_candidate_papers",
        ],
        max_turns=30,
        model="sonnet",
        effort="medium",
    )


async def _run_author_query(prompt: str, section: str) -> str:
    options = _build_author_options(section)
    result_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            result_text = message.result
    return result_text


async def draft_or_revise(problem_or_feedback: str, section: str) -> str:
    """Run the Author on either the initial problem statement or a
    critic's rejection feedback, and return the resulting draft text.
    Retries once with a "wrap up now" nudge if the first attempt errors
    out (e.g. hitting the turn budget mid tool-call) or returns nothing,
    rather than crashing the whole pipeline run on one bad attempt."""
    try:
        result_text = await _run_author_query(problem_or_feedback, section)
        if result_text:
            return result_text
    except Exception as e:
        print(f"Author call failed ({e}); retrying with a tighter instruction.")

    nudge = (
        f"{problem_or_feedback}\n\n"
        f"(Note: a previous attempt at this didn't finish. This time, "
        f"use at most 3 tool calls total and finalize your draft well "
        f"within budget, marking anything unchecked with "
        f'"%% UNVERIFIED: ..." rather than risking another incomplete run.)'
    )
    result_text = await _run_author_query(nudge, section)
    if not result_text:
        raise RuntimeError(
            "Author failed to produce a draft after a retry. Check the "
            "traceback above for the underlying error."
        )
    return result_text
