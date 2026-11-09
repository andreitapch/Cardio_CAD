"""Tool for leaving a note addressed to another section of the paper --
e.g. the Introduction critic noticing that a claim needs to be backed up
with a specific validation result in Methods. Appends to review_notes.md
at PAPER_ROOT, grouped under a '## <section>' heading, which
paper_context.load_review_notes_for_section() reads back when that
section is later reviewed.
"""

from datetime import datetime, timezone
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server, ToolAnnotations

from paper_context import NOTES_PATH

VALID_SECTIONS = {"introduction", "methods", "results", "discussion", "conclusion"}


@tool(
    "add_review_note",
    "Leave a note for a DIFFERENT section of the paper to address later "
    "(e.g. while reviewing the Introduction, noting that Methods needs to "
    "describe a specific validation step to support a claim made here). "
    "Do not use this for issues within the current section -- those go in "
    "your normal accept/reject feedback instead. target_section must be "
    "one of: introduction, methods, results, discussion, conclusion.",
    {"target_section": str, "note": str},
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
)
async def add_review_note(args: dict[str, Any]) -> dict[str, Any]:
    target = args["target_section"].strip().lower()
    note = args["note"].strip()

    if target not in VALID_SECTIONS:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"'{target}' is not a valid section. Use one of: "
                    f"{', '.join(sorted(VALID_SECTIONS))}.",
                }
            ],
            "is_error": True,
        }

    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    heading = f"## {target}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"- [{timestamp}] {note}\n"

    if NOTES_PATH.is_file():
        text = NOTES_PATH.read_text(encoding="utf-8", errors="ignore")
    else:
        text = ""

    if heading in text:
        # Insert right after the existing heading line.
        lines = text.splitlines(keepends=True)
        out = []
        inserted = False
        for line in lines:
            out.append(line)
            if not inserted and line.strip() == heading:
                out.append(entry)
                inserted = True
        text = "".join(out)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"\n{heading}\n{entry}"

    NOTES_PATH.write_text(text, encoding="utf-8")
    return {
        "content": [
            {"type": "text", "text": f"Note added for '{target}' in {NOTES_PATH.name}."}
        ]
    }


review_notes_server = create_sdk_mcp_server(
    name="review_notes",
    version="1.0.0",
    tools=[add_review_note],
)
