"""Paper-level context shared by the Author and both critics, so a single
section isn't reviewed in total isolation from the rest of the paper.

Two files live at PAPER_ROOT (not inside this pipeline folder):

- paper_overview.md   Title/abstract/outline. Static context, edited by
                       hand. Read fresh on every call so edits take effect
                       on the next round without restarting anything.
- review_notes.md      Cross-section comments left by the Author/critics
                       via the add_review_note tool (see review_notes.py),
                       grouped by target section. Read fresh the same way.

Both are optional -- if missing, a short placeholder is returned instead
of failing, so the pipeline still runs before you've created them.
"""

import os
from pathlib import Path

PAPER_ROOT = Path(
    os.environ.get(
        "CARDIO_PAPER_ROOT",
        r"D:\PHD\Papers\Journal-Multi-Method_Feature_Selection_and clinical_Based_Interpretation",
    )
).resolve()

OVERVIEW_PATH = PAPER_ROOT / "paper_overview.md"
NOTES_PATH = PAPER_ROOT / "review_notes.md"


def load_paper_overview() -> str:
    if OVERVIEW_PATH.is_file():
        return OVERVIEW_PATH.read_text(encoding="utf-8", errors="ignore").strip()
    return (
        "(No paper_overview.md found at "
        f"{OVERVIEW_PATH} -- reviewing this section with no knowledge of "
        "the paper's overall argument, title, or other sections.)"
    )


def load_review_notes_for_section(section: str) -> str:
    """Returns only the notes addressed to `section` (case-insensitive
    match on the '## <section>' heading), or a placeholder if none exist."""
    if not NOTES_PATH.is_file():
        return "(No review_notes.md yet -- no cross-section notes to consider.)"

    text = NOTES_PATH.read_text(encoding="utf-8", errors="ignore")
    target_heading = f"## {section.strip().lower()}"
    lines = text.splitlines()
    collected: list[str] = []
    capturing = False
    for line in lines:
        if line.strip().lower().startswith("## "):
            capturing = line.strip().lower() == target_heading
            continue
        if capturing:
            collected.append(line)

    body = "\n".join(collected).strip()
    return body if body else f"(No review notes addressed to '{section}' yet.)"
