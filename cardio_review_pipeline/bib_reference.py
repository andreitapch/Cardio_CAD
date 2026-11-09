"""Bibliography reference tool: lets the Author/Critics check citations in
the draft against the ACTUAL .bib file(s), instead of reasoning about
citations blind. Scoped to BIB_DIR -- searches every *.bib file found
there (recursively) rather than requiring an exact filename.
"""

import os
import re
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server, ToolAnnotations

BIB_DIR = Path(
    os.environ.get(
        "CARDIO_BIB_DIR",
        r"D:\PHD\Papers\Journal-Multi-Method_Feature_Selection_and clinical_Based_Interpretation\sections",
    )
).resolve()

MAX_CHARS = 4000
MAX_MATCHES = 30

_ENTRY_START_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,")


def _find_bib_files() -> list[Path]:
    return sorted(BIB_DIR.rglob("*.bib"))


def _extract_entry_simple(text: str, at_symbol_idx: int) -> str | None:
    """Given the index of the '@' starting a bibtex entry, extract the
    full entry up to its matching closing brace."""
    brace_idx = text.find("{", at_symbol_idx)
    if brace_idx == -1:
        return None
    depth = 0
    for i in range(brace_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[at_symbol_idx : i + 1]
    return None


@tool(
    "read_bib_entry",
    "Look up the exact BibTeX entry for a citation key (as used in "
    "\\cite{key}) across all .bib files in the bibliography folder. "
    "Returns the raw entry (type, fields, title, etc.) or 'not found' if "
    "the key doesn't exist in the bibliography -- which itself is an "
    "issue worth flagging (a \\cite to a nonexistent key).",
    {"key": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def read_bib_entry(args: dict[str, Any]) -> dict[str, Any]:
    key = args["key"].strip()
    bib_files = _find_bib_files()
    if not bib_files:
        return {
            "content": [{"type": "text", "text": f"No .bib files found under {BIB_DIR}."}],
            "is_error": True,
        }

    for path in bib_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in _ENTRY_START_RE.finditer(text):
            if m.group(2) == key:
                entry = _extract_entry_simple(text, m.start())
                if entry:
                    if len(entry) > MAX_CHARS:
                        entry = entry[:MAX_CHARS] + "\n... [truncated]"
                    return {
                        "content": [
                            {"type": "text", "text": f"Found in {path.name}:\n{entry}"}
                        ]
                    }

    return {
        "content": [
            {
                "type": "text",
                "text": f"Key '{key}' NOT FOUND in any .bib file under {BIB_DIR} "
                f"(searched: {', '.join(p.name for p in bib_files)}).",
            }
        ]
    }


@tool(
    "search_bib",
    "Search all .bib files in the bibliography folder for a keyword "
    "(author surname, title fragment, year, etc.) -- useful for spotting "
    "possible duplicate entries for the same source under different "
    "citation keys, or finding what key an author/title actually uses.",
    {"query": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def search_bib(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"]
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches: list[str] = []
    for path in _find_bib_files():
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if pattern.search(line):
                matches.append(f"{path.name}:{i}: {line.strip()}")
                if len(matches) >= MAX_MATCHES:
                    break
        if len(matches) >= MAX_MATCHES:
            break
    text = "\n".join(matches) if matches else f"No matches for '{query}' in any .bib file."
    return {"content": [{"type": "text", "text": text}]}


bib_reference_server = create_sdk_mcp_server(
    name="bib_reference",
    version="1.0.0",
    tools=[read_bib_entry, search_bib],
)
