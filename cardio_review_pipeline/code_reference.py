"""Code reference tool: lets the Author/Critics look up specific parts of
the actual implementation to verify methodology claims in the paper match
what the code does. Scoped strictly to CODE_ROOT -- no path traversal
outside it. Deliberately targeted (search, then read a range) rather than
dumping the whole codebase into every prompt.
"""

import os
import re
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server, ToolAnnotations

CODE_ROOT = Path(
    os.environ.get("CARDIO_CODE_ROOT", r"D:\PHD\Development\Python\Cardio_CAD\cardio_app")
).resolve()

MAX_CHARS = 8000  # cap individual tool results; targeted reads, not dumps
MAX_MATCHES = 30


def _resolve_within_root(relative_path: str) -> Path:
    candidate = (CODE_ROOT / relative_path).resolve()
    if candidate != CODE_ROOT and CODE_ROOT not in candidate.parents:
        raise ValueError(
            f"'{relative_path}' resolves outside the allowed code root ({CODE_ROOT})."
        )
    return candidate


@tool(
    "search_codebase",
    "Search the paper's implementation code for a keyword, function name, "
    "or pattern (e.g. 'cross_val', 'feature_selection', 'SHAP') to find "
    "which file(s) and line(s) implement something described in the "
    "paper's Methods section. Use this BEFORE read_code_file to locate the "
    "right spot rather than guessing a path.",
    {"query": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def search_codebase(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"]
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches: list[str] = []
    for path in CODE_ROOT.rglob("*.py"):
        try:
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if pattern.search(line):
                    rel = path.relative_to(CODE_ROOT)
                    matches.append(f"{rel}:{i}: {line.strip()}")
                    if len(matches) >= MAX_MATCHES:
                        break
        except (UnicodeDecodeError, PermissionError):
            continue
        if len(matches) >= MAX_MATCHES:
            break
    text = "\n".join(matches) if matches else f"No matches for '{query}' under {CODE_ROOT}."
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "read_code_file",
    "Read a specific file, or a line range within it, from the paper's "
    "implementation code -- to verify a methods claim against the actual "
    "code. Find the right file/line with search_codebase first. Path is "
    "relative to the code root, e.g. 'feature_selection/methods.py'. "
    "Omit start_line/end_line to read the first ~80 lines.",
    {
        "type": "object",
        "properties": {
            "relative_path": {"type": "string"},
            "start_line": {"type": "integer", "description": "1-indexed; omit for line 1"},
            "end_line": {"type": "integer", "description": "1-indexed; omit for start+80"},
        },
        "required": ["relative_path"],
    },
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def read_code_file(args: dict[str, Any]) -> dict[str, Any]:
    relative_path = args["relative_path"]
    try:
        path = _resolve_within_root(relative_path)
    except ValueError as e:
        return {"content": [{"type": "text", "text": str(e)}], "is_error": True}

    if not path.is_file():
        return {
            "content": [{"type": "text", "text": f"Not a file: {relative_path}"}],
            "is_error": True,
        }

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = max(int(args.get("start_line", 1)) - 1, 0)
    end = min(int(args.get("end_line", start + 80)), len(lines))
    snippet = "\n".join(lines[start:end])
    if len(snippet) > MAX_CHARS:
        snippet = snippet[:MAX_CHARS] + "\n... [truncated, request a narrower range]"

    return {
        "content": [
            {"type": "text", "text": f"{relative_path} (lines {start + 1}-{end}):\n{snippet}"}
        ]
    }


code_reference_server = create_sdk_mcp_server(
    name="code_reference",
    version="1.0.0",
    tools=[search_codebase, read_code_file],
)
