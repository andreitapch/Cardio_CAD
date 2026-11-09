"""Citation discovery tool: searches the Semantic Scholar API (a real,
external academic database -- not the model's memory) for papers that
could support a claim needing a citation.

Deliberately SUGGEST-ONLY: this tool never writes to the .bib file and
the Author is instructed to never \\cite a key from these results
directly. It returns real titles/authors/years/DOIs for a human to
review and add themselves, closing the gap where an LLM might otherwise
invent a plausible-looking source instead of admitting nothing fits.
"""

from typing import Any

import requests
from claude_agent_sdk import tool, create_sdk_mcp_server, ToolAnnotations

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,venue,externalIds,abstract"
TIMEOUT_SECONDS = 10


@tool(
    "search_candidate_papers",
    "Search Semantic Scholar (a real external academic database) for "
    "papers that could support a claim needing a citation. Returns real "
    "titles/authors/years/DOIs for the user to review and add to the "
    ".bib themselves -- this does NOT add anything to the bibliography, "
    "and you must NEVER \\cite a key from these results directly, since "
    "no .bib entry exists for them yet. Use this when a claim genuinely "
    "needs a citation and none of the existing .bib entries fit.",
    {"query": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def search_candidate_papers(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"]
    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params={"query": query, "limit": 5, "fields": FIELDS},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        papers = response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Search failed ({e}). Do not fabricate a citation as a "
                    "fallback -- report this claim as needing manual literature search "
                    "instead (%% NEEDS CITATION: ...).",
                }
            ],
            "is_error": True,
        }

    if not papers:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"No candidates found for '{query}'. Do not fabricate a "
                    "citation -- mark this claim %% NEEDS CITATION: ... instead.",
                }
            ]
        }

    lines = [f"Candidates for '{query}' (from Semantic Scholar, real external results):"]
    for p in papers:
        authors = ", ".join(a.get("name", "?") for a in (p.get("authors") or [])[:4])
        doi = (p.get("externalIds") or {}).get("DOI", "no DOI")
        abstract = (p.get("abstract") or "")[:200]
        lines.append(
            f"- \"{p.get('title', '?')}\" ({p.get('year', '?')}) -- {authors} -- "
            f"{p.get('venue', '?')} -- DOI: {doi}"
            + (f"\n  Abstract snippet: {abstract}..." if abstract else "")
        )
    lines.append(
        "\nThese are SUGGESTIONS only. Do not \\cite any of these directly -- no .bib "
        "entry exists for them. Mark the claim with "
        '"%% SUGGESTED CITATION: <title> (<year>) -- <DOI> -- verify and add to .bib" '
        "so the user can review and add the real entry themselves."
    )
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


citation_discovery_server = create_sdk_mcp_server(
    name="citation_discovery",
    version="1.0.0",
    tools=[search_candidate_papers],
)
