# cardio_review_pipeline

Self-review loop for the LV remodeling manuscript (cardioCad), built on the
Claude Agent SDK. Drop this folder inside the cardioCad repo, alongside the
LaTeX source — it does not touch the manuscript files directly, only reads
and proposes edits to them.

## What it does

An **Author** agent drafts or revises a section (prose or a proof/derivation).
Before every run, it (and both critics) read `paper_overview.md` and
`review_notes.md` from the **paper root** (not this folder), so a section
is reviewed with knowledge of the paper's overall argument and any
comments left by other sections — not in total isolation. Neither the
Author nor the critics ever fabricate a citation or a result: if
something's missing, they write `%% NEEDS CITATION: ...` or
`%% RESULTS PENDING: ...` inline instead of inventing a plausible-looking
one.

Before finalizing, the Author can call:

- **LLM council** — parallel calls to Claude playing different reviewer
  personas (skeptical reader, clarity editor, domain expert), returned as one
  combined note.
- **Compute node** — a SymPy-backed tool that checks algebraic/derivation
  steps symbolically.
- **Code reference** — search and read specific files/line ranges from the
  paper's implementation code, to verify Methods claims (which feature
  selection methods were used, how results were combined, validation
  strategy) against what the code actually does. Scoped strictly to one
  folder (default `D:\PHD\Development\Python\Cardio_CAD\dash_viewer`,
  override with the `CARDIO_CODE_ROOT` environment variable) and used
  deliberately narrowly: search first, then read only the relevant lines
  — never a full-codebase dump into the prompt.
- **Bib reference** — look up and search the actual `.bib` file(s) so
  citation checks (does this key exist, does the entry plausibly support
  the claim, are two keys duplicates of the same source) are grounded in
  the real bibliography rather than judged from memory. Scoped to the
  folder containing the `.bib` file(s) (default is the same folder as
  `Introduction.tex`; override with the `CARDIO_BIB_DIR` environment
  variable) and searches every `*.bib` file found there.
- **Review notes** — leave a note for a *different* section to pick up
  later (e.g. the Introduction noting that Methods needs to describe a
  specific validation step). Appended to `review_notes.md` at the paper
  root; read back automatically the next time that section is run.
- **Citation discovery** — when a claim genuinely needs a citation and
  nothing in the `.bib` fits, searches Semantic Scholar (a real external
  academic database, not the model's memory) for candidates. This is
  suggest-only: it never writes to the `.bib` and the Author is
  instructed to never `\cite` a result directly. Instead it marks the
  spot with `%% SUGGESTED CITATION: <title> (<year>) -- <DOI> -- verify
  and add to .bib` for you to review and add the real entry yourself.
  This exists specifically to avoid the alternative failure mode: an LLM
  inventing a plausible-looking fake citation instead of admitting
  nothing fits.

The draft then goes to a critic. Every round is reviewed by the **Stateful
critic** (same SDK session — it remembers what it flagged last time and
checks whether the Author actually fixed it), except every `k=3` rounds,
when the **Fresh critic** runs instead in a brand-new session, with no memory
of prior rounds, to catch things the Stateful critic may have gotten
anchored on. Both critics apply the same rubric (see `rubric.py`), which
includes the citation-accuracy checks from the earlier audit of this
manuscript: duplicate citation keys, claim-citation mismatches, and
malformed bib fields.

A rejection sends the critic's reasoning back to the Author for another
round. An acceptance returns the final draft.

## Setup

```bash
pip install claude-agent-sdk sympy
export ANTHROPIC_API_KEY=...   # or run inside an authenticated Claude Code session
```

Also create `paper_overview.md` in the **paper root**, not this folder
(default `D:\PHD\Papers\Journal-Multi-Method_Feature_Selection_and clinical_Based_Interpretation\`,
override with `CARDIO_PAPER_ROOT`). A first draft is included here as
`paper_overview_DRAFT.md` — it's inferred, not verified; read and correct
it, then save it as `paper_overview.md` in the paper root.

## Run

```bash
python orchestrator.py --input path/to/introduction.tex --section introduction
```

`--section` must be one of: `introduction`, `methods`, `results`,
`discussion`, `conclusion`. If omitted, it's inferred from the input
filename (so `Introduction.tex` alone is usually enough).

## Files

- `rubric.py` — shared evaluation criteria both critics apply
- `paper_context.py` — loads paper_overview.md and per-section review notes
- `paper_overview_DRAFT.md` — first-draft paper overview; correct and move to the paper root
- `review_notes.py` — tool for leaving cross-section notes
- `compute_node.py` — SymPy verification tool
- `code_reference.py` — scoped code search/read tool for verifying Methods claims
- `bib_reference.py` — scoped .bib search/read tool for verifying citations
- `citation_discovery.py` — Semantic Scholar search tool for suggesting (never adding) real candidate citations
- `council.py` — multi-persona LLM council tool
- `author.py` — the Author agent
- `critics.py` — Stateful critic (session resume) and Fresh critic (new session)
- `orchestrator.py` — the round-by-round control loop, entry point

## Notes

- This is a first-pass scaffold: the compute node currently only verifies
  algebraic equalities via `sympy.simplify`. Extend it as the kinds of
  derivations in the paper require more (limits, series, etc.).
- Citation checks are LLM-judgment based (the critics are told the rubric),
  not a hard bibtex parser. For a stricter pass, consider adding a dedicated
  `.bib`-parsing tool later.
