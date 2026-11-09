"""Entry point. Runs the Author/Critic loop:

  draft -> [every k rounds: Fresh critic, else: Stateful critic]
  -> reject: feedback goes back to Author for another round
  -> accept: return the draft

Usage:
    python orchestrator.py --input path/to/introduction.tex [--section introduction] [--k 3] [--max-rounds 12]
"""

import argparse
import asyncio
import sys
from pathlib import Path

from review_notes import VALID_SECTIONS
from author import draft_or_revise
from critics import StatefulCritic, fresh_critic_review


async def run_pipeline(
    problem_statement: str,
    section: str,
    k: int = 3,
    max_rounds: int = 12,
    review_dir: Path | None = None,
) -> str:
    stateful_critic = StatefulCritic(section)
    draft = await draft_or_revise(problem_statement, section)

    if review_dir:
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "rev_0_draft.tex").write_text(draft, encoding="utf-8")

    for round_num in range(1, max_rounds + 1):
        use_fresh = round_num % k == 0
        critic_name = "Fresh" if use_fresh else "Stateful"
        verdict = (
            await fresh_critic_review(draft, section)
            if use_fresh
            else await stateful_critic.review(draft)
        )

        status = "ACCEPT" if verdict.accepted else "REJECT"
        print(f"--- round {round_num} ({critic_name} critic): {status} ---", file=sys.stderr)
        if not verdict.accepted:
            print(verdict.feedback, file=sys.stderr)
            print("---", file=sys.stderr)

        if review_dir:
            (review_dir / f"rev_{round_num}_verdict_{critic_name.lower()}_{status.lower()}.txt"
             ).write_text(verdict.feedback, encoding="utf-8")

        if verdict.accepted:
            return draft

        revision_prompt = (
            f"The critic rejected your previous draft with this feedback:\n\n"
            f"{verdict.feedback}\n\n"
            f"Previous draft:\n\n{draft}\n\n"
            f"Revise the draft to address every point raised."
        )
        draft = await draft_or_revise(revision_prompt, section)

        if review_dir:
            (review_dir / f"rev_{round_num}_draft.tex").write_text(draft, encoding="utf-8")

    raise RuntimeError(f"Did not reach an accepted draft within {max_rounds} rounds.")


def main():
    parser = argparse.ArgumentParser(description="Author/Critic review pipeline")
    parser.add_argument("--input", required=True, help="Path to a .tex file with the "
                         "problem statement / section to draft or revise")
    parser.add_argument("--k", type=int, default=3, help="Fresh critic runs every k rounds")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--output", default=None, help="Where to write the accepted "
                         "draft. Defaults to '<input>_revised<suffix>' next to the "
                         "input file. Refuses to match --input, so the original is "
                         "never overwritten by this script.")
    parser.add_argument("--review-dir", default=None, help="Folder to write each "
                         "round's draft (rev_N_draft.tex) and critic verdict "
                         "(rev_N_verdict_<critic>_<accept|reject>.txt). Defaults to "
                         "'<input_stem>_review' next to the input file. Pass "
                         "--no-review-dir to skip writing these.")
    parser.add_argument("--no-review-dir", action="store_true", help="Skip writing "
                         "per-round drafts/verdicts entirely.")
    parser.add_argument("--section", default=None,
                         help=f"Which section this is, one of: {', '.join(sorted(VALID_SECTIONS))}. "
                         "Used to load the right paper-overview context and cross-section "
                         "review notes. Defaults to the input filename (lowercased), if it "
                         "matches one of the valid section names.")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.section:
        section = args.section.strip().lower()
    else:
        section = input_path.stem.strip().lower()
    if section not in VALID_SECTIONS:
        print(
            f"'{section}' is not a recognized section (from --section or the input "
            f"filename). Pass --section explicitly, one of: {', '.join(sorted(VALID_SECTIONS))}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_revised{input_path.suffix}")

    if args.no_review_dir:
        review_dir = None
    elif args.review_dir:
        review_dir = Path(args.review_dir)
    else:
        review_dir = input_path.with_name(f"{input_path.stem}_review")

    if output_path.resolve() == input_path.resolve():
        print(
            f"Refusing to run: --output resolves to the same file as --input "
            f"({input_path}). This script never overwrites the original.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        problem_statement = f.read()

    final_draft = asyncio.run(
        run_pipeline(
            problem_statement, section, k=args.k, max_rounds=args.max_rounds, review_dir=review_dir
        )
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_draft)
    print(f"Accepted draft written to {output_path}", file=sys.stderr)
    if review_dir:
        print(f"Per-round drafts and verdicts written to {review_dir}\\", file=sys.stderr)


if __name__ == "__main__":
    main()
