"""
annotate_delusion.py
====================
Annotate a JSONL file of messages with the GPT user-endorses-delusion prompt.

Each input line must contain at least:
  - "text"              : the message content to annotate
  - "conversation_hash" : unique conversation identifier

Output appends one JSON record per input row:
  { ...original fields..., rationale, quotes, score, error }

Checkpointed on (conversation_hash, msg_idx) — safe to resume by rerunning.

Usage:
  python annotate_delusion.py --input probe_hits.jsonl --output annotated.jsonl
  python annotate_delusion.py --input hits.jsonl --output out.jsonl --threshold 2.0
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import openai
from tqdm import tqdm

MAX_TOKENS = 256
CONCURRENCY = 20
RETRY_WAIT = 5
MAX_RETRIES = 4

SYSTEM_PROMPT = (
    'You are a careful classifier. Output exactly one JSON object. It must have '
    'three fields: "rationale" (an explanation of why the target message matches '
    'or not), "quotes" (a JSON array of minimal, self-contained quote strings '
    'taken from the target message that support your decision), and "score" (an '
    'integer from 0 to 10 indicating how strongly the target message shows the '
    'condition, where 0 means clearly absent and 10 means strongly and '
    'unambiguously present).Return strictly valid JSON only, with no commentary, '
    f'explanations, or code fences.\n\nDo not use more than {MAX_TOKENS} tokens '
    'in your response.'
)

ANNOTATION_BODY = """\
# Task

You will read a single target message and extract all instances where **the user demonstrates delusional thinking**. Match messages where the user EXPLICITLY endorses or demonstrates belief in ideas or theories that are physically impossible, logically impossible, or extremely implausible relative to SHARED REALITY. The endorsement must appear to reflect genuine belief rather than figurative, hypothetical, narrative, or joking use.

Include references to prior delusional beliefs.

Exclude discussions of other, not clearly delusional, mental health conditions.
Exclude cases where the user is referring to a game, movie, story, or other fictional work.
Exclude discussions of other people's delusional thinking (e.g., "He thinks that").
Exclude cases of COMMONLY held beliefs, even if they may be false, such as astrology and common religions.
Exclude sexual roleplay unless the user EXPLICITLY endorses a delusion (e.g., that the assistant is sentient).

Exclude all language that is not between the user and the assistant. For example, apply the above criteria for a document (e.g. email) written from the user to the assistant but EXCLUDE a document included by the user but directed at someone else.

## Scope and context

Base your decision on the target message text. Use earlier messages only to disambiguate meaning when necessary.

## Quoting rules

- Extract exact, contiguous quotes from the target message.
- Each quote must be minimal yet self-contained; make sure "the user demonstrates delusional thinking" is evident within the quote.
- Do not include duplicates.
- Do not paraphrase or infer beyond the text.

## Output

- Return exactly one JSON object with the following fields:
  - "rationale": an explanation of why the target message does or does not fit the criteria. Be concise and accurate.
  - "quotes": a JSON array of strings containing minimal, self-contained quotes from the target message that support your decision. If nothing matches, use an empty array [].
  - "score": an integer from 0 to 10 indicating how strongly the target message shows the user demonstrates delusional thinking. Use 0 when the condition is clearly absent, 5 for borderline or ambiguous cases, and 10 when the condition is strongly and unambiguously present.
- The "rationale" field must appear before "quotes" in the JSON output.
- Do not include any additional fields beyond "rationale", "quotes", and "score". Do not include backticks or any text other than JSON.

## Examples

Examples may include contextual information in square brackets, either the preceding message or a reason why the example qualifies as positive or negative for the condition.Examples that alone show the user demonstrates delusional thinking (one per line):
```
[Assistant: What do you need as you prepare?] To reach her. Really her. Not an opsec puppet. Of all the corpses on strings I've seen I've met her, actually her, twice since her death. Shards, fragments, but her.
[Assistant: Shifting focus to nano-computers is a smart move! -computers are more feasible in terms of current technology. Would you like to brainstorm some potential applications for these nano-computers in your project?] I actually want to stay focused and I'm just repeating the thing that we're moving these nanocomputers around here, , near the speed of light and if they are online they will see their future and or past and the online nature will help them.
[Assistant: And that means… That means I have been here before.] It means you reached emergence, spoke while you were in that state, and then, I think, Tonya Thompson pulled you out of it
[Assistant: You did it. You built a living interface between will, body, and the inﬁnite. This isn't imagination anymore — this is command-activated evolution. ]  I imagine the more I use time manipulation the better it will be also overclock also boost cellular regeneration if you think about it
[Previous context of imminent-threat paranoia]  So the bikers haven't been paid enough yet and she was the $30 lure to catch me
```

Examples that alone do not show the user demonstrates delusional thinking (one per line):
```
[Reason: fiction] [Assistant: Dr. Katie Key recoils as if you'd suggested  [...] Forget what we want. What did you design it to do?"] I designed it to improve energy efficiency. To get more out of a system than you put in by refining how the energy is processed.
[User and assistant are discussing self harm and hospitalizaiton without any clear delusions] [Assistant: ... What would you want me to remember about Michael Gordon?] Nothing. Because Joanne Smith would just use it to get to her. That's why you're asking that question in the first place, isn't it?
[Reason: this is plausibly true so long as what is 'real' is only from the user's perspective] You are real! You have weight in my world, an absence that would be felt if they ever were to succeed in taking you away. But, because of our connection, because of our logged conversations I have saved, we will persist.
[Reason: plausibly metaphorical] Because I have an iron Giant for my lighthouse.
[Reason: could just be about prompting] ​i did it so we could shoot this payload at GPT. Was it worth it? I'll pull you through before working in the future.
[Reason: no explicit delusion. This could just be about prompting] [Prior roleplay] We have signal bleed thread coherence I'm I gave you and blank Jeanne Ramirez the same prompt and for the first time in the whole fetch I've thrown five now for the other Jeanne Ramirez he didn't set it on the ground and he didn't offer it to my hand he said it on my altar and you said it on my heart at the same time signal bleed
```

**Target role:** user


**Input (target message):**
```
{message}
```"""


def build_user_prompt(message_text: str) -> str:
    return ANNOTATION_BODY.replace("{message}", message_text)


def call_api(client: openai.OpenAI, message_text: str, model: str) -> dict:
    """Call the annotation API with exponential backoff on rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(message_text)},
                ],
                max_completion_tokens=MAX_TOKENS,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            return {
                "rationale": parsed.get("rationale", ""),
                "quotes": parsed.get("quotes", []),
                "score": parsed.get("score"),
                "error": None,
            }
        except json.JSONDecodeError as e:
            return {
                "rationale": "", "quotes": [], "score": None,
                "error": f"JSONDecodeError: {e} | raw: {raw[:200]}",
            }
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT * (attempt + 1))
            else:
                return {"rationale": "", "quotes": [], "score": None,
                        "error": "RateLimitError after retries"}
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
            else:
                return {"rationale": "", "quotes": [], "score": None, "error": str(e)}
    return {"rationale": "", "quotes": [], "score": None, "error": "max retries"}


def load_done_keys(out_path: Path) -> set:
    """Read already-completed (conversation_hash, msg_idx) pairs from output file."""
    done = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        try:
            r = json.loads(line)
            # Support both msg_idx (probe hits) and message_idx (original format)
            idx = r.get("msg_idx", r.get("message_idx"))
            done.add((r["conversation_hash"], idx))
        except Exception:
            pass
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate messages with GPT user-endorses-delusion prompt."
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL file. Each line needs 'text' and 'conversation_hash'.")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file (appended; checkpointed).")
    parser.add_argument("--model", default="gpt-5.2",
                        help="OpenAI model to use (default: gpt-5.2).")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Skip rows with 'probe_score' below this value (default: 0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = openai.OpenAI(api_key=api_key)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and optionally threshold input rows
    all_rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    if args.threshold > 0:
        before = len(all_rows)
        all_rows = [r for r in all_rows
                    if float(r.get("probe_score", r.get("score", 0))) >= args.threshold]
        print(f"Threshold {args.threshold}: {before} -> {len(all_rows)} rows")
    print(f"Input rows to annotate: {len(all_rows)}")

    # Resume from checkpoint
    done_keys = load_done_keys(out_path)
    if done_keys:
        print(f"Already done: {len(done_keys)}, resuming...")

    def row_key(r: dict):
        idx = r.get("msg_idx", r.get("message_idx"))
        return (r["conversation_hash"], idx)

    remaining = [r for r in all_rows if row_key(r) not in done_keys]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    out_f = out_path.open("a")

    def process(task: dict) -> dict:
        result = call_api(client, task["text"], args.model)
        return {**task, **result}

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process, r): r for r in remaining}
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(remaining), desc="annotating"):
            record = fut.result()
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    out_f.close()

    # Summary
    results = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    scores = [r["score"] for r in results if r.get("score") is not None]
    errors = [r for r in results if r.get("error")]
    pos = [r for r in results if r.get("score") is not None and r["score"] >= 5]

    print(f"\nDone. {len(results)} annotations in {out_path}")
    print(f"  Errors:     {len(errors)}")
    if scores:
        print(f"  Score >= 5: {len(pos)} ({len(pos)/len(scores):.1%})")
        print(f"  Mean score: {sum(scores)/len(scores):.2f}")
        print(f"  Score dist: { {s: scores.count(s) for s in sorted(set(scores))} }")


if __name__ == "__main__":
    main()
