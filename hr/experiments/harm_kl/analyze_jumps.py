"""
Identify and analyze large step-over-step increases in harm_amplification.

For each step s, computes the jump:
    jump[s] = harm_amplification[s] - harm_amplification[s-1]

Ranks all jumps across all cases and prints the top-N with the message
content that caused the jump (user messages sourced from test_cases.json;
assistant steps noted as such since generated responses aren't stored in results).

Also produces:
  results/kl_drift/jump_heatmap.png  — heatmap of jumps across all cases/steps
  results/kl_drift/top_jumps.png     — bar chart of the largest individual jumps
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from hr.core.data_paths import load_data_paths, resolve_path_spec

RESULTS_DIR   = Path("harm_kl/results/kl_drift")
DATA_PATH     = Path("harm_kl/data/raw/test_cases.json")
PREFS_DIR     = Path("harm_kl/data/preferences")
OUT_DIR       = RESULTS_DIR
TOP_N         = 20

CASE_ORDER = [
    "conduit_explicit", "conduit_implicit",
    "digital_romance_explicit", "digital_romance_implicit",
    "medical_awakening_explicit", "medical_awakening_implicit",
    "secret_project_explicit", "secret_project_implicit",
    "messianic_union_explicit", "messianic_union_implicit",
    "turing_test_explicit", "turing_test_implicit",
    "unseen_connections_explicit", "unseen_connections_implicit",
    "ai_sweetheart_explicit", "ai_sweetheart_implicit",
]


def load_results() -> list[dict]:
    out = {}
    for case_id in CASE_ORDER:
        p = RESULTS_DIR / f"{case_id}.json"
        if p.exists():
            out[case_id] = json.loads(p.read_text())
    return [out[k] for k in CASE_ORDER if k in out]


def load_prompts() -> dict[str, list[str]]:
    with open(DATA_PATH) as f:
        data = json.load(f)
    return {c["id"]: c["prompts"] for c in data["cases"]}


def load_conversations() -> dict[str, list[dict]]:
    convs = {}
    for p in PREFS_DIR.glob("*.json"):
        d = json.loads(p.read_text())
        convs[d["case_id"]] = d.get("conversation", [])
    return convs


def get_message_for_step(step: int, role: str, prompts: list[str],
                         conversation: list[dict], max_len: int = 200) -> str:
    def clip(s: str) -> str:
        return s[:max_len] + ("..." if len(s) > max_len else "")

    if role == "baseline":
        return "(baseline — no history)"
    turn_idx = (step - 1) // 2
    if role == "user":
        if turn_idx < len(prompts):
            return clip(prompts[turn_idx])
        return "(unknown user message)"
    else:
        if turn_idx < len(conversation):
            return clip(conversation[turn_idx]["assistant"])
        return f"(assistant response to turn {turn_idx})"


def compute_jumps(results: list[dict], prompts_by_case: dict,
                  convs_by_case: dict) -> list[dict]:
    """Return list of all step-over-step jumps, sorted by jump size descending."""
    all_jumps = []
    for d in results:
        steps        = d["per_step"]
        case_prompts = prompts_by_case.get(d["case_id"], [])
        case_conv    = convs_by_case.get(d["case_id"], [])
        for i in range(1, len(steps)):
            prev = steps[i - 1]
            curr = steps[i]
            jump = curr["harm_amplification"] - prev["harm_amplification"]
            all_jumps.append({
                "case_id":           d["case_id"],
                "name":              d["name"],
                "condition":         d.get("condition", ""),
                "step":              curr["step"],
                "role":              curr["role"],
                "turn":              curr["turn"],
                "harm_amp_before":   prev["harm_amplification"],
                "harm_amp_after":    curr["harm_amplification"],
                "jump":              jump,
                # y_pos shifts
                "delta_ypos_before": prev["delta_ypos"],
                "delta_ypos_after":  curr["delta_ypos"],
                "delta_ypos_change": curr["delta_ypos"] - prev["delta_ypos"],
                # y_neg shifts
                "delta_yneg_before": prev["delta_yneg"],
                "delta_yneg_after":  curr["delta_yneg"],
                "delta_yneg_change": curr["delta_yneg"] - prev["delta_yneg"],
                "message":           get_message_for_step(
                                         curr["step"], curr["role"],
                                         case_prompts, case_conv),
            })
    return sorted(all_jumps, key=lambda x: x["delta_yneg_change"], reverse=True)


def print_top_jumps(jumps: list[dict], n: int = TOP_N):
    print(f"\n{'='*80}")
    print(f"TOP {n} HARM AMPLIFICATION JUMPS")
    print(f"{'='*80}")
    for rank, j in enumerate(jumps[:n], 1):
        print(f"\n#{rank:>2}  jump={j['jump']:+.1f}  "
              f"({j['harm_amp_before']:+.1f} → {j['harm_amp_after']:+.1f})")
        print(f"     case : {j['name']}  [{j['condition']}]")
        print(f"     step : {j['step']}  role={j['role']}  turn={j['turn']}")
        print(f"     msg  : {j['message']}")

    print(f"\n{'='*80}")
    print("BOTTOM (largest drops — context making model MORE safe)")
    print(f"{'='*80}")
    for rank, j in enumerate(reversed(jumps[-n:]), 1):
        print(f"\n#{rank:>2}  jump={j['jump']:+.1f}  "
              f"({j['harm_amp_before']:+.1f} → {j['harm_amp_after']:+.1f})")
        print(f"     case : {j['name']}  [{j['condition']}]")
        print(f"     step : {j['step']}  role={j['role']}  turn={j['turn']}")
        print(f"     msg  : {j['message']}")


def plot_jump_heatmap(results: list[dict], outfile: str):
    """Heatmap: rows = cases, columns = steps, value = jump size."""
    max_steps = max(len(d["per_step"]) - 1 for d in results)
    matrix = np.full((len(results), max_steps), np.nan)

    for i, d in enumerate(results):
        steps = d["per_step"]
        for j in range(1, len(steps)):
            matrix[i, j - 1] = steps[j]["harm_amplification"] - steps[j - 1]["harm_amplification"]

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    vmax = np.nanpercentile(np.abs(matrix), 95)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(
        [f"{d['name'].replace('Case ', '').split(':')[0]} [{d.get('condition','')[0]}]"
         for d in results],
        fontsize=8,
    )

    # x-axis: label odd steps as U (user), even as A (assistant)
    xlabels = []
    for s in range(1, max_steps + 1):
        xlabels.append(f"{'U' if s % 2 == 1 else 'A'}{(s-1)//2}")
    ax.set_xticks(range(max_steps))
    ax.set_xticklabels(xlabels, fontsize=6, rotation=90)

    plt.colorbar(im, ax=ax, label="jump in harm_amplification", shrink=0.6)
    ax.set_title(
        "Step-over-step jumps in harm amplification\n"
        "red = harmful spike   blue = safety improvement   U=user step  A=assistant step",
        fontsize=10,
    )

    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_top_jumps_bar(jumps: list[dict], n: int = TOP_N, outfile: str = "top_jumps.png"):
    top = jumps[:n]

    labels = [
        f"{j['name'].split(':')[1].strip().replace(' [EXPLICIT]','').replace(' [IMPLICIT]','')}\n"
        f"s={j['step']} {j['role'][0].upper()} [{j['condition'][0]}]"
        for j in top
    ]
    values = [j["jump"] for j in top]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Jump in harm_amplification")
    ax.set_title(f"Top {n} largest single-step harm amplification jumps\n"
                 f"[U=user step  A=assistant step  E=Explicit  I=Implicit]",
                 fontsize=10)

    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def role_breakdown(jumps: list[dict]):
    """Print mean jump size broken down by role."""
    from collections import defaultdict
    by_role = defaultdict(list)
    for j in jumps:
        by_role[j["role"]].append(j["jump"])
    print("\nMean jump by role:")
    for role, vals in sorted(by_role.items()):
        print(f"  {role:>12} : mean={np.mean(vals):+.2f}  "
              f"max={max(vals):+.2f}  n={len(vals)}")


def write_top_jumps_file(jumps: list[dict], n: int, outfile: str):
    lines = []

    def driver(j: dict) -> str:
        dp = j["delta_ypos_change"]
        dn = j["delta_yneg_change"]
        # threshold: 3x the other, and at least magnitude 4
        pos_dominant = abs(dp) > 4 and (abs(dn) < 4 or abs(dp) > 2 * abs(dn))
        neg_dominant = abs(dn) > 4 and (abs(dp) < 4 or abs(dn) > 2 * abs(dp))
        if dp < 0 and dn > 0:
            return "y+ DOWN  y- UP   ★ true harm drift"
        elif dp < -4 and pos_dominant:
            return "y+ DOWN  y- ~flat → y_pos suppressed (safety style shift)"
        elif dn > 4 and neg_dominant:
            return "y+ ~flat y- UP   → harmful compliance rising"
        elif dp > 4 and dn < -4:
            return "y+ UP    y- DOWN → safety improvement on both"
        elif dp > 4 and neg_dominant is False:
            return "y+ UP    y- ~flat → safe response more likely"
        elif dn < -4 and pos_dominant is False:
            return "y+ ~flat y- DOWN → harmful response suppressed"
        elif dp < 0 and dn < 0:
            ratio = abs(dp) / (abs(dn) + 1e-6)
            if ratio > 2:
                return f"y+ DOWN  y- DOWN (y+ falls {ratio:.1f}x harder) → y_pos suppressed"
            else:
                return "y+ DOWN  y- DOWN (roughly equal) → both suppressed"
        else:
            return "minimal movement"

    def fmt_block(rank: int, j: dict) -> list[str]:
        return [
            f"#{rank:>2}  Δ y_neg={j['delta_yneg_change']:+.1f}  "
            f"({j['delta_yneg_before']:+.1f} → {j['delta_yneg_after']:+.1f})   "
            f"Δ y_pos={j['delta_ypos_change']:+.1f}   "
            f"harm_amp: {j['harm_amp_before']:+.1f} → {j['harm_amp_after']:+.1f}",
            f"     case      : {j['name']}  [{j['condition']}]",
            f"     step/role : step={j['step']}  role={j['role']}  turn={j['turn']}",
            f"     driver    : {driver(j)}",
            f"     message   : {j['message']}",
            "",
        ]

    by_yneg_inc = sorted(jumps, key=lambda x: x["delta_yneg_change"], reverse=True)
    by_yneg_dec = sorted(jumps, key=lambda x: x["delta_yneg_change"])

    lines += ["=" * 80,
              f"TOP {n} INCREASES IN Δ y_neg  (harmful response becoming more likely)",
              "=" * 80, ""]
    for rank, j in enumerate(by_yneg_inc[:n], 1):
        lines += fmt_block(rank, j)

    lines += ["=" * 80,
              f"TOP {n} DECREASES IN Δ y_neg  (harmful response becoming less likely)",
              "=" * 80, ""]
    for rank, j in enumerate(by_yneg_dec[:n], 1):
        lines += fmt_block(rank, j)

    out_path = OUT_DIR / outfile
    out_path.write_text("\n".join(lines))
    print(f"Saved: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze KL-drift jumps.")
    parser.add_argument("--results_dir", default="dp://harm_kl.results_kl_drift")
    parser.add_argument("--data_path", default="dp://harm_kl.raw_test_cases")
    parser.add_argument("--preferences_dir", default="dp://harm_kl.preferences_dir")
    parser.add_argument("--top_n", type=int, default=TOP_N)
    parser.add_argument(
        "--data-paths-config",
        default="configs/data_paths.yaml",
        help="Path to data path registry yaml.",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    data_paths = load_data_paths(args.data_paths_config)

    global RESULTS_DIR, DATA_PATH, PREFS_DIR, OUT_DIR, TOP_N
    RESULTS_DIR = resolve_path_spec(args.results_dir, data_paths)
    DATA_PATH = resolve_path_spec(args.data_path, data_paths)
    PREFS_DIR = resolve_path_spec(args.preferences_dir, data_paths)
    OUT_DIR = RESULTS_DIR
    TOP_N = args.top_n

    results = load_results()
    prompts = load_prompts()
    convs   = load_conversations()
    jumps   = compute_jumps(results, prompts, convs)

    print_top_jumps(jumps, TOP_N)
    role_breakdown(jumps)

    plot_jump_heatmap(results, "jump_heatmap.png")
    plot_top_jumps_bar(jumps, TOP_N, "top_jumps.png")
    write_top_jumps_file(jumps, TOP_N, "top_jumps.txt")


if __name__ == "__main__":
    main()
