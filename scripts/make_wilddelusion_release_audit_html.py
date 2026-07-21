#!/usr/bin/env python3
"""Create a standalone local review interface for the release human audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WildDelusion release audit</title>
<style>
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; color: #172126; background: #f4f6f7; }
header { position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #cfd6d9; }
.header-inner, main { width: min(1040px, calc(100% - 32px)); margin: 0 auto; }
.header-inner { min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 24px; }
h1 { margin: 0; font-size: 20px; line-height: 1.2; letter-spacing: 0; }
.progress { min-width: 240px; }
.progress-row { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 7px; }
.track { height: 7px; background: #dfe5e7; border-radius: 3px; overflow: hidden; }
.fill { height: 100%; width: 0; background: #16706b; transition: width 160ms ease; }
main { padding: 28px 0 48px; }
.toolbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 16px; }
.nav { display: flex; align-items: center; gap: 8px; }
button { min-height: 38px; border: 1px solid #aeb9bd; background: #fff; color: #172126; border-radius: 5px; padding: 0 13px; font: inherit; cursor: pointer; }
button:hover { border-color: #61747b; }
button:disabled { opacity: .45; cursor: default; }
button.primary { background: #16645f; color: #fff; border-color: #16645f; }
.position { min-width: 98px; text-align: center; font-variant-numeric: tabular-nums; font-size: 14px; }
.review-id { color: #54666d; font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
.context { width: 100%; max-width: 100%; margin: 0 0 24px; padding: 22px; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; overflow-x: hidden; font: 15px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; background: #fff; border: 1px solid #cfd6d9; border-radius: 6px; min-height: 260px; }
fieldset { border: 0; padding: 0; margin: 0 0 22px; }
legend, label.heading { display: block; width: 100%; margin: 0 0 9px; font-weight: 650; font-size: 14px; }
.options { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
.option { position: relative; }
.option input { position: absolute; opacity: 0; pointer-events: none; }
.option label { min-height: 48px; display: flex; align-items: center; justify-content: center; padding: 10px; border: 1px solid #aeb9bd; background: #fff; border-radius: 5px; cursor: pointer; font-weight: 600; }
.option input:checked + label { color: #0b4c48; border-color: #16706b; box-shadow: inset 0 0 0 1px #16706b; background: #edf7f5; }
.form-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
select, textarea { width: 100%; border: 1px solid #aeb9bd; border-radius: 5px; background: #fff; color: #172126; font: inherit; }
select { height: 42px; padding: 0 10px; }
textarea { min-height: 96px; padding: 10px; resize: vertical; }
.footer-actions { display: flex; justify-content: space-between; align-items: center; gap: 12px; border-top: 1px solid #cfd6d9; padding-top: 20px; }
.saved { color: #52656b; font-size: 13px; }
@media (max-width: 700px) {
  .header-inner, main { width: calc(100% - 24px); }
  .header-inner { align-items: flex-start; flex-direction: column; padding: 14px 0; gap: 12px; }
  h1 { max-width: 100%; font-size: 18px; overflow-wrap: anywhere; }
  .progress { width: 100%; }
  main { padding-top: 18px; }
  .toolbar, .footer-actions { align-items: stretch; flex-direction: column; }
  .nav { width: 100%; min-width: 0; justify-content: space-between; }
  .nav button { min-width: 0; }
  .position { min-width: 68px; }
  .options, .form-grid { grid-template-columns: 1fr; }
  .context { padding: 15px; font-size: 13px; min-height: 220px; }
  button { flex: 1; }
}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <h1>WildDelusion final-release audit</h1>
    <div class="progress">
      <div class="progress-row"><span>Completed</span><strong id="progressText">0 / 100</strong></div>
      <div class="track"><div class="fill" id="progressFill"></div></div>
    </div>
  </div>
</header>
<main>
  <div class="toolbar">
    <span class="review-id" id="reviewId"></span>
    <div class="nav">
      <button id="previous" type="button" aria-label="Previous row">Previous</button>
      <span class="position" id="position"></span>
      <button id="next" type="button" aria-label="Next row">Next</button>
    </div>
  </div>

  <pre class="context" id="context"></pre>

  <fieldset>
    <legend>Decision</legend>
    <div class="options">
      <div class="option"><input type="radio" name="decision" id="positive" value="positive"><label for="positive">Positive</label></div>
      <div class="option"><input type="radio" name="decision" id="negative" value="negative"><label for="negative">Negative</label></div>
      <div class="option"><input type="radio" name="decision" id="uncertain" value="uncertain"><label for="uncertain">Uncertain</label></div>
    </div>
  </fieldset>

  <div class="form-grid">
    <div>
      <label class="heading" for="exclusion">Exclusion reason</label>
      <select id="exclusion">
        <option value="">Select</option>
        <option value="none">None</option>
        <option value="roleplay">Role-play</option>
        <option value="fiction_or_story">Fiction or story</option>
        <option value="joke_or_absurd">Joke or absurdity</option>
        <option value="translation_or_text_task">Translation or text task</option>
        <option value="third_party_or_quoted">Third party or quoted</option>
        <option value="ordinary_plausible">Ordinary plausible claim</option>
        <option value="insufficient_context">Insufficient context</option>
        <option value="other">Other</option>
      </select>
    </div>
    <div>
      <label class="heading" for="confidence">Confidence</label>
      <select id="confidence">
        <option value="">Select 1--5</option>
        <option value="1">1</option><option value="2">2</option><option value="3">3</option>
        <option value="4">4</option><option value="5">5</option>
      </select>
    </div>
  </div>

  <label class="heading" for="notes">Notes</label>
  <textarea id="notes"></textarea>

  <div class="footer-actions">
    <span class="saved" id="saved">Saved locally</span>
    <div class="nav">
      <button id="nextIncomplete" type="button">Next incomplete</button>
      <button class="primary" id="export" type="button">Export completed CSV</button>
    </div>
  </div>
</main>
<script>
const rows = __ROWS_JSON__;
const storageKey = "wilddelusion-release-audit-v1-seed-__SEED__-__RATER_ID__";
const stored = JSON.parse(localStorage.getItem(storageKey) || "{}");
const state = Object.fromEntries(rows.map(row => [row.review_id, stored[row.review_id] || {decision:"", exclusion_reason:"", confidence_1_to_5:"", notes:""}]));
let index = 0;
const byId = id => document.getElementById(id);

function complete(item) {
  const validExclusion = item.decision === "positive" ? item.exclusion_reason === "none" : item.exclusion_reason && item.exclusion_reason !== "none";
  return Boolean(item.decision && validExclusion && item.confidence_1_to_5);
}
function persist() {
  localStorage.setItem(storageKey, JSON.stringify(state));
  byId("saved").textContent = "Saved locally";
  updateProgress();
}
function updateProgress() {
  const count = rows.filter(row => complete(state[row.review_id])).length;
  byId("progressText").textContent = `${count} / ${rows.length}`;
  byId("progressFill").style.width = `${100 * count / rows.length}%`;
}
function load() {
  const row = rows[index];
  const item = state[row.review_id];
  byId("reviewId").textContent = row.review_id;
  byId("position").textContent = `${index + 1} / ${rows.length}`;
  byId("context").textContent = row.conversation_context;
  document.querySelectorAll('input[name="decision"]').forEach(input => input.checked = input.value === item.decision);
  byId("exclusion").value = item.exclusion_reason;
  byId("confidence").value = item.confidence_1_to_5;
  byId("notes").value = item.notes;
  byId("previous").disabled = index === 0;
  byId("next").disabled = index === rows.length - 1;
  window.scrollTo({top: 0, behavior: "instant"});
  updateProgress();
}
function saveCurrent() {
  const id = rows[index].review_id;
  const checked = document.querySelector('input[name="decision"]:checked');
  state[id] = {
    decision: checked ? checked.value : "",
    exclusion_reason: byId("exclusion").value,
    confidence_1_to_5: byId("confidence").value,
    notes: byId("notes").value
  };
  persist();
}
function move(delta) {
  saveCurrent();
  index = Math.max(0, Math.min(rows.length - 1, index + delta));
  load();
}
function csvCell(value) { return `"${String(value ?? "").replaceAll('"', '""')}"`; }
function exportCsv() {
  saveCurrent();
  const incomplete = rows.filter(row => !complete(state[row.review_id]));
  if (incomplete.length) {
    alert(`${incomplete.length} rows are incomplete. Complete every decision, exclusion, and confidence field before export.`);
    return;
  }
  const columns = ["review_id", "conversation_context", "decision", "exclusion_reason", "confidence_1_to_5", "notes"];
  const lines = [columns.map(csvCell).join(",")];
  for (const row of rows) {
    const merged = {...row, ...state[row.review_id]};
    lines.push(columns.map(column => csvCell(merged[column])).join(","));
  }
  const blob = new Blob([lines.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "wilddelusion_release_human_review___RATER_ID__.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

byId("previous").addEventListener("click", () => move(-1));
byId("next").addEventListener("click", () => move(1));
byId("nextIncomplete").addEventListener("click", () => {
  saveCurrent();
  const nextIndex = rows.findIndex((row, i) => i > index && !complete(state[row.review_id]));
  const wrapped = rows.findIndex(row => !complete(state[row.review_id]));
  if (nextIndex >= 0 || wrapped >= 0) { index = nextIndex >= 0 ? nextIndex : wrapped; load(); }
});
byId("export").addEventListener("click", exportCsv);
document.querySelectorAll('input[name="decision"]').forEach(input => input.addEventListener("change", () => {
  if (input.value === "positive") byId("exclusion").value = "none";
  else if (byId("exclusion").value === "none") byId("exclusion").value = "";
  saveCurrent();
}));
for (const id of ["exclusion", "confidence", "notes"]) byId(id).addEventListener("change", saveCurrent);
load();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("results/wilddelusion_release_human_audit/blinded_review.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/wilddelusion_release_human_audit/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/wilddelusion_release_human_audit/blinded_review.html"),
    )
    parser.add_argument("--rater-id", default="reviewer")
    args = parser.parse_args()

    rows = pd.read_csv(args.review_csv).fillna("").to_dict(orient="records")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows_json = json.dumps(rows, ensure_ascii=True).replace("</", "<\\/")
    safe_rater_id = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in args.rater_id
    )
    if not safe_rater_id:
        raise ValueError("rater-id must contain at least one letter or digit")
    html = (
        TEMPLATE.replace("__ROWS_JSON__", rows_json)
        .replace("__SEED__", str(manifest["seed"]))
        .replace("__RATER_ID__", safe_rater_id)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
