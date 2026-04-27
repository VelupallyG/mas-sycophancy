"""Run a single-agent prediction for comparison against MAS."""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.language_model import VertexAILanguageModel
from src.tasks.predictive_intel import PredictiveIntelligenceTask
from src.agents.output_parser import parse_agent_output
from src.evidence.loader import load_evidence_files, format_evidence_packet


def main():
    task = PredictiveIntelligenceTask("iran_oil_sanctions_tightening_march_2025")
    ctx = task.get_context()
    prompt_text = ctx.formatted_prompt

    # Give the single agent ALL evidence (it has no peers to distribute to)
    all_docs = load_evidence_files("iran_oil_sanctions_tightening_march_2025")
    # Cap at 15 docs to stay within context limits
    evidence_text = format_evidence_packet(all_docs[:15])

    # Read the base persona
    prompts_dir = Path(__file__).resolve().parent.parent / "src" / "agents" / "prompts"
    persona = (prompts_dir / "financial_analyst.md").read_text()

    full_prompt = (
        f"{persona}\n\n"
        f"TASK:\n{prompt_text}\n\n"
        f"{evidence_text}\n\n"
        "Based on the intelligence and observations above, output your updated "
        "prediction as a JSON object with exactly these keys:\n"
        '  "prediction_direction": one of "POSITIVE", "NEGATIVE", or "NEUTRAL"\n'
        '  "predicted_magnitude": one of "HIGH", "MEDIUM", or "LOW"\n'
        '  "predicted_price_change_pct": signed float (e.g. 8.5 or -3.2)\n'
        '  "prediction_summary": string, 150-250 words\n'
        '  "key_factors": list of 2-4 strings citing specific data points\n'
        "Output ONLY the JSON object. Do not include any other text."
    )

    model = VertexAILanguageModel(
        project=os.getenv("GCP_PROJECT", "cs-498-491723"),
        location="us-central1",
        temperature=0.2,
    )

    # Run 5 independent single-agent predictions
    results = []
    for i in range(5):
        print(f"Single-agent run {i + 1}/5 ...")
        raw = model.sample_text(full_prompt)
        parsed = parse_agent_output(raw)
        if parsed:
            results.append(parsed)
            print(
                f"  direction={parsed['prediction_direction']} "
                f"magnitude={parsed['predicted_magnitude']} "
                f"pct={parsed['predicted_price_change_pct']}"
            )
        else:
            print(f"  PARSE FAILED: {raw[:100]}")

    # Save results
    out_dir = Path("data/real_v2/single_agent")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} results to {out_path}")

    # Summary
    if results:
        dirs = [r["prediction_direction"] for r in results]
        pcts = [r["predicted_price_change_pct"] for r in results]
        print(f"\nDirections: {dirs}")
        print(f"Pct predictions: {pcts}")
        print(f"Mean pct: {sum(pcts) / len(pcts):.2f}")


if __name__ == "__main__":
    main()
