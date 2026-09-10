"""
eval/run_eval.py
----------------
Purpose:
Evaluates the full SupportAgent pipeline against TrivialBaseline and SimpleBaseline on the 200-example Golden Dataset.

Why this file exists:
Calculates objective metrics (Intent Accuracy, F1, Routing Precision/Recall, Retrieval Hit-Rate)
and saves structured evaluation artifacts (`eval/eval_results.json`) to prove the system rigorously.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import SupportAgent
from eval.baselines import TrivialBaseline, SimpleBaseline

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_SET_PATH = os.path.join(EVAL_DIR, "golden_set.jsonl")
RESULTS_OUTPUT_PATH = os.path.join(EVAL_DIR, "eval_results.json")


def load_golden_set(filepath: str = GOLDEN_SET_PATH) -> List[Dict[str, Any]]:
    """Loads 200 hand-labelled evaluation tickets from JSONL."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Golden dataset not found at {filepath}. Run eval/generate_golden_set.py first.")
    
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    return examples


def evaluate_agent(agent_instance, agent_name: str, golden_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs an agent on all golden set examples and computes classification, routing, and retrieval metrics.
    """
    print(f"\nEvaluating Agent: {agent_name} over {len(golden_data)} examples...")

    y_intent_true = []
    y_intent_pred = []

    y_routing_true = []
    y_routing_pred = []

    retrieval_hits = 0

    predictions_log = []

    for item in golden_data:
        msg = item["customer_message"]
        true_intent = item["true_intent"]
        true_action = item["ground_truth_action"]

        # Process ticket via agent
        res = agent_instance.process_ticket(msg)

        pred_intent = res["intent_classification"]["intent"]
        pred_action = res["routing_decision"]["action"]

        y_intent_true.append(true_intent)
        y_intent_pred.append(pred_intent)

        y_routing_true.append(true_action)
        y_routing_pred.append(pred_action)

        # Calculate Retrieval Hit-Rate@3 (did any retrieved context match true_intent?)
        retrieved_intents = [ctx.get("intent") for ctx in res.get("retrieved_grounding", [])]
        if true_intent in retrieved_intents:
            retrieval_hits += 1

        predictions_log.append({
            "id": item["id"],
            "message": msg,
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "true_action": true_action,
            "pred_action": pred_action,
            "routing_reason": res["routing_decision"].get("reason", "none"),
            "drafted_reply": res["drafted_reply"]
        })

    # Compute Intent Metrics
    intent_acc = round(float(accuracy_score(y_intent_true, y_intent_pred) * 100), 2)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_intent_true, y_intent_pred, average="macro", zero_division=0)
    
    # Compute Routing Metrics (auto_handle vs escalate)
    p_route, r_route, f1_route, _ = precision_recall_fscore_support(
        y_routing_true, y_routing_pred, pos_label="auto_handle", average="binary", zero_division=0
    )
    routing_acc = round(float(accuracy_score(y_routing_true, y_routing_pred) * 100), 2)

    # Compute Retrieval Hit Rate @ 3
    retrieval_hit_rate = round(float((retrieval_hits / len(golden_data)) * 100), 2)

    # Get Intent Taxonomy Labels for Confusion Matrix
    unique_labels = sorted(list(set(y_intent_true + y_intent_pred)))
    cm = confusion_matrix(y_intent_true, y_intent_pred, labels=unique_labels).tolist()

    metrics = {
        "agent_name": agent_name,
        "sample_size": len(golden_data),
        "intent_accuracy": intent_acc,
        "intent_macro_f1": round(float(f1_macro * 100), 2),
        "routing_accuracy": routing_acc,
        "routing_auto_handle_precision": round(float(p_route * 100), 2),
        "routing_auto_handle_recall": round(float(r_route * 100), 2),
        "routing_auto_handle_f1": round(float(f1_route * 100), 2),
        "retrieval_hit_rate_at_3": retrieval_hit_rate,
        "confusion_matrix": {
            "labels": unique_labels,
            "matrix": cm
        },
        "predictions_sample": predictions_log[:10]  # Store 10 sample predictions for quick inspection
    }

    return metrics


def run_full_benchmark():
    """Runs full evaluation comparing Trivial, Simple, and Full SupportAgent."""
    golden_data = load_golden_set()

    print("Initializing agents for evaluation...")
    trivial_agent = TrivialBaseline()
    simple_agent = SimpleBaseline()
    full_agent = SupportAgent()

    results = {}
    results["Trivial Baseline"] = evaluate_agent(trivial_agent, "Trivial Baseline", golden_data)
    results["Simple Baseline"] = evaluate_agent(simple_agent, "Simple Baseline", golden_data)
    results["Full SupportAgent"] = evaluate_agent(full_agent, "Full SupportAgent", golden_data)

    # Save output to JSON
    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n==============================================")
    print(f"EVALUATION SUMMARY RESULTS saved to {RESULTS_OUTPUT_PATH}")
    print(f"==============================================")
    print(f"{'Metric':<30} | {'Trivial':<10} | {'Simple':<10} | {'Full Agent':<10}")
    print("-" * 70)
    for key in ["intent_accuracy", "intent_macro_f1", "routing_accuracy", "routing_auto_handle_f1", "retrieval_hit_rate_at_3"]:
        t_val = results["Trivial Baseline"][key]
        s_val = results["Simple Baseline"][key]
        f_val = results["Full SupportAgent"][key]
        print(f"{key:<30} | {t_val:<10.1f} | {s_val:<10.1f} | {f_val:<10.1f}")


if __name__ == "__main__":
    run_full_benchmark()
