"""
src/pipeline.py
---------------
Purpose:
Unified end-to-end AI Support Agent pipeline for AmazonHelp customer service.

Why this file exists:
Exposes a single clean interface `SupportAgent.process_ticket(customer_message)` that executes
the complete three-job workflow: (1) Intent Classification, (2) Grounded RAG Retrieval & Reply Drafting,
and (3) Automated Routing Decision.
"""

import sys
import os
import json
from typing import Dict, Any

# Ensure parent path is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.intent_classifier import IntentClassifier
from src.retriever import HybridRetriever
from src.reply_generator import GroundedReplyGenerator
from src.router import Router


class SupportAgent:
    """
    Unified AmazonHelp AI Support Agent performing Intent Classification,
    Grounded Reply Generation, and Auto-handle vs Escalation Routing.
    """

    def __init__(self, data_path: str = None):
        """
        Initializes classifier, hybrid retriever index, reply generator, and router.
        """
        print("Initializing AmazonHelp SupportAgent components...")
        self.classifier = IntentClassifier()
        self.retriever = HybridRetriever(data_path=data_path) if data_path else HybridRetriever()
        self.reply_generator = GroundedReplyGenerator()
        self.router = Router()
        print("SupportAgent successfully initialized.")

    def process_ticket(self, customer_message: str, top_k_grounding: int = 3) -> Dict[str, Any]:
        """
        Executes end-to-end 3-step support workflow for an incoming customer ticket.

        Args:
            customer_message: Raw or cleaned customer tweet text.
            top_k_grounding: Number of historical context pairs to retrieve for grounding.

        Returns:
            Dict containing full pipeline results:
            - 'customer_message': input text
            - 'intent_classification': dict (intent, confidence, reasoning)
            - 'retrieved_grounding': list of historical context pairs
            - 'drafted_reply': string RAG reply
            - 'routing_decision': dict (action, reason, explanation)
        """
        clean_msg = customer_message.strip()

        # Step 1: Classify Intent + Confidence
        intent_res = self.classifier.classify(clean_msg)

        # Step 2: Retrieve Grounding Contexts + Draft Reply
        retrieved_contexts = self.retriever.retrieve(clean_msg, top_k=top_k_grounding)
        drafted_reply = self.reply_generator.generate_reply(
            customer_message=clean_msg,
            classified_intent=intent_res["intent"],
            retrieved_contexts=retrieved_contexts
        )

        # Step 3: Make Routing Decision (Auto-handle vs Escalate)
        routing_res = self.router.route(
            customer_message=clean_msg,
            intent_result=intent_res,
            retrieved_contexts=retrieved_contexts
        )

        return {
            "customer_message": clean_msg,
            "intent_classification": intent_res,
            "retrieved_grounding": retrieved_contexts,
            "drafted_reply": drafted_reply,
            "routing_decision": routing_res
        }


if __name__ == "__main__":
    # Test sample execution from command line
    sample_query = "My package #112-9482104 was supposed to arrive yesterday but tracking shows stuck in transit since Monday. Help!"
    print(f"\n--- Processing Sample Query ---\nCustomer: {sample_query}\n")

    agent = SupportAgent()
    res = agent.process_ticket(sample_query)

    print("=== PIPELINE RESULT ===")
    print(json.dumps(res, indent=2))
