"""
eval/baselines.py
-----------------
Purpose:
Implements two mandatory baseline agents (Trivial and Simple) to compare against the full SupportAgent.

Why this file exists:
Proving an AI agent requires showing that sophisticated components (LLMs, hybrid RAG) outperform
simple heuristics and trivial rules, avoiding misleading benchmark framing.
"""

from typing import Dict, Any, List
from src.retriever import HybridRetriever


class TrivialBaseline:
    """
    Trivial Baseline:
    - Intent: Majority class ('shipping_delivery_status')
    - Reply: Canned static template response
    - Routing: Always 'auto_handle'
    """

    def process_ticket(self, customer_message: str) -> Dict[str, Any]:
        return {
            "customer_message": customer_message,
            "intent_classification": {
                "intent": "shipping_delivery_status",
                "confidence": 0.50,
                "reasoning": "Trivial baseline majority class prediction."
            },
            "retrieved_grounding": [],
            "drafted_reply": "We're sorry for the inconvenience! Please DM us your order number so we can assist.",
            "routing_decision": {
                "action": "auto_handle",
                "reason": "none",
                "explanation": "Trivial baseline always auto-handles."
            }
        }


class SimpleBaseline:
    """
    Simple Baseline:
    - Intent: Keyword regex classifier
    - Reply: Raw top-1 nearest-neighbor historical reply (no LLM generation)
    - Routing: Rule threshold based on keyword checks
    """

    def __init__(self, data_path: str = None):
        self.retriever = HybridRetriever(data_path=data_path) if data_path else HybridRetriever()

    def process_ticket(self, customer_message: str) -> Dict[str, Any]:
        msg_lower = customer_message.lower()

        # Simple keyword intent classification
        if any(w in msg_lower for w in ["track", "ship", "deliver", "where", "package"]):
            intent = "shipping_delivery_status"
        elif any(w in msg_lower for w in ["damaged", "missing", "broken", "wrong"]):
            intent = "missing_damaged_item"
        elif any(w in msg_lower for w in ["return", "refund", "label"]):
            intent = "refund_return_request"
        elif any(w in msg_lower for w in ["cancel", "address"]):
            intent = "cancellation_order_change"
        elif any(w in msg_lower for w in ["charge", "card", "billing", "pay"]):
            intent = "billing_payment_issue"
        elif any(w in msg_lower for w in ["prime", "video", "kindle"]):
            intent = "prime_digital_services"
        elif any(w in msg_lower for w in ["password", "login", "otp", "2fa"]):
            intent = "account_login_security"
        elif any(w in msg_lower for w in ["stock", "price"]):
            intent = "product_stock_inquiry"
        else:
            intent = "other_unclear"

        # Direct Nearest-Neighbor Reply Retrieval (No LLM generation step)
        contexts = self.retriever.retrieve(customer_message, top_k=1)
        drafted_reply = contexts[0]["brand_reply"] if contexts else "Please send us a DM with your order number."

        # Simple Routing
        if intent in ["account_login_security", "cancellation_order_change"] or "cancel" in msg_lower or "password" in msg_lower:
            action = "escalate"
            reason = "requires_account_access" if "cancel" in msg_lower else "safety_sensitive"
        else:
            action = "auto_handle"
            reason = "none"

        return {
            "customer_message": customer_message,
            "intent_classification": {
                "intent": intent,
                "confidence": 0.75,
                "reasoning": "Keyword regex rule match."
            },
            "retrieved_grounding": contexts,
            "drafted_reply": drafted_reply,
            "routing_decision": {
                "action": action,
                "reason": reason,
                "explanation": "Simple rule-based routing."
            }
        }
