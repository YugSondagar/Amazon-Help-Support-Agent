"""
src/router.py
-------------
Purpose:
Evaluates customer messages, classification metrics, and retrieval scores to decide whether
a support ticket can be auto-handled by the AI agent or must be escalated to a human support agent.

Why this file exists:
Automation without safety guardrails is dangerous. The decision router enforces strict business logic
and machine-readable escalation reasons (e.g. low intent confidence, account security, angry customer).
"""

import re
from typing import Dict, Any, List

# Explicit machine-readable escalation reasons required by project specification:
# 1. low_confidence_intent
# 2. safety_sensitive
# 3. requires_account_access
# 4. angry_customer
# 5. novel_issue_no_precedent

SAFETY_SENSITIVE_KEYWORDS = [
    "password", "2fa", "otp", "hacked", "stolen", "security", "phishing", "scam", "suspicious", "lawyer", "sue", "legal"
]

ACCOUNT_ACTION_KEYWORDS = [
    "cancel order", "change address", "change shipping", "change payment", "update card", "refund my money", "delete account"
]

ANGRY_KEYWORDS = [
    "terrible", "worst", "horrible", "furious", "unacceptable", "lawsuit", "scam", "disgusted", "trash", "sucks"
]


class Router:
    """
    Decision engine determining auto-handle vs escalate with explicit machine-readable rationale.
    """

    def __init__(
        self,
        min_intent_confidence: float = 0.70,
        min_retrieval_similarity: float = 0.40
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

    def route(
        self,
        customer_message: str,
        intent_result: Dict[str, Any],
        retrieved_contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Decides routing strategy for an incoming customer ticket.

        Args:
            customer_message: incoming customer text.
            intent_result: dict from IntentClassifier with 'intent' and 'confidence'.
            retrieved_contexts: list of retrieved historical grounding contexts.

        Returns:
            Dict containing:
            - 'action': 'auto_handle' or 'escalate'
            - 'reason': Machine-readable reason tag (or 'none' if auto_handle)
            - 'explanation': Human-readable description of routing decision.
        """
        text_lower = customer_message.lower()

        # Check Rule 1: Safety Sensitive (Account security, legal threats, phishing)
        if any(kw in text_lower for kw in SAFETY_SENSITIVE_KEYWORDS) or intent_result.get("intent") == "account_login_security":
            return {
                "action": "escalate",
                "reason": "safety_sensitive",
                "explanation": "Ticket involves account security, identity verification, or legal sensitivity."
            }

        # Check Rule 2: Angry / Hostile Customer
        if any(kw in text_lower for kw in ANGRY_KEYWORDS):
            return {
                "action": "escalate",
                "reason": "angry_customer",
                "explanation": "High customer frustration or hostile sentiment detected; requires human empathy."
            }

        # Check Rule 3: Requires Direct Account Mutation Action
        if any(kw in text_lower for kw in ACCOUNT_ACTION_KEYWORDS):
            return {
                "action": "escalate",
                "reason": "requires_account_access",
                "explanation": "Request requires direct modification of account, address, or payment records."
            }

        # Check Rule 4: Low Intent Classification Confidence
        confidence = intent_result.get("confidence", 0.0)
        intent = intent_result.get("intent", "other_unclear")

        if confidence < self.min_intent_confidence or intent == "other_unclear":
            return {
                "action": "escalate",
                "reason": "low_confidence_intent",
                "explanation": f"Intent confidence ({confidence:.2f}) is below threshold ({self.min_intent_confidence}) or intent is unclear."
            }

        # Check Rule 5: Novel Issue with No Historical Precedent
        top_similarity = retrieved_contexts[0].get("similarity_score", 0.0) if retrieved_contexts else 0.0
        if top_similarity < self.min_retrieval_similarity:
            return {
                "action": "escalate",
                "reason": "novel_issue_no_precedent",
                "explanation": f"Top grounding similarity score ({top_similarity:.2f}) is below threshold ({self.min_retrieval_similarity})."
            }

        # If all checks pass -> Auto Handle
        return {
            "action": "auto_handle",
            "reason": "none",
            "explanation": f"Ticket is routine ({intent}), intent confidence is high ({confidence:.2f}), and grounding precedent exists."
        }
