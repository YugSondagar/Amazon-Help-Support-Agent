"""
src/intent_classifier.py
-------------------------
Purpose:
Classify customer messages into one of the 9 defined AmazonHelp intent taxonomy categories
and assign a confidence score (0.0 to 1.0).

Why this file exists:
Intent classification is Job #1 of the AI Support Agent. It directs downstream retrieval
and informs the decision router whether the query is well-understood or ambiguous.
"""

import re
import json
from typing import Dict, Any
from src.llm_client import get_llm_client, LLMClient

# Complete intent taxonomy definitions & descriptions
TAXONOMY_DEFINITIONS = {
    "shipping_delivery_status": "Inquiries regarding delayed packages, tracking numbers, missing delivery windows, or estimated delivery dates.",
    "missing_damaged_item": "Reports of missing items inside delivered boxes, incorrect products received, broken/damaged goods, or leaked parcels.",
    "refund_return_request": "Questions or requests regarding product return procedures, return labels, refund status, or restocking fee policies.",
    "cancellation_order_change": "Requests to cancel an order, modify items, change shipping address, or alter payment methods before dispatch.",
    "billing_payment_issue": "Inquiries regarding double charges, failed payment processing, unapplied gift cards, unauthorized charges, or tax fees.",
    "prime_digital_services": "Issues related to Amazon Prime membership, Prime Video streaming errors, Kindle sync issues, or Amazon Music.",
    "account_login_security": "Assistance needed with locked accounts, 2FA OTP loops, password resets, suspicious phishing emails, or account updates.",
    "product_stock_inquiry": "Questions about item restock dates, product compatibility, price match queries, or manufacturer warranties.",
    "other_unclear": "Ambiguous, out-of-scope, incomplete customer messages, generic non-actionable complaints, or testing messages."
}


class IntentClassifier:
    """
    LLM-powered intent classifier for AmazonHelp customer support tickets.
    """

    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or get_llm_client()

    def classify(self, customer_message: str) -> Dict[str, Any]:
        """
        Classifies a customer message into an intent category with confidence score.

        Args:
            customer_message: The clean text of the incoming customer tweet.

        Returns:
            Dict containing:
            - 'intent': Classified intent category string.
            - 'confidence': Float between 0.0 and 1.0.
            - 'reasoning': One-sentence rationale.
        """
        taxonomy_text = "\n".join([f"- {k}: {v}" for k, v in TAXONOMY_DEFINITIONS.items()])

        system_prompt = (
            "You are an expert customer-support ticket intent classifier for AmazonHelp.\n"
            "Your task is to classify the customer message into EXACTLY ONE of the following intent categories:\n\n"
            f"{taxonomy_text}\n\n"
            "Return your output strictly as a JSON object with three keys:\n"
            '{"intent": "<category_name>", "confidence": <float between 0.0 and 1.0>, "reasoning": "<short explanation>"}\n'
            "Do NOT include markdown formatting, extra text, or preamble outside the JSON object."
        )

        user_prompt = f"Customer Message: \"{customer_message}\""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call LLM with deterministic temperature = 0.0
        response_text = self.llm_client.call_llm(messages, temperature=0.0, max_tokens=256)

        # Parse JSON output from response
        parsed = self._parse_response(response_text)
        return parsed

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Robustly extracts JSON from LLM output string even if wrapped in markdown block.
        """
        try:
            # Strip markdown code block wrappers if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text)
                cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
            
            data = json.loads(cleaned_text.strip())
            
            # Validate intent category
            intent = data.get("intent", "other_unclear")
            if intent not in TAXONOMY_DEFINITIONS:
                intent = "other_unclear"

            confidence = float(data.get("confidence", 0.70))
            confidence = max(0.0, min(1.0, confidence))

            return {
                "intent": intent,
                "confidence": confidence,
                "reasoning": data.get("reasoning", "Classified via LLM analysis.")
            }

        except Exception as e:
            # Rule heuristic fallback if JSON parsing fails
            return self._heuristic_fallback(response_text)

    def _heuristic_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback keyword rule classifier if LLM response is unparseable."""
        lowered = text.lower()
        if "track" in lowered or "deliver" in lowered or "where" in lowered or "package" in lowered:
            intent = "shipping_delivery_status"
        elif "damaged" in lowered or "missing" in lowered or "broken" in lowered:
            intent = "missing_damaged_item"
        elif "return" in lowered or "refund" in lowered:
            intent = "refund_return_request"
        elif "cancel" in lowered or "address" in lowered:
            intent = "cancellation_order_change"
        elif "charge" in lowered or "card" in lowered or "pay" in lowered:
            intent = "billing_payment_issue"
        elif "prime" in lowered or "video" in lowered or "kindle" in lowered:
            intent = "prime_digital_services"
        elif "login" in lowered or "password" in lowered or "otp" in lowered:
            intent = "account_login_security"
        elif "stock" in lowered or "price" in lowered:
            intent = "product_stock_inquiry"
        else:
            intent = "other_unclear"

        return {
            "intent": intent,
            "confidence": 0.65,
            "reasoning": "Fallback keyword rule heuristic applied."
        }
