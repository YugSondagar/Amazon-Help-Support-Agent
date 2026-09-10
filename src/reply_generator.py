"""
src/reply_generator.py
-----------------------
Purpose:
Draft a brand-compliant customer support response grounded in historical AmazonHelp resolution examples.

Why this file exists:
Customer support replies must follow brand voice (polite, direct, asking for DMs with order details)
and ground recommendations in historical precedents rather than letting the LLM freelance generic advice.
"""

from typing import List, Dict, Any
from src.llm_client import get_llm_client, LLMClient


class GroundedReplyGenerator:
    """
    RAG-based support response generator grounded in retrieved historical (customer -> agent) context pairs.
    """

    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or get_llm_client()

    def generate_reply(
        self,
        customer_message: str,
        classified_intent: str,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Drafts a grounded support response.

        Args:
            customer_message: Incoming customer tweet.
            classified_intent: Intent category assigned by classifier.
            retrieved_contexts: Top-k historical context pairs from hybrid retriever.

        Returns:
            Draft reply text string.
        """
        # Format retrieved context pairs into readable prompt block
        context_str = ""
        for i, ctx in enumerate(retrieved_contexts, 1):
            context_str += (
                f"Historical Example {i} (Similarity: {ctx.get('similarity_score', 0.0)}):\n"
                f"  Customer asked: \"{ctx['customer_message']}\"\n"
                f"  Brand resolved: \"{ctx['brand_reply']}\"\n\n"
            )

        if not context_str:
            context_str = "No historical context found."

        system_prompt = (
            "You are an official AmazonHelp customer support representative on Twitter.\n"
            "Your goal is to draft a helpful, empathetic, and concise reply to the customer.\n"
            "CRITICAL REQUIREMENT: Your reply MUST be grounded in how AmazonHelp has historically resolved similar issues as shown in the provided examples.\n"
            "Follow these brand rules:\n"
            "1. Be polite, concise, and professional.\n"
            "2. If order/account details are needed, politely ask the customer to send a Direct Message (DM) with their order ID or account details.\n"
            "3. Do NOT invent fake order numbers or pretend you already know their account details.\n"
            "4. Match the tone and resolution pattern of the historical examples."
        )

        user_prompt = (
            f"Classified Intent: {classified_intent}\n"
            f"Customer Message: \"{customer_message}\"\n\n"
            f"HISTORICAL GROUNDING CONTEXT:\n{context_str}\n"
            "Draft the official AmazonHelp Twitter reply now:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call LLM with temperature ~0.3 for grounded, non-hallucinatory generation
        reply = self.llm_client.call_llm(messages, temperature=0.3, max_tokens=150)
        return reply
