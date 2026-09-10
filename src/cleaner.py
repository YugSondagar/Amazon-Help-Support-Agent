"""
src/cleaner.py
--------------
Purpose:
Reconstruct multi-turn customer support threads from raw Twitter dataset CSVs
and clean noise specific to Twitter (e.g., @handles, URLs, boilerplate text, short thank-yous).

Why this file exists:
Raw Twitter data is exceptionally noisy. To train retrieval models and ground an AI support agent,
we must reconstruct clear (customer message -> brand reply) pairs and eliminate non-informative
or duplicate tweets, maintaining an explicit audit log of how much data was cleaned/dropped.
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Any


def strip_twitter_noise(text: str) -> str:
    """
    Cleans raw tweet text by removing @handles, URLs, and extra spaces.

    Args:
        text: Raw tweet string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs (e.g. https://t.co/xyz123)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove user handles (e.g. @AmazonHelp, @115982)
    text = re.sub(r'@\w+', '', text)

    # Replace multiple whitespaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_meaningful_customer_message(text: str) -> Tuple[bool, str]:
    """
    Determines if a customer tweet contains substantive info or if it should be dropped.

    We filter out tweets under 3 words because they are almost always 'thanks!' or 'ok'
    with no actionable customer support intent.

    Args:
        text: Cleaned text string.

    Returns:
        Tuple of (is_valid: bool, drop_reason: str).
    """
    if not text or len(text.strip()) == 0:
        return False, "empty_text"

    words = text.split()

    # Drop messages under 3 words
    if len(words) < 3:
        return False, "too_short"

    # Drop messages that consist purely of generic thank you phrases
    thanks_pattern = r'^(thanks?|thank\s+you|thx|tysm|great|awesome|ok|okay|got\+it)[!.\s]*$'
    if re.match(thanks_pattern, text.lower()):
        return False, "thanks_only"

    return True, "valid"


def clean_brand_reply(reply_text: str) -> str:
    """
    Strips brand boilerplate signatures like 'DM us your order number' or '^AB' agent tags.

    Args:
        reply_text: Cleaned brand reply text.

    Returns:
        Normalized brand reply text.
    """
    cleaned = strip_twitter_noise(reply_text)
    
    # Strip common twitter agent initials like ^AB, ^JM at end of replies
    cleaned = re.sub(r'\^[A-Z]{2,3}$', '', cleaned).strip()
    
    return cleaned


def process_raw_tweets(df: pd.DataFrame, brand_handle: str = "AmazonHelp") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingests raw Twitter dataframe, reconstructs (customer -> brand) threads,
    applies cleaning filters, and calculates data drop statistics.

    Expected columns in df:
    - tweet_id
    - author_id
    - text
    - created_at
    - in_response_to_tweet_id
    - response_tweet_id

    Returns:
        Tuple of (cleaned_threads_df, drop_stats_dict)
    """
    initial_total_tweets = len(df)
    stats = {
        "initial_total_tweets": initial_total_tweets,
        "brand_handle": brand_handle,
        "dropped_non_brand_replies": 0,
        "dropped_too_short": 0,
        "dropped_thanks_only": 0,
        "dropped_duplicates": 0,
        "reconstructed_pairs": 0,
        "retained_percentage": 0.0
    }

    # Step 1: Separate Brand Replies from Customer Tweets
    brand_df = df[df['author_id'].astype(str).str.lower() == brand_handle.lower()].copy()
    customer_df = df[df['author_id'].astype(str).str.lower() != brand_handle.lower()].copy()

    # Index customer tweets by tweet_id for fast lookup
    customer_tweets_by_id = customer_df.set_index('tweet_id').to_dict('index')

    pairs = []

    # Step 2: Reconstruct pairs by finding brand replies that answer customer tweets
    for _, brand_row in brand_df.iterrows():
        # Check in_response_to_tweet_id link
        in_response_to = brand_row.get('in_response_to_tweet_id')
        
        if pd.isna(in_response_to):
            continue

        try:
            in_response_to_id = int(in_response_to)
        except (ValueError, TypeError):
            continue

        if in_response_to_id in customer_tweets_by_id:
            cust_tweet = customer_tweets_by_id[in_response_to_id]
            
            cust_text_raw = str(cust_tweet.get('text', ''))
            brand_reply_raw = str(brand_row.get('text', ''))

            cust_text_clean = strip_twitter_noise(cust_text_raw)
            brand_reply_clean = clean_brand_reply(brand_reply_raw)

            # Check customer message validity
            is_valid, reason = is_meaningful_customer_message(cust_text_clean)
            if not is_valid:
                if reason == "too_short":
                    stats["dropped_too_short"] += 1
                elif reason == "thanks_only":
                    stats["dropped_thanks_only"] += 1
                continue

            pairs.append({
                "customer_tweet_id": in_response_to_id,
                "brand_tweet_id": brand_row.get('tweet_id'),
                "customer_author_id": cust_tweet.get('author_id'),
                "created_at": cust_tweet.get('created_at'),
                "customer_message": cust_text_clean,
                "brand_reply": brand_reply_clean
            })

    pairs_df = pd.DataFrame(pairs)

    if pairs_df.empty:
        stats["reconstructed_pairs"] = 0
        stats["retained_percentage"] = 0.0
        return pairs_df, stats

    # Step 3: Deduplicate exact duplicate customer messages
    before_dedup = len(pairs_df)
    pairs_df = pairs_df.drop_duplicates(subset=['customer_message']).reset_index(drop=True)
    stats["dropped_duplicates"] = before_dedup - len(pairs_df)

    stats["reconstructed_pairs"] = len(pairs_df)
    stats["retained_percentage"] = round((len(pairs_df) / max(initial_total_tweets, 1)) * 100, 2)

    return pairs_df, stats
