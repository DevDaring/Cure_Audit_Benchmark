"""
File: Dry_Run/dry_run_bedrock.py
Purpose: Sanity-check that AWS_ACCESS_KEY + AWS_SECRET_KEY can reach the
         Bedrock-routed API models (built dynamically from config.API_MODELS):
           - qwen.qwen3-next-80b-a3b
           - amazon.nova-2-lite-v1:0

Uses the Bedrock Converse API (model-agnostic).
Keys are loaded from .env via config.py — never printed.

Run standalone : python Dry_Run/dry_run_bedrock.py
Run via dry_run_all.py: included automatically
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # loads .env

logger = logging.getLogger(__name__)

REGION = "us-east-1"
TEST_PROMPT = "Reply with exactly the words: AWS OK"
MAX_TOKENS = 20

# Only the models that route through Bedrock
BEDROCK_MODELS = [
    cfg for cfg in config.API_MODELS if cfg["primary_route"] == "bedrock"
]


def _test_model(boto_client, model_id: str, name: str) -> bool:
    """Call one model via Converse API. Returns True on success."""
    logger.info("Testing Bedrock model: %s (%s)", name, model_id)
    t0 = time.monotonic()
    try:
        response = boto_client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": TEST_PROMPT}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS},
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        output_text = ""
        try:
            output_text = response["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError):
            output_text = str(response.get("output", ""))

        stop_reason = response.get("stopReason", "—")
        usage = response.get("usage", {})
        logger.info(
            "  PASS  %s | %d ms | stop=%s | in=%s out=%s | response=%r",
            name,
            latency_ms,
            stop_reason,
            usage.get("inputTokens", "?"),
            usage.get("outputTokens", "?"),
            output_text.strip(),
        )
        return True

    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.error("  FAIL  %s | %d ms | %s", name, latency_ms, exc)
        return False


def run() -> bool:
    """
    Run Bedrock connectivity check for both API models.
    Returns True if all models respond successfully.
    """
    try:
        import boto3  # type: ignore
    except ImportError:
        logger.error("boto3 not installed. Run: pip install boto3")
        return False

    logger.info("=== Bedrock dry-run: region=%s, models=%d ===", REGION, len(BEDROCK_MODELS))
    logger.info(
        "Key lengths: access=%d secret=%d",
        len(config.AWS_ACCESS_KEY),
        len(config.AWS_SECRET_KEY),
    )

    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY,
        aws_secret_access_key=config.AWS_SECRET_KEY,
    )

    results = []
    for cfg in BEDROCK_MODELS:
        ok = _test_model(client, cfg["model_id"], cfg["name"])
        results.append((cfg["name"], ok))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    logger.info("Bedrock dry-run: %d/%d models passed", passed, total)
    for name, ok in results:
        logger.info("  %s  %s", "PASS" if ok else "FAIL", name)

    return passed == total


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    success = run()
    sys.exit(0 if success else 1)
