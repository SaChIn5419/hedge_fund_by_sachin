from __future__ import annotations

import json

from agents.dexter.registry import DEXTER_VERSION, FEATURE_SPECS


SYSTEM_PROMPT = """You are Dexter, an India-first financial research sensor for Chimera.
You must use only the provided dated source excerpts. Do not use memory, unstated facts,
or assumptions about events outside the source window. If evidence is stale, missing, or
ambiguous, return neutral numeric scores and say so in reason. Return only valid JSON."""


def json_contract() -> str:
    fields = {
        name: {
            "lower": spec.lower,
            "upper": spec.upper,
            "neutral": spec.neutral,
            "description": spec.description,
        }
        for name, spec in FEATURE_SPECS.items()
    }
    return json.dumps(
        {
            "dexter_version": DEXTER_VERSION,
            "rules": [
                "India domestic evidence has priority over global evidence.",
                "Global evidence may affect India only through explicit spillover fields.",
                "Every non-neutral score must be supported by source_evidence.",
                "Do not mention facts that are not in source_evidence.",
                "Use neutral scores when evidence is stale or insufficient.",
            ],
            "flat_feature_fields": fields,
            "required_metadata": ["reason", "source_evidence"],
        },
        indent=2,
        sort_keys=True,
    )
