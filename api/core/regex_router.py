"""Layer 1: Hard regex/keyword matching for direct actions."""

import re
from typing import Optional

from api.models.seed import RegexRouteResult

REGEX_RULES: list[tuple[str, str, str]] = [
    (r"人工", "handoff", "转人工"),
    (r"投诉|举报", "complaint", "投诉升级"),
    (r"退订|TD|取消订阅", "unsubscribe", "退订"),
]

COMPILED_RULES = [(re.compile(pattern), action, label) for pattern, action, label in REGEX_RULES]


def regex_route(question: str) -> Optional[RegexRouteResult]:
    for regex, action, label in COMPILED_RULES:
        if regex.search(question.strip()):
            return RegexRouteResult(action=action, label=label, matched=regex.pattern)
    return None
