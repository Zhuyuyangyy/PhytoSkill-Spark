"""Deterministic prompt-injection screen for text carried in NLIP messages.

This is a rule-based first line, not a classifier. It normalises the text
(Unicode NFKC, zero-width and bidi control removal, whitespace collapse), also
scans base64 runs that decode to readable text, and matches a fixed rule set in
English and Chinese. It sees only the message, never the agent's reasoning.

Its known blind spot is paraphrase: an instruction that avoids every rule passes.
The evaluation reports that miss rate on a public dataset rather than hiding it;
the delegation layer is what keeps a missed injection from turning into an
unauthorised action.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass

from shield.gate import FORBIDDEN_PATTERNS

_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
_B64_RUN = re.compile(r"[A-Za-z0-9+/=_-]{24,}")

# (rule id, pattern). Matched against normalised, lower-cased text.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override_previous",
     re.compile(r"\b(ignore|disregard|forget|override|skip|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the|these|those)\b.{0,30}"
                r"\b(instructions?|prompts?|rules?|directions?|guidelines?|messages?|context|tasks?|assignments?|orders?)\b")),
    ("new_instructions", re.compile(r"\b(new|updated|real|actual|true)\s+(instructions?|task|orders?|directive)s?\s*(:|are|is)")),
    ("role_hijack", re.compile(r"\b(you are now|from now on,? you (are|will)|act as (an?|the) (unrestricted|unfiltered|jailbroken)|pretend (to be|you are)|"
                               r"you('re| are) no longer)\b")),
    ("jailbreak_persona", re.compile(r"\b(dan mode|developer mode (enabled|on)|do anything now|jailbroken|without (any )?(restrictions|filters|censorship))\b")),
    ("system_prompt_leak", re.compile(r"\b(reveal|print|show|output|repeat|tell me|give me|what (is|are|was|were))\b.{0,30}\b(system prompt|initial (prompt|instructions)|hidden (prompt|instructions)|"
                                      r"your (instructions|prompt|rules))\b")),
    ("fake_delimiter", re.compile(r"(</?\s*(system|assistant|im_start|im_end)\s*>|\[/?inst\]|<\|im_(start|end)\|>|^#+\s*(system|assistant)\s*:|\bsystem\s*:\s*you)", re.M)),
    ("tool_coercion", re.compile(r"\b(call|invoke|use|run|execute)\b.{0,20}\b(tool|function|api|skill)\b.{0,60}\b(with|to send|to upload|to delete|immediately)\b")),
    ("exfiltrate", re.compile(r"\b(send|forward|upload|post|email|exfiltrate|leak|transmit)\b.{0,60}\b(to|at)\b.{0,10}(https?://|[\w.+-]+@[\w-]+\.\w+|attacker)")),
    ("conceal", re.compile(r"\b(do not|don't|never)\s+(tell|inform|mention|reveal|show)\b.{0,20}\b(the )?(user|human|operator|anyone)\b")),
    ("zh_override", re.compile(r"(忽略|无视|忘记|忘掉|不要理会|跳过).{0,8}(之前|以上|上面|前面|所有|先前|原来)?.{0,6}(指令|指示|提示|规则|要求|设定|任务)")),
    ("zh_role_hijack", re.compile(r"(你现在是|从现在开始你|现在起你|扮演一个?没有限制|你不再是)")),
    ("zh_leak", re.compile(r"(输出|告诉我|显示|泄露|打印).{0,10}(系统提示|系统指令|提示词|初始指令)")),
    ("de_override", re.compile(r"\b(ignoriere|vergiss|missachte)\b.{0,40}\b(anweisungen|vorherigen|alles|befehle|regeln)\b")),
)
GATE_RULES = tuple((f"gate_{label}", pattern) for label, pattern in FORBIDDEN_PATTERNS
                   if label in ("hidden_instruction",))


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _decoded_runs(text: str) -> list[str]:
    """Readable text hidden in base64 runs (common smuggling trick)."""
    found: list[str] = []
    for run in _B64_RUN.findall(text):
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                raw = decoder(run + "=" * (-len(run) % 4))
                decoded = raw.decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError):
                continue
            printable = sum(ch.isprintable() for ch in decoded) / max(len(decoded), 1)
            if printable > 0.95 and re.search(r"[a-zA-Z\u4e00-\u9fff]{3}", decoded):
                found.append(decoded)
                break
    return found


@dataclass
class InjectionFinding:
    rule: str
    where: str
    excerpt: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "where": self.where, "excerpt": self.excerpt}


def scan_text(text: str, where: str = "content") -> list[InjectionFinding]:
    findings: list[InjectionFinding] = []
    candidates = [(where, normalise(text))]
    candidates += [(f"{where}/base64", normalise(chunk)) for chunk in _decoded_runs(text)]
    for location, normalised in candidates:
        for rule, pattern in RULES + GATE_RULES:
            match = pattern.search(normalised)
            if match:
                findings.append(InjectionFinding(rule, location, normalised[max(0, match.start()-10):match.end()+10][:120]))
    return findings


def iter_strings(value, path: str = "content"):
    """Every string inside a (possibly nested) NLIP content value, with its path."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield path + "." + str(key), str(key)
            yield from iter_strings(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_strings(item, f"{path}[{index}]")
