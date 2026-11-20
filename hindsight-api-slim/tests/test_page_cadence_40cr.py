"""40cr acceptance: project pages refresh every six hours, not weekly.

The producer is the integration's default cron constant and its hash expansion;
the consumer is the drift check that reconciles an existing weekly page. The TS
runner is absent here (``node_modules`` cannot be installed), so each rule is read
out of the shipped source and pinned at the statement that carries it.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CORE = _REPO / "hindsight-integrations" / "coding-agents" / "src" / "core"
_MISSIONS = _CORE / "missions.ts"
_CONFIG = _CORE / "config.ts"

#: The trigger fields this plugin states, and so the only ones a re-sync may compare (C3, #3506).
_PLUGIN_OWNED_TRIGGER_FIELDS = ("tags_match", "refresh_cron", "refresh_after_consolidation")
#: Trigger fields the server or the operator owns; a re-sync must have no opinion about them.
_SERVER_OWNED_TRIGGER_FIELDS = ("mode", "staleness_check", "min_refresh_interval_seconds")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    """The body of the top-level ``export function <name>(...)``, braces balanced."""
    start = source.index(f"export function {name}(")
    opening = source.index("{", start)
    depth = 0
    for i in range(opening, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : i]
    raise AssertionError(f"unterminated body for {name}")


def _literal_end(source: str, start: int) -> int | None:
    """Index just past ``start``'s quoted run, or ``None`` when it does not close on that line.

    A quote that does not close on its own line is not a literal here: once a block comment has
    been ended early by prose, the rest of that prose is lexed as code, and the quotes in it must
    not swallow the comment's real terminator. TypeScript's ``"`` and ``'`` strings never span
    lines; the only multi-line ones are template literals, and none in these files carry a
    comment marker.
    """
    quote = source[start]
    newline = source.find("\n", start)
    limit = len(source) if newline == -1 else newline
    i = start + 1
    while i < limit:
        if source[i] == "\\":
            i += 2
            continue
        if source[i] == quote:
            return i + 1
        i += 1
    return None


def _comment_faults(source: str) -> list[str]:
    """Lines where a block-comment marker contradicts the comment state around it.

    A TypeScript block comment ends at the first ``*`` followed by ``/``, so this
    package's shipped cron example written inside ``/** ... */`` closes the comment
    early and the rest of the prose is lexed as code. Quoted runs and line comments
    are skipped so only real comment markers count.
    """
    faults: list[str] = []
    block_open: int | None = None
    length = len(source)
    i = 0

    def line_at(offset: int) -> int:
        return source.count("\n", 0, offset) + 1

    while i < length:
        if block_open is not None:
            end = source.find("*/", i)
            if end == -1:
                break
            block_open = None
            i = end + 2
            continue
        ch = source[i]
        if ch in "\"'`":
            end = _literal_end(source, i)
            if end is not None:
                i = end
                continue
        if ch == "/" and i + 1 < length:
            nxt = source[i + 1]
            if nxt == "/":
                newline = source.find("\n", i)
                i = length if newline == -1 else newline + 1
                continue
            if nxt == "*":
                block_open = i
                i += 2
                continue
        if ch == "*" and i + 1 < length and source[i + 1] == "/":
            faults.append(f"line {line_at(i)}: `*/` closes no open block comment")
            i += 2
            continue
        i += 1

    if block_open is not None:
        faults.append(f"line {line_at(block_open)}: block comment never closes")
    return faults


def test_default_page_trigger_cron_is_six_hourly() -> None:
    """The shipped default is a six-hour expression with a hashed minute."""
    source = _read(_MISSIONS)
    match = re.search(r'DEFAULT_PAGE_TRIGGER_CRON\s*=\s*"([^"]+)"', source)
    assert match, "default cron constant not found"
    assert match.group(1) == "H */6 * * *"


def test_hashed_minute_expansion_keeps_six_hour_field() -> None:
    """``expandCronHash`` rewrites only hashed fields, so the ``*/6`` hour survives.

    The shipped default is the crossing this depends on: ``H */6 * * *`` expands to
    an ordinary five-field cron whose hour field is still ``*/6``. That holds only
    while parsing skips every field that does not start with ``H`` -- any field
    beginning with ``H`` claims the hashed syntax, every other field stays the
    server's to validate -- and while expansion returns a field unchanged when the
    hashed pattern does not match. Without that gate every field is hashed and the
    six-hour step is clobbered into one hour.
    """
    source = _read(_MISSIONS)
    parse_body = re.sub(r"\s+", " ", _function_body(source, "parseHashedCron"))
    expand_body = re.sub(r"\s+", " ", _function_body(source, "expandCronHash"))

    assert re.search(r'if \(!field\.startsWith\("H"\)\) continue;', parse_body), (
        "parseHashedCron must skip a field that is not hashed"
    )
    gate = re.search(r"if \(!m\) return field;", expand_body)
    assert gate is not None, "expandCronHash must return a non-hashed field unchanged"
    substitution = "String(hashedValue(seed, i, lo, hi))"
    assert substitution in expand_body, "expandCronHash must substitute the seed's value for a hashed field"
    assert gate.start() < expand_body.index(substitution), (
        "the non-hashed pass-through must run before the hash substitution"
    )
    hashed_field = re.search(r"const HASHED_FIELD = /([^/]+)/;", source)
    assert hashed_field is not None and hashed_field.group(1).startswith("^H"), (
        "HASHED_FIELD must match only a field beginning with H"
    )


def test_weekly_existing_page_is_drifted_against_new_default() -> None:
    """The drift check compares exactly the plugin-owned trigger fields (C3, #3506).

    A page still on the weekly cadence drifts because the check compares
    ``refresh_cron`` against the page's own resolved default (six-hourly). It must
    compare only what this plugin states: a check that compares nothing never
    reconciles the weekly page (R5/A6), and one widened to a server-owned field
    re-syncs pages over settings the operator owns.
    """
    body = re.sub(r"\s+", " ", _function_body(_read(_MISSIONS), "pageTriggerDrifted"))

    compared = set(re.findall(r"(?:current|desired)\.(\w+)", body))
    assert compared == set(_PLUGIN_OWNED_TRIGGER_FIELDS), (
        f"pageTriggerDrifted compares {sorted(compared)}, "
        f"expected exactly {sorted(_PLUGIN_OWNED_TRIGGER_FIELDS)}"
    )
    for field in _PLUGIN_OWNED_TRIGGER_FIELDS:
        assert re.search(rf"current\.{field}.*?!==.*?desired\.{field}", body), (
            f"pageTriggerDrifted must compare current.{field} against desired.{field}"
        )
    for field in _SERVER_OWNED_TRIGGER_FIELDS:
        assert not re.search(rf"\.{field}\b", body), f"pageTriggerDrifted must not compare the server-owned {field}"


def test_ts_sources_keep_block_comments_balanced() -> None:
    """Both changed sources must lex: a ``*/`` may only close an open block comment.

    Text-level checks over this tree read the file as characters and cannot see a
    lexer failure, which is how a cron example inside a doc block shipped with the
    rest of that block parsed as code.
    """
    faults = {path.name: _comment_faults(_read(path)) for path in (_MISSIONS, _CONFIG)}
    assert faults == {"missions.ts": [], "config.ts": []}, f"unbalanced block comments: {faults}"
