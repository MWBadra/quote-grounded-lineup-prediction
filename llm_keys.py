"""Key discovery, quota accounting and rate limiting for the v2 extraction stage.

Two things changed from v1 on purpose.

v1 pinned each Gemini key to a role (refiner, injury, sentiment, rotation). v2
makes one extraction call per team-match, so the keys are interchangeable now
and the whole pool feeds a single queue.

v1's rate limiter itself was fine, but the callers swallowed exceptions, so the
retry path never actually ran - see weight_calculator._run_scorer. Here the
network helpers raise and the retry decorator is the only place a failure gets
absorbed. It records the failure too, instead of quietly returning zero, which
is how v1 ended up with columns full of silent zeros.

No network I/O anywhere in this module.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / ".quota"
load_dotenv(HERE.parent / ".env")

# Gemini free tier, per key.
GEMINI_RPM = 15
GEMINI_RPD = 500
# Tavily: a 429 or timeout costs no credit, only a successful search does, so the
# credit budget is driven by unique searches rather than by attempts.
TAVILY_CREDITS_ADVANCED = 2
TAVILY_CREDITS_BASIC = 1
TAVILY_RPM = 100
# Tavily bills per successful search, 2 credits at advanced depth, against a
# ~1000-credit monthly allowance per free key. Expressed as searches so the pool
# retires a key at its real ceiling instead of at a placeholder.
TAVILY_SEARCHES_PER_KEY = 500


class QuotaExhausted(RuntimeError):
    """Every key in the pool has spent its daily allowance."""


class CreditsExhausted(RuntimeError):
    """
    The provider refused the call for billing reasons rather than transiently.

    Distinct from QuotaExhausted (our own daily counter) and from a transient
    error: retrying a credit refusal only burns wall-clock, so this propagates
    straight through with_retry and halts the run so keys can be topped up and
    the job resumed from cache.
    """


CREDIT_MARKERS = ("credit", "quota exceeded", "usage limit", "plan limit",
                  "insufficient", "payment required", "402", "432",
                  "exceeded your", "out of credits")


def looks_like_credit_error(e: Exception) -> bool:
    m = str(e).lower()
    return any(k in m for k in CREDIT_MARKERS)


class ExtractionFailure(RuntimeError):
    """A call failed after all retries. Callers must record NULL, never 0."""


def discover_keys(prefix: str) -> list[tuple[str, str]]:
    """
    All env vars whose name starts with `prefix`, as (name, value).

    Matches both the plain form (TAVILY_API_KEY) and any numbered or role-suffixed
    variants (GEMINI_ROTATION_KEY_1, TAVILY_API_KEY_2), so the existing .env works
    unchanged and new keys are picked up by adding them.
    """
    out = []
    seen = set()
    for name, val in os.environ.items():
        if not name.startswith(prefix) or not val or len(val) < 20:
            continue
        if "YourActual" in val or "your_" in val.lower():
            continue
        if val in seen:          # same key listed under two names
            continue
        seen.add(val)
        out.append((name, val))
    return sorted(out)


@dataclass
class KeyState:
    name: str
    value: str
    rpm: int
    rpd: int
    calls: list[float] = field(default_factory=list)   # rolling 60s window
    used_today: int = 0
    day: str = field(default_factory=lambda: date.today().isoformat())

    @property
    def _path(self) -> Path:
        safe = re.sub(r"\W+", "_", self.name)
        return STATE_DIR / f"{safe}.json"

    def load(self) -> None:
        if self._path.exists():
            d = json.loads(self._path.read_text())
            if d.get("day") == date.today().isoformat():
                self.used_today, self.day = d.get("used", 0), d["day"]

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"day": self.day, "used": self.used_today}))

    def roll_day(self) -> None:
        today = date.today().isoformat()
        if today != self.day:
            self.day, self.used_today = today, 0
            self.save()

    def remaining(self) -> int:
        self.roll_day()
        return max(0, self.rpd - self.used_today)

    def wait_needed(self) -> float:
        """Seconds to wait before this key may be used again under its RPM cap."""
        now = time.monotonic()
        self.calls = [t for t in self.calls if t > now - 60]
        if len(self.calls) < self.rpm:
            return 0.0
        return max(0.0, 60 - (now - self.calls[0]) + 0.05)

    def charge(self) -> None:
        self.calls.append(time.monotonic())
        self.used_today += 1
        self.save()


class KeyPool:
    """
    Round-robins a pool of keys, always picking the one that is usable soonest.

    Sequential rotation across N keys gives N x the single-key RPM for free: by
    the time the rotation returns to a key, enough wall-clock has usually passed
    that its window has drained.
    """

    def __init__(self, prefix: str, rpm: int, rpd: int, label: str = ""):
        found = discover_keys(prefix)
        if not found:
            raise RuntimeError(
                f"No API keys found for prefix '{prefix}'. Add them to .env.")
        self.label = label or prefix
        self.keys = [KeyState(n, v, rpm, rpd) for n, v in found]
        for k in self.keys:
            k.load()

    def __len__(self) -> int:
        return len(self.keys)

    def capacity_today(self) -> int:
        return sum(k.remaining() for k in self.keys)

    def acquire(self) -> KeyState:
        """
        Block until a key is free, then charge and return it.

        Selection is by (wait, usage) rather than by wait alone. When the request
        rate is well under the RPM cap -- which it is for Tavily at ~11s per
        fixture -- every key reports zero wait, so ordering by wait alone always
        returned keys[0] and drove a single key to its credit limit while the
        rest sat untouched. Adding used_today as the tiebreak spreads the load
        evenly, which is the entire point of holding several keys.
        """
        live = [k for k in self.keys if k.remaining() > 0]
        if not live:
            raise QuotaExhausted(
                f"{self.label}: all {len(self.keys)} keys exhausted for today")
        best = min(live, key=lambda k: (k.wait_needed(), k.used_today))
        w = best.wait_needed()
        if w > 0:
            time.sleep(w)
        best.charge()
        return best

    def report(self) -> str:
        lines = [f"{self.label}: {len(self.keys)} key(s), "
                 f"{self.capacity_today():,} calls remaining today"]
        for k in self.keys:
            lines.append(f"    {k.name:<28} {k.remaining():>5} / {k.rpd} left")
        return "\n".join(lines)


def with_retry(fn, *args, attempts: int = 4, base: float = 2.0, **kwargs):
    """
    Retry with exponential backoff and jitter.

    This is the ONLY place a failure is absorbed. `fn` must raise on failure --
    a helper that catches its own exceptions and returns a sentinel makes this
    decorator a no-op, which is exactly the bug that disabled every retry in v1.
    On final failure this raises ExtractionFailure so the caller stores NULL
    rather than a zero that would be indistinguishable from "no signal".
    """
    last = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except (QuotaExhausted, CreditsExhausted):
            raise
        except Exception as e:  # noqa: BLE001 - deliberately broad, then re-raised
            if looks_like_credit_error(e):
                raise CreditsExhausted(str(e)) from e
            last = e
            if i < attempts - 1:
                time.sleep(base * (2 ** i) + random.uniform(0, 1))
    raise ExtractionFailure(f"failed after {attempts} attempts: {last}") from last


def gemini_pool() -> KeyPool:
    return KeyPool("GEMINI", GEMINI_RPM, GEMINI_RPD, label="Gemini")


def tavily_pool() -> KeyPool:
    # 100 requests/minute per key. The daily figure is left high because Tavily
    # bills by credit, not by call count -- the real ceiling is the monthly
    # credit budget, which retrieval.py accounts for separately.
    return KeyPool("TAVILY", rpm=TAVILY_RPM, rpd=TAVILY_SEARCHES_PER_KEY, label="Tavily")


if __name__ == "__main__":
    for maker in (gemini_pool, tavily_pool):
        try:
            print(maker().report())
        except RuntimeError as e:
            print(e)
        print()
