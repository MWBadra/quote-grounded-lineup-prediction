"""Turns a pre-match corpus into two player-level weights plus one team-level
rotation weight.

Three things are done differently here, each because of something v1 got wrong.

Categories instead of a float. v1 asked for a score in [-1, +1] and got 0.0 back
for 88% of players. Picking one label off a short closed list is a much easier
call to make than placing a continuous value, and when there's nothing to say it
degrades to no_mention rather than to a fake midpoint.

Quote-grounding, checked in code and not just asked for in the prompt. Every
non-null signal has to carry a verbatim span out of the retrieved text; verify()
string-matches it and anything that doesn't match drops to no_mention. That's
the hallucination control and the leakage control in one, since a model reciting
a lineup it memorised in training can't produce a quote that appears in articles
we actually fetched. Structural, not a promise.

Three states, never two. signal / nothing-in-the-text / call-failed all stay
distinct. v1 folded failures into 0.0 where they were indistinguishable from no
news, which is probably most of why its features came out 88-92% zero.

The team-level weight is separate on purpose. A general remark ("I'm thinking
about Wednesday") lifts a squad-wide rotation weight rather than any one
player's, and it goes out as its own column instead of being multiplied into the
player scores, so the model works out how the two interact rather than us
deciding for it.

extract() does no network I/O unless allow_network=True.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llm_keys import ExtractionFailure, gemini_pool, with_retry
from retrieval import Corpus

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".extractcache"

MODEL_NAME = "gemini-3.1-flash-lite"

# --- the label sets ---
# Ordered so the numeric encoding is monotone in "less likely to start".
ROTATION_LABELS = {
    "confirmed_starter": 0.0,   # manager states the player will start
    "expected_starter":  0.2,   # reported as expected to keep his place
    "no_mention":        None,  # nothing said -- distinct from "said to be fine"
    "competition_lost":  0.7,   # reported to have fallen out of the XI
    "rotation_risk":     0.8,   # rest / rotation explicitly floated
    "ruled_out":         1.0,   # unavailable
}
SENTIMENT_LABELS = {
    "praised":     1.0,
    "backed":      0.5,
    "no_mention":  None,
    "questioned": -0.5,
    "criticised": -1.0,
}
TEAM_ROTATION_LABELS = {"none": 0.0, "low": 0.25, "moderate": 0.6, "high": 1.0}
STRENGTHS = {"explicit", "implied", "none"}

# Where a signal came from matters as much as what it says. A manager naming his
# XI and a columnist guessing at it are not the same evidence, and fantasy
# football advice is not evidence at all.
SOURCE_TYPES = {"manager_quote", "club_official", "journalist_prediction",
                "fantasy_tips", "unknown"}
SOURCE_POLICY = {
    "manager_quote":         "keep",
    "club_official":         "keep",
    "journalist_prediction": "downgrade",   # kept, but never "explicit"
    "fantasy_tips":          "reject",      # not about this manager's XI at all
    "unknown":               "downgrade",
}
# One sentence naming four attackers is a weaker claim about each of them than a
# sentence about one player -- but it is still a claim. Rejecting the group threw
# away real signal in the pilot ("the attacking talent of Gordon, Barnes, Elanga
# and Woltemade" genuinely is information), so these are demoted, not discarded.
MAX_PLAYERS_PER_QUOTE = 2

SYSTEM_PROMPT = """You extract team-selection signals from pre-match football reporting.

You are working on ONE club, named below as TARGET CLUB. Sources often cover both
sides of a fixture. Anything about the OPPONENT is irrelevant, however
interesting -- report only players on the TARGET CLUB roster.

You will be given:
  TARGET CLUB - the club whose team selection is being predicted
  ROSTER      - the exact player names available to that club
  SOURCES     - numbered articles published BEFORE this fixture

Return ONLY raw JSON. No markdown fences, no commentary.

{
  "team_rotation": {
    "level": "none|low|moderate|high",
    "quote": "<verbatim span from SOURCES, or empty>",
    "source_index": <int or null>,
    "reason": "<short phrase>"
  },
  "players": [
    {
      "name": "<exact name from ROSTER>",
      "rotation": "confirmed_starter|expected_starter|no_mention|competition_lost|rotation_risk|ruled_out",
      "sentiment": "praised|backed|no_mention|questioned|criticised",
      "strength": "explicit|implied|none",
      "source_type": "manager_quote|club_official|journalist_prediction|fantasy_tips",
      "quote": "<verbatim span from SOURCES, or empty>",
      "source_index": <int or null>
    }
  ]
}

WHAT COUNTS AS A SIGNAL
Ask one question of every sentence: does this tell me whether this player starts
this weekend? If not, it is not a signal.

  SIGNAL      "Slot confirmed Salah will start"          -> confirmed_starter
              "Havertz is not ready to go from the off"  -> rotation_risk
              "Rogers has lost his place to Tielemans"   -> competition_lost
              "Bradley misses the rest of the season"    -> ruled_out
              "Trippier comes in at right-back"          -> expected_starter

  NOT A SIGNAL   goals scored on international duty; contract or transfer talk;
                 last season's form; praise for a performance already played;
                 fantasy football advice; a player's price or ownership.

RULES
- Only list players the SOURCES actually discuss. Everyone else is no_mention by
  default. Do not pad the list to cover the roster.
- "quote" MUST be copied character-for-character from SOURCES. Never paraphrase,
  never merge two fragments, never insert an ellipsis. A quote that does not
  appear verbatim will be rejected and the signal discarded.
- The quote must SUPPORT the label you chose. If the only quote you can find is
  praise, do not label that player rotation_risk.
- If you cannot supply a supporting verbatim quote, use no_mention.
- Do not attach one quote to several players. A sentence listing four names is a
  list, not a judgement about each of them.
- "source_type" is WHO is speaking:
    manager_quote         - the manager\'s own words about his squad
    club_official         - the club\'s own channel reporting its team news
    journalist_prediction - a reporter\'s guess, not a quote
    fantasy_tips          - fantasy football advice. Never a selection signal.
  Label this honestly; a columnist predicting an XI is not the manager naming it.
- "team_rotation" is for the SQUAD as a whole -- fixture congestion, prioritising
  another match, "changes are coming". A statement about one player goes in
  "players". Use "none" unless the sources genuinely say the squad will rotate.
- Judge only what the SOURCES say. Do not use anything you remember about this
  fixture, this season, or these players."""


@dataclass
class PlayerSignal:
    name: str
    rotation: str = "no_mention"
    sentiment: str = "no_mention"
    strength: str = "none"
    quote: str = ""
    source_index: int | None = None
    source_type: str = "unknown"
    verified: bool = False
    reject_reason: str = ""


@dataclass
class TeamSignal:
    level: str = "none"
    quote: str = ""
    source_index: int | None = None
    reason: str = ""
    verified: bool = False


@dataclass
class Extraction:
    team_code: int
    gameweek: int
    match_id: str
    status: str = "ok"                 # ok | empty_corpus | failed
    error: str = ""
    team: TeamSignal = field(default_factory=TeamSignal)
    players: list[PlayerSignal] = field(default_factory=list)
    n_verified: int = 0
    n_rejected: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Extraction":
        t = TeamSignal(**d.pop("team", {}))
        ps = [PlayerSignal(**p) for p in d.pop("players", [])]
        return Extraction(team=t, players=ps, **d)


# --- quote verification ---

def _norm(s: str) -> str:
    """
    Collapse whitespace, unify quote glyphs, and strip accents.

    Accent folding matters: the pilot rejected a genuine quote about "Hincapie"
    because the model wrote the name without its acute accent.
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_is_grounded(quote: str, corpus_text: str, min_len: int = 25,
                      roster: list[str] | None = None) -> bool:
    """
    True only if the quote genuinely appears in the retrieved text.

    Short spans are normally rejected -- a 10-character fragment matches by
    chance and proves nothing. The exception is a short span that names a squad
    member: the pilot discarded "Doubtful Havertz (knee)", a real injury-table
    entry, purely for being 23 characters long. Naming a player is itself
    evidence the span is specific, so those are allowed down to 12 characters.
    """
    q = _norm(quote)
    if not q:
        return False
    floor = min_len
    if roster:
        names = {n for full in roster
                 for n in re.split(r"[\s.'-]+", _norm(str(full))) if len(n) >= 4}
        if any(n in q for n in names):
            floor = 12
    if len(q) < floor:
        return False
    return q in _norm(corpus_text)


def verify(ex: Extraction, corpus: Corpus,
           roster: list[str] | None = None) -> Extraction:
    """
    Enforce grounding, source policy and the one-quote-many-players cap.

    Grounding alone proved insufficient in the pilot: it confirms the text
    exists but not that the text is manager selection talk. A columnist's guess
    and an FPL tips column both pass a string match.
    """
    text = corpus.searchable_text()

    # A quote reused across many players is a list of names, not a judgement
    # about each of them -- demote the whole group rather than trust it.
    reuse = Counter(_norm(p.quote) for p in ex.players if p.quote)

    if ex.team.level != "none":
        if quote_is_grounded(ex.team.quote, text):
            ex.team.verified = True
        else:
            ex.team = TeamSignal(level="none", reason="quote not grounded")

    kept = []
    for p in ex.players:
        says_something = (p.rotation != "no_mention") or (p.sentiment != "no_mention")
        if not says_something:
            continue
        policy = SOURCE_POLICY.get(p.source_type, "downgrade")
        if policy == "reject":
            p.verified = False
            p.reject_reason = f"source type '{p.source_type}' is not selection evidence"
            ex.n_rejected += 1
            kept.append(p)
            continue
        if policy == "downgrade" and p.strength == "explicit":
            p.strength = "implied"
        if reuse[_norm(p.quote)] > MAX_PLAYERS_PER_QUOTE:
            p.strength = "implied"
            p.reject_reason = (f"shared quote across {reuse[_norm(p.quote)]} players "
                               "- demoted to implied")
        if quote_is_grounded(p.quote, text, roster=roster):
            p.verified = True
            ex.n_verified += 1
            kept.append(p)
        else:
            p.verified = False
            p.reject_reason = ("quote too short" if len(_norm(p.quote)) < 25
                               else "quote not found in sources")
            ex.n_rejected += 1
            kept.append(p)          # retained for the audit trail, value nulled below
    ex.players = kept
    return ex


# --- numeric encoding ---

def to_rows(ex: Extraction, roster: list[str]) -> list[dict]:
    """
    One row per roster player. Unverified or absent signals become None, NOT 0 --
    a null means "we learned nothing", which a model can treat differently from
    "we learned he is fine".
    """
    by_name = {p.name: p for p in ex.players if p.verified}
    team_val = (TEAM_ROTATION_LABELS.get(ex.team.level, 0.0)
                if ex.team.verified else None)
    rows = []
    for name in roster:
        p = by_name.get(name)
        rows.append({
            "player_name": name,
            "team_code": ex.team_code,
            "gameweek": ex.gameweek,
            "match_id": ex.match_id,
            "llm_team_rotation": team_val,
            "llm_player_rotation": ROTATION_LABELS.get(p.rotation) if p else None,
            "llm_manager_sentiment": SENTIMENT_LABELS.get(p.sentiment) if p else None,
            "llm_strength": p.strength if p else "none",
            "llm_source_type": p.source_type if p else "",
            "llm_quote": p.quote if p else "",
            "llm_status": ex.status,
        })
    return rows


# --- the call ---

def build_user_prompt(roster: list[str], corpus: Corpus) -> str:
    return (f"TARGET CLUB: {corpus.team}\n"
            f"ROSTER ({len(roster)} players): {', '.join(roster)}\n\n"
            f"FIXTURE: {corpus.team}, gameweek {corpus.gameweek}, "
            f"kickoff {corpus.match_date}\n"
            f"SOURCE WINDOW: {corpus.window_start} to {corpus.window_end}\n\n"
            f"SOURCES:\n{corpus.as_prompt_text()}")


def _call_gemini(system: str, user: str) -> dict:
    """
    One Gemini call. RAISES on any failure so with_retry() actually retries --
    the opposite of v1's scorer, which swallowed its own exceptions and thereby
    disabled the retry wrapper entirely.

    Uses langchain_google_genai because that is what is installed and proven
    against these keys; the raw google-generativeai SDK is not present here.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = gemini_pool().acquire()
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=key.value,
                                 temperature=0.0)
    resp = llm.invoke([("system", system), ("human", user)])
    # Newer LangChain returns content as a list of typed blocks rather than a
    # bare string, so flatten before parsing.
    raw = getattr(resp, "content", "")
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    txt = (raw or "").strip()
    if not txt:
        raise RuntimeError("empty response")
    txt = re.sub(r"^\s*```(?:json)?|```\s*$", "", txt, flags=re.MULTILINE).strip()
    data = json.loads(txt)          # raises on malformed JSON -> triggers retry
    if not isinstance(data, dict) or "players" not in data:
        raise RuntimeError(f"unexpected schema: {list(data)[:5]}")
    return data


def parse(data: dict, corpus: Corpus) -> Extraction:
    ex = Extraction(team_code=corpus.team_code, gameweek=corpus.gameweek,
                    match_id=corpus.match_id)
    t = data.get("team_rotation") or {}
    lvl = str(t.get("level", "none")).lower()
    ex.team = TeamSignal(level=lvl if lvl in TEAM_ROTATION_LABELS else "none",
                         quote=str(t.get("quote", "") or ""),
                         source_index=t.get("source_index"),
                         reason=str(t.get("reason", "") or ""))
    for p in data.get("players") or []:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        rot = str(p.get("rotation", "no_mention")).lower()
        sen = str(p.get("sentiment", "no_mention")).lower()
        st = str(p.get("strength", "none")).lower()
        ex.players.append(PlayerSignal(
            name=str(p["name"]),
            rotation=rot if rot in ROTATION_LABELS else "no_mention",
            sentiment=sen if sen in SENTIMENT_LABELS else "no_mention",
            strength=st if st in STRENGTHS else "none",
            quote=str(p.get("quote", "") or ""),
            source_index=p.get("source_index"),
            source_type=(str(p.get("source_type", "unknown")).lower()
                         if str(p.get("source_type", "")).lower() in SOURCE_TYPES
                         else "unknown")))
    return ex


def cache_path(team_code: int, gameweek: int, match_id: str | None = None) -> Path:
    """Keyed on the FIXTURE, not just (team, gameweek) -- see retrieval.cache_path."""
    if match_id is None:
        return CACHE_DIR / f"gw{gameweek:02d}_team{team_code}.json"
    tag = hashlib.sha1(str(match_id).encode("utf-8")).hexdigest()[:8]
    return CACHE_DIR / f"gw{gameweek:02d}_team{team_code}_{tag}.json"


def load_cached_extraction(team_code: int, gameweek: int,
                           match_id: str | None = None) -> "Extraction | None":
    """Legacy-key fallback, but never returns an extraction from another fixture."""
    p = cache_path(team_code, gameweek, match_id)
    if p.exists():
        return Extraction.from_dict(json.loads(p.read_text(encoding="utf-8")))
    legacy = cache_path(team_code, gameweek)
    if not legacy.exists():
        return None
    ex = Extraction.from_dict(json.loads(legacy.read_text(encoding="utf-8")))
    if match_id is not None and str(ex.match_id) != str(match_id):
        return None
    return ex


def is_cached(team_code: int, gameweek: int, match_id: str | None = None) -> bool:
    """True only if an extraction for THIS fixture is already on disk."""
    return load_cached_extraction(team_code, gameweek, match_id) is not None


def extract(roster: list[str], corpus: Corpus, *,
            allow_network: bool = False, refresh: bool = False) -> Extraction:
    """
    Extract signals for one team-gameweek. No network I/O unless allow_network.
    Re-extraction is free of Tavily cost because the corpus is already on disk --
    which is what makes prompt iteration cheap.
    """
    p = cache_path(corpus.team_code, corpus.gameweek, corpus.match_id)
    if not refresh:
        hit = load_cached_extraction(corpus.team_code, corpus.gameweek, corpus.match_id)
        if hit is not None:
            return hit

    if corpus.status != "ok" or not corpus.articles:
        return Extraction(team_code=corpus.team_code, gameweek=corpus.gameweek,
                          match_id=corpus.match_id, status="empty_corpus")

    if not allow_network:
        raise RuntimeError(
            f"No cached extraction for {corpus.team} GW{corpus.gameweek} and "
            "allow_network=False. Pass allow_network=True to spend Gemini quota.")

    try:
        data = with_retry(_call_gemini, SYSTEM_PROMPT,
                          build_user_prompt(roster, corpus))
        ex = verify(parse(data, corpus), corpus, roster=roster)
    except ExtractionFailure as e:
        ex = Extraction(team_code=corpus.team_code, gameweek=corpus.gameweek,
                        match_id=corpus.match_id, status="failed", error=str(e))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ex.to_dict(), indent=1), encoding="utf-8")
    return ex
