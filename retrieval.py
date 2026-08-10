"""Pre-match news retrieval for the v2 LLM stage.

Revision 2, after the first pilot returned a corpus that was only 24% on-topic
(10 of 25 articles were Champions League coverage, 6 were junk including horse
racing, and all 25 came from just 3 of the 22 whitelisted domains).

What changed, and why:

Dropped include_domains. Handing Tavily a domain whitelist seems to force
results from those domains instead of ranking on relevance, so we just got
whatever Sky Sports put out that week. Domains are filtered after ranking now,
so relevance drives and the tabloids still get dropped.

Two queries per team-match instead of one, then merged. One binds to the fixture
("<team> team news <opponent>"), the other to the press conference. The old
single OR-soup query was bound to neither.

Team relevance is now required. Either the club is named in the title or a squad
member appears in the body. That one rule kills most of the off-team results.

European fixtures get penalised. The window straddles midweek continental games
and that coverage swamps domestic previews, so when the target is a league
fixture, European markers count against an article.

Content gets trimmed before it goes in the prompt, down to sentences that
mention a squad player or a selection keyword. Quote verification still runs
against the full retrieved text, so trimming can't cause a false rejection.

Window runs LOOKBACK_DAYS (5) back from kickoff to the day before, clamped so it
can't reach past the club's previous fixture. In a congested week that leaves
1-2 days, which is also what keeps midweek European coverage out.

No network I/O unless you pass allow_network=True.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

from llm_keys import (TAVILY_CREDITS_ADVANCED, TAVILY_CREDITS_BASIC,
                      CreditsExhausted, ExtractionFailure, tavily_pool, with_retry)

HERE = Path(__file__).resolve().parent
CACHE_VERSION = "v3"                       # window + disambiguation change
CACHE_DIR = HERE / f".newscache_{CACHE_VERSION}"

LOOKBACK_DAYS = 5
MIN_SCORE = 5
KEEP_ARTICLES = 8
MIN_ROSTER_MENTIONS = 2

# Applied AFTER ranking. Broad enough to admit the club beat outlets, strict
# enough to keep aggregators and tabloid rumour mills out.
ALLOWED_DOMAINS = {
    # national
    "skysports.com", "bbc.com", "bbc.co.uk", "theguardian.com", "theathletic.com",
    "espn.com", "espn.co.uk", "independent.co.uk", "telegraph.co.uk",
    "standard.co.uk", "football365.com", "goal.com", "eurosport.com",
    "premierleague.com", "90min.com", "givemesport.com",
    # club beat reporters -- the richest team-news source for smaller clubs
    "football.london", "liverpoolecho.co.uk", "manchestereveningnews.co.uk",
    "birminghammail.co.uk", "chroniclelive.co.uk", "yorkshireeveningpost.co.uk",
    "lancs.live", "sunderlandecho.com", "nottinghampost.com", "expressandstar.com",
    "bournemouthecho.co.uk", "theargus.co.uk", "hulldailymail.co.uk",
    "gloucestershirelive.co.uk", "examinerlive.co.uk", "leicestermercury.co.uk",
    "grimsbytelegraph.co.uk", "walesonline.co.uk",
    # official club channels -- authoritative on their own team news
    "arsenal.com", "avfc.co.uk", "afcb.co.uk", "brentfordfc.com",
    "brightonandhovealbion.com", "burnleyfootballclub.com", "chelseafc.com",
    "cpfc.co.uk", "evertonfc.com", "fulhamfc.com", "leedsunited.com",
    "liverpoolfc.com", "mancity.com", "manutd.com", "nufc.co.uk",
    "nottinghamforest.co.uk", "tottenhamhotspur.com", "safc.com", "whufc.com",
    "wolves.co.uk",
}

PREFER = [
    "team news", "injury news", "injury update", "predicted line", "predicted xi",
    "expected xi", "expected line", "press conference", "preview", "squad news",
    "team selection", "could start", "set to start", "doubt", "fitness",
    "line-up", "lineup", "rotation", "rested", "available", "ruled out",
    "returns", "recalled", "dropped", "benched", "starting xi",
]

# Any of these in a title rejects the article outright -- no score can rescue it.
HARD_EXCLUDE = [
    "racing", "derby hero", "chester", "cheltenham", "ascot",       # horse racing
    "live commentary", "live blog", "as it happened", "minute-by-minute",
    "watch ", "highlights", "iconic", "greatest", "best goals",
    "best bets", "predictions and", "betting", "odds", "accumulator", "quiz",
    "player ratings", "match report", "full-time", "full time", "reaction",
    "wsl", "women", "u21", "u18", "u23", "academy", "youth cup",
    "transfer rumour", "gossip", "linked with", "weighing up", "eye move",
    # fantasy football advice reads like selection talk but is written by
    # columnists about FPL squads, not by managers about their own XI
    "fpl", "fantasy", "gameweek", "captaincy", "differential", "wildcard",
    "tips and advice", "price change", "bandwagon",
    # found leaking through in the rev-3 pilot
    "the ashes", "cricket", "test match", "rugby", "f1", "grand prix", "golf",
    "pick your", "combined xi", "all-time", "greatest ever", "vote",
    "manager of the month", "player of the month", "award",
    "deadline day", "done deals", "transfer window", "sack", "next manager",
    "how to watch", "tv channel", "kick-off time and more",
]

# Marks coverage of a continental fixture rather than the domestic one.
EURO_MARKERS = [
    "champions league", "europa league", "conference league", "uefa",
    "real madrid", "barcelona", "bayern", "psg", "paris saint", "inter milan",
    "juventus", "atletico", "atalanta", "napoli", "dortmund", "ajax",
    "sporting", "benfica", "porto", "galatasaray", "olympiacos",
]

# Every Premier League club, so an article plainly owned by a club that is
# neither us nor our opponent can be rejected. The rev-3 pilot returned
# "Rosenior provides the latest Chelsea team news ahead of Wolves" on a Wolves
# query: the title named us, so the opponent check passed, but the article
# belonged to Chelsea.
PL_CLUBS = [
    "arsenal", "aston villa", "bournemouth", "brentford", "brighton", "burnley",
    "chelsea", "crystal palace", "everton", "fulham", "leeds", "liverpool",
    "man city", "manchester city", "man utd", "manchester united", "newcastle",
    "nottingham forest", "nott'm forest", "tottenham", "spurs", "sunderland",
    "west ham", "wolves", "wolverhampton",
]

SCORE_RE = re.compile(r"\b\d+\s*[-–]\s*\d+\b")
SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Article:
    title: str
    url: str
    content: str
    published: str = ""
    score: int = 0
    domain: str = ""


@dataclass
class Corpus:
    team: str
    team_code: int
    gameweek: int
    match_id: str
    match_date: str
    window_start: str
    window_end: str
    articles: list[Article] = field(default_factory=list)
    status: str = "ok"
    error: str = ""
    credits_spent: int = 0
    n_raw: int = 0                       # how many results the searches returned
    n_kept: int = 0

    def as_prompt_text(self, roster: list[str] | None = None,
                       max_chars: int = 9000) -> str:
        """
        Numbered sources, trimmed to the passages that matter.

        Sending whole articles buries two useful sentences inside eight hundred
        words of match narrative; the model then quotes the narrative.
        """
        parts, total = [], 0
        for i, a in enumerate(self.articles):
            body = relevant_excerpt(a.content, roster) if roster else a.content
            block = (f"[SOURCE {i}] {a.title}\n"
                     f"PUBLISHED: {a.published}\n{body.strip()}\n")
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts) if parts else "(no pre-match sources retrieved)"

    def drop_excluded(self) -> int:
        """
        Re-apply the current HARD_EXCLUDE list to already-cached articles.

        Cached corpora were filtered by whatever rules were in force when they
        were fetched; this lets a later tightening take effect without spending
        credits to search again. Returns the number removed.
        """
        before = len(self.articles)
        self.articles = [a for a in self.articles
                         if not any(k in a.title.lower() for k in HARD_EXCLUDE)]
        self.n_kept = len(self.articles)
        return before - len(self.articles)

    def searchable_text(self) -> str:
        """FULL text for quote verification -- never the trimmed version, so a
        model quoting a passage we trimmed away is still credited."""
        return "\n".join(f"{a.title}\n{a.content}" for a in self.articles)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["articles"] = [a.__dict__ for a in self.articles]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Corpus":
        arts = [Article(**a) for a in d.pop("articles", [])]
        return Corpus(articles=arts, **d)


# --- window ---

def search_window(match_date: str, prev_match_date: str | None = None) -> tuple[str, str]:
    m = datetime.strptime(match_date[:10], "%Y-%m-%d")
    end = m - timedelta(days=1)
    start = m - timedelta(days=LOOKBACK_DAYS)
    if prev_match_date:
        p = datetime.strptime(prev_match_date[:10], "%Y-%m-%d")
        start = max(start, p + timedelta(days=1))
    if start > end:
        start = end
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# --- content trimming ---

def surnames(roster: list[str]) -> set[str]:
    out = set()
    for n in roster or []:
        for tok in re.split(r"[\s.'-]+", str(n)):
            if len(tok) >= 4:
                out.add(tok.lower())
    return out


def relevant_excerpt(content: str, roster: list[str] | None,
                     max_sents: int = 8) -> str:
    """Keep only sentences naming a squad player or carrying selection language."""
    names = surnames(roster)
    keep = []
    for s in SENT_RE.split(content or ""):
        low = s.lower()
        if any(n in low for n in names) or any(k in low for k in PREFER):
            keep.append(s.strip())
        if len(keep) >= max_sents:
            break
    return " ".join(keep) if keep else (content or "")[:400]


# --- scoring ---

def score_article(title: str, content: str, team: str,
                  roster: list[str] | None = None,
                  domestic: bool = True, opponent: str | None = None) -> int:
    """
    Score an article for relevance to THIS club's team selection.

    The v2 pilot retrieved the opponent's team news for 6 of 15 fixtures: a title
    like "Eddie Howe press conference: Chelsea vs Newcastle" names both clubs, so
    a Newcastle article scored full marks on a Chelsea query. Titles cannot
    settle ownership when both clubs appear in them -- the body has to, by how
    many of OUR squad it actually discusses.
    """
    t, c = title.lower(), (content or "").lower()
    head = c[:800]

    if any(k in t for k in HARD_EXCLUDE):
        return -99
    if SCORE_RE.search(title):
        return -99

    names = surnames(roster)
    club = team.lower().split()[0]
    opp = opponent.lower().split()[0] if opponent else None
    body = c[:2500]

    named = sum(1 for n in names if n in body)          # OUR players, in the body
    on_team = club in t or club in head

    if not on_team and named == 0:
        return -99

    # Both clubs in the title -> the title proves nothing. Demand body evidence
    # that this article is about our squad, not the opponent's.
    ambiguous = bool(opp and opp in t and club in t)
    if ambiguous and named < MIN_ROSTER_MENTIONS:
        return -99
    # Opponent named in the title but we are not -> it is their article.
    if opp and opp in t and club not in t and named < MIN_ROSTER_MENTIONS:
        return -99

    # A third club owns the title (neither us nor our opponent) -> not our news.
    others = [k for k in PL_CLUBS
              if k in t and k not in club and not (opp and k in opp)
              and club not in k and not (opp and opp in k)]
    if others and named < MIN_ROSTER_MENTIONS:
        return -99

    s = 0
    s += 0 if ambiguous else (4 if club in t else 0)
    s += 2 * sum(1 for k in PREFER if k in t)
    s += min(3, sum(1 for k in PREFER if k in head))
    s += min(6, 2 * named)                               # body evidence dominates
    if domestic:
        s -= 4 * sum(1 for k in EURO_MARKERS if k in t)
        s -= min(3, sum(1 for k in EURO_MARKERS if k in head))
    return s


def filter_articles(raw: list[dict], team: str, roster: list[str] | None = None,
                    domestic: bool = True, keep: int = KEEP_ARTICLES,
                    opponent: str | None = None) -> list[Article]:
    scored = []
    for r in raw:
        title, content = str(r.get("title", "")), str(r.get("content", ""))
        if not title or not content:
            continue
        dom = urlparse(str(r.get("url", ""))).netloc.replace("www.", "").lower()
        if dom not in ALLOWED_DOMAINS:
            continue
        s = score_article(title, content, team, roster, domestic, opponent)
        if s < MIN_SCORE:
            continue
        scored.append(Article(title=title, url=str(r.get("url", "")), content=content,
                              published=str(r.get("published_date", "") or ""),
                              score=s, domain=dom))
    scored.sort(key=lambda a: -a.score)
    out, seen = [], set()
    for a in scored:
        k = a.title.lower()[:60]
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
        if len(out) >= keep:
            break
    return out


def build_queries(team: str, opponent: str | None) -> list[str]:
    """Two complementary queries: one bound to the fixture, one to the presser."""
    opp = f" {opponent}" if opponent else ""
    return [
        f'"{team}" team news injury update predicted starting lineup{opp}',
        f'"{team}" manager press conference squad news team selection',
    ]


# --- cache ---

def cache_path(team_code: int, gameweek: int, match_id: str | None = None) -> Path:
    """
    One file per (team, gameweek, FIXTURE).

    match_id belongs in the key because a club can play two Premier League
    fixtures inside the same gameweek when a match is rescheduled: ten clubs do
    so in 2025-26 (GW26, GW33, GW36), for twenty team-matches. Keying on
    (team, gameweek) alone made the second fixture silently reuse the first
    fixture's corpus, and told the extraction prompt the wrong date.

    Passing match_id=None yields the legacy key, which `load_cached` still reads
    -- but only after checking that the cached corpus really belongs to the
    fixture being asked for.
    """
    if match_id is None:
        return CACHE_DIR / f"gw{gameweek:02d}_team{team_code}.json"
    tag = hashlib.sha1(str(match_id).encode("utf-8")).hexdigest()[:8]
    return CACHE_DIR / f"gw{gameweek:02d}_team{team_code}_{tag}.json"


def load_cached(team_code: int, gameweek: int,
                match_id: str | None = None) -> Corpus | None:
    """
    Read a cached corpus, refusing one that was fetched for a different fixture.

    The legacy fallback keeps ~730 already-paid-for corpora usable without
    re-spending Tavily credits, while the match_id guard means a colliding
    fixture returns None (i.e. "not cached") instead of the wrong article set.
    """
    p = cache_path(team_code, gameweek, match_id)
    if p.exists():
        return Corpus.from_dict(json.loads(p.read_text(encoding="utf-8")))
    legacy = cache_path(team_code, gameweek)
    if not legacy.exists():
        return None
    c = Corpus.from_dict(json.loads(legacy.read_text(encoding="utf-8")))
    if match_id is not None and str(c.match_id) != str(match_id):
        return None
    return c


def is_cached(team_code: int, gameweek: int, match_id: str | None = None) -> bool:
    """True only if a corpus for THIS fixture is already on disk (legacy key included)."""
    return load_cached(team_code, gameweek, match_id) is not None


def save_cached(c: Corpus) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(c.team_code, c.gameweek, c.match_id).write_text(
        json.dumps(c.to_dict(), indent=1), encoding="utf-8")


# --- retrieval ---

def _tavily_search(query: str, start: str, end: str, depth: str) -> list[dict]:
    """One Tavily call. RAISES on failure so with_retry() actually retries."""
    from tavily import TavilyClient
    key = tavily_pool().acquire()
    client = TavilyClient(api_key=key.value)
    resp = client.search(query=query, topic="news", search_depth=depth,
                         max_results=15, start_date=start, end_date=end)
    if not isinstance(resp, dict):
        raise RuntimeError(f"unexpected Tavily response type: {type(resp)}")
    return resp.get("results", [])


def fetch_team_context(team: str, team_code: int, gameweek: int, match_id: str,
                       match_date: str, prev_match_date: str | None = None,
                       opponent: str | None = None, roster: list[str] | None = None,
                       *, allow_network: bool = False, depth: str = "advanced",
                       refresh: bool = False) -> Corpus:
    if not refresh:
        hit = load_cached(team_code, gameweek, match_id)
        if hit is not None:
            return hit

    start, end = search_window(match_date, prev_match_date)
    if end >= match_date[:10]:
        raise AssertionError(
            f"window leak: end={end} is not strictly before kickoff {match_date[:10]}")

    if not allow_network:
        raise RuntimeError(
            f"No cached corpus for {team} GW{gameweek} and allow_network=False. "
            "Pass allow_network=True to spend Tavily credits.")

    c = Corpus(team=team, team_code=team_code, gameweek=gameweek, match_id=match_id,
               match_date=match_date, window_start=start, window_end=end)
    per = TAVILY_CREDITS_ADVANCED if depth == "advanced" else TAVILY_CREDITS_BASIC
    raw: list[dict] = []
    try:
        for q in build_queries(team, opponent):
            raw.extend(with_retry(_tavily_search, q, start, end, depth))
            c.credits_spent += per
        # de-duplicate by URL across the two queries before scoring
        seen, uniq = set(), []
        for r in raw:
            u = str(r.get("url", ""))
            if u and u not in seen:
                seen.add(u)
                uniq.append(r)
        c.n_raw = len(uniq)
        c.articles = filter_articles(uniq, team, roster, opponent=opponent)
        c.n_kept = len(c.articles)
        c.status = "ok" if c.articles else "empty"
    except CreditsExhausted:
        # Do NOT cache: an uncached fixture is simply retried on resume, whereas
        # caching an empty corpus here would permanently skip it.
        raise
    except ExtractionFailure as e:
        c.status, c.error = "failed", str(e)
    save_cached(c)
    return c


def estimate_cost(n_team_matches: int, depth: str = "advanced",
                  queries_each: int = 2) -> dict:
    per = TAVILY_CREDITS_ADVANCED if depth == "advanced" else TAVILY_CREDITS_BASIC
    n = n_team_matches * queries_each
    return {"searches": n, "credits": n * per,
            "minutes_at_100rpm": round(n / 100, 1)}
