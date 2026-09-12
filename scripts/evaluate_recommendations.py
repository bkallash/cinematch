"""Offline, synthetic held-out benchmark. Run with python -m scripts.evaluate_recommendations.

The fixture and baseline are frozen before ranking changes. Unknown outcomes have
zero evaluation gain, but are never treated as dislikes. The provider double is
an identity reranker: this measures retrieval, not real-model comprehension.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config import settings
from app.database import get_db, init_db
from app.services.embeddings import embedding_service
import app.services.orchestrator as orchestration
from app.services.tmdb import tmdb_service


FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/rating_evaluation.json"
GAINS = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3}


class FrozenTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 11, tzinfo=timezone.utc)


def metrics(ids, held_out):
    import math
    positives = {tid for tid, score in held_out.items() if score >= 4}
    gains = [GAINS.get(held_out.get(tid, 0), 0) for tid in ids[:10]]
    ideal = sorted((GAINS[score] for score in held_out.values()), reverse=True)[:10]
    dcg = lambda values: sum(gain / math.log2(i + 2) for i, gain in enumerate(values))
    dislikes = sum(0 < held_out.get(tid, 0) <= 3 for tid in ids[:10])
    return {"recall_at_10": len(set(ids[:10]) & positives) / len(positives) if positives else 0,
            "ndcg_at_10": dcg(gains) / dcg(ideal) if dcg(ideal) else 0,
            "dislikes": dislikes, "returned": len(ids[:10]),
            "dislike_rate": dislikes / len(ids[:10]) if ids[:10] else 0,
            "unknown": sum(tid not in held_out for tid in ids[:10])}


async def evaluate():
    fixture = json.loads(FIXTURE.read_text())
    cases = []
    root = Path(".test-tmp").resolve()
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="held-out-", dir=root) as directory, \
            patch.object(settings, "OPENROUTER_API_KEY", ""), \
            patch.object(settings, "TMDB_API_KEY", ""), \
            patch.object(tmdb_service, "api_key", ""), \
            patch.object(settings, "EMBEDDING_PROVIDER", "local"), \
            patch.object(settings, "EMBEDDING_DIM", 1536), \
            patch.object(orchestration, "datetime", FrozenTime):
        for profile in fixture["profiles"]:
            with patch.object(settings, "DATABASE_PATH", Path(directory) / (profile["name"] + ".db")):
                init_db()
                with get_db() as conn:
                    for title in fixture["titles"]:
                        vec = await embedding_service.get_embedding(embedding_service.build_title_embedding_text(title))
                        conn.execute("""INSERT INTO titles
                            (id, tmdb_id, title, media_type, release_year, overview, genres,
                             vote_average, vote_count, popularity, embedding, embedding_dim, embedding_model)
                            VALUES (?, ?, ?, ?, ?, ?, ?, 7.5, 1000, 50, ?, ?, ?)""",
                            (title["id"], title["id"], title["title"], title["media_type"], 2020,
                             title["overview"], json.dumps(title["genres"]), vec.tobytes(), len(vec), embedding_service.model_key))
                    # Held-out values never enter Ratings, Dossier, direct evidence or caches.
                    for rating in profile["training"]:
                        conn.execute("INSERT INTO ratings (title_id, score, created_at, updated_at) VALUES (?, ?, ?, ?)",
                                     (rating["id"], rating["score"], rating["at"], rating["at"]))
                for media in ("movie", "tv"):
                    service = orchestration.OrchestratorService()
                    captured = []
                    request_bytes = []

                    async def provider(**kwargs):
                        prompt = kwargs["messages"][-1]["content"]
                        marker = "### Candidate"
                        if marker in prompt:
                            request_bytes.append(len(json.dumps(kwargs).encode("utf-8")))
                            tail = prompt.split(marker, 1)[1]
                            candidates, _ = json.JSONDecoder().raw_decode(tail[tail.index("["):])
                            captured.extend(c["title_id"] for c in candidates)
                            data = {"picks": [{"title_id": c["title_id"], "title": c["title"],
                                               "reason": "Selected from supplied catalog evidence."} for c in candidates[:10]],
                                    "rejected_title_ids": []}
                        else:
                            data = {"full_summary": "Synthetic training evidence only."}
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])

                    service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=provider))))
                    # Dossier is built offline, then remains cached during mocked reranking.
                    await orchestration.taste_dossier_service.get_or_update_dossier()
                    started = time.perf_counter()
                    with patch.object(settings, "OPENROUTER_API_KEY", "synthetic-provider"), \
                            patch.object(orchestration.taste_dossier_service, "_client", service._client):
                        result = await service.get_personalized_picks(limit=10, media_type_preference=media)
                    held = {r["id"]: r["score"] for r in profile["held_out"]
                            if fixture["titles"][r["id"] - 1]["media_type"] == media}
                    ids = [p["title_id"] for p in result["picks"]]
                    positives = {tid for tid, score in held.items() if score >= 4}
                    cases.append({"profile": profile["name"], "media": media, "ids": ids,
                                  "candidate_ids": captured, **metrics(ids, held),
                                  "candidate_recall": len(set(captured) & positives) / len(positives),
                                  "reranker_request_bytes": sum(request_bytes),
                                  "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
    def aggregate(rows):
        return {**{key: sum(r[key] for r in rows) / len(rows)
                   for key in ("recall_at_10", "ndcg_at_10", "candidate_recall", "returned")},
                "dislikes": sum(r["dislikes"] for r in rows),
                "dislike_rate": sum(r["dislikes"] for r in rows) / max(1, sum(r["returned"] for r in rows))}
    return {"fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "embedding": "local:token-hash-v1:1536", "schema": embedding_service.model_key.split(":")[-1],
            "reranker": "identity provider double", "shortlist_budget": 15,
            "gain_mapping": GAINS, "dislike_denominator": "all returned top-ten Titles, including unknown outcomes",
            "cases": cases, "aggregate": aggregate(cases),
            "per_media": {media: aggregate([r for r in cases if r["media"] == media]) for media in ("movie", "tv")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    report = asyncio.run(evaluate())
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        assert baseline["fixture_sha256"] == report["fixture_sha256"], "Fixture changed"
        old, new = baseline["aggregate"], report["aggregate"]
        report["acceptance"] = (new["recall_at_10"] >= old["recall_at_10"] and
            new["ndcg_at_10"] >= old["ndcg_at_10"] and new["dislike_rate"] <= old["dislike_rate"] and
            (new["recall_at_10"] > old["recall_at_10"] or new["ndcg_at_10"] > old["ndcg_at_10"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("aggregate", "per_media")}, indent=2))
    if report.get("acceptance") is False:
        raise SystemExit("Held-out aggregate acceptance gate failed; inspect the report.")
