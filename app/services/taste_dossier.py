import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from app.config import settings
from app.database import get_db

logger = logging.getLogger(__name__)

class TasteDossierService:
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = settings.OPENROUTER_API_KEY or "dummy_key_for_init"
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.OPENROUTER_BASE_URL
            )
        return self._client

    @staticmethod
    def _format_age(ts_str: Optional[str]) -> str:
        """Return human-readable relative age for a rating timestamp."""
        if not ts_str:
            return "rated long ago"
        try:
            dt = datetime.fromisoformat(str(ts_str).strip().replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days = max(0, int((now - dt).total_seconds() // 86400))
            if days == 0:
                return "rated today"
            elif days == 1:
                return "rated 1 day ago"
            elif days < 30:
                return f"rated {days} days ago"
            elif days < 365:
                months = max(1, round(days / 30.4))
                return f"rated {months} month{'s' if months != 1 else ''} ago"
            else:
                years = max(1, round(days / 365.25))
                return f"rated {years} year{'s' if years != 1 else ''} ago"
        except Exception:
            return "rated long ago"

    @staticmethod
    def mark_dirty(user_id: str = "default_user"):
        """Flag the user's taste dossier as needing re-synthesis."""
        with get_db() as conn:
            conn.execute(
                "UPDATE taste_dossiers SET is_dirty = 1 WHERE user_id = ?",
                (user_id,)
            )

    async def get_or_update_dossier(self, user_id: str = "default_user", force: bool = False) -> Dict[str, Any]:
        """Fetch current taste dossier, re-synthesizing via GPT-4o if dirty."""
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM taste_dossiers WHERE user_id = ?", (user_id,)
            ).fetchone()

            ratings = conn.execute("""
                SELECT r.score, r.aspect_tags, r.notes, r.created_at, r.updated_at,
                       t.title, t.media_type, t.release_year, t.genres,
                       t.director_or_creator, t.cast_top
                FROM ratings r
                JOIN titles t ON r.title_id = t.id
                ORDER BY COALESCE(r.updated_at, r.created_at) DESC
            """).fetchall()

        ratings_count = len(ratings)

        if row and not row["is_dirty"] and not force and ratings_count == row["ratings_count_at_synthesis"]:
            return {
                "user_id": user_id,
                "core_loves": json.loads(row["core_loves"] or "[]"),
                "deal_breakers": json.loads(row["deal_breakers"] or "[]"),
                "creator_affinities": json.loads(row["creator_affinities"] or "[]"),
                "atmospheric_preferences": json.loads(row["atmospheric_preferences"] or "[]"),
                "narrative_tropes": json.loads(row["narrative_tropes"] or "[]"),
                "full_summary": row["full_summary"] or "",
                "ratings_count_at_synthesis": row["ratings_count_at_synthesis"],
                "is_dirty": False,
                "updated_at": row["updated_at"]
            }

        # If less than 3 ratings, use a friendly starter summary without calling LLM
        if ratings_count < 3:
            summary = (
                f"You have logged {ratings_count} rating{'s' if ratings_count != 1 else ''} so far. "
                "CineMatch needs at least 3 ratings to unlock deep pattern analysis. "
                "Rate a few more movies or series in the Rating Deck or Search to synthesize your taste dossier!"
            )
            return {
                "user_id": user_id,
                "core_loves": [],
                "deal_breakers": [],
                "creator_affinities": [],
                "atmospheric_preferences": [],
                "narrative_tropes": [],
                "full_summary": summary,
                "ratings_count_at_synthesis": ratings_count,
                "is_dirty": False,
                "updated_at": None
            }

        if not settings.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set. Cannot synthesize taste dossier with GPT-4o.")
            return {
                "user_id": user_id,
                "core_loves": ["Rich character development", "Strong visual identity"],
                "deal_breakers": ["Predictable storytelling"],
                "creator_affinities": [],
                "atmospheric_preferences": ["Atmospheric immersion"],
                "narrative_tropes": [],
                "full_summary": f"Taste dossier generated from {ratings_count} ratings. Configure your OPENROUTER_API_KEY in .env to activate deep GPT-4o taste analysis.",
                "ratings_count_at_synthesis": ratings_count,
                "is_dirty": False,
                "updated_at": None
            }

        # Prepare ratings summary for GPT-4o
        loved_titles = []
        liked_titles = []
        mediocre_titles = []
        disliked_titles = []

        for r in ratings:
            genres_list = json.loads(r["genres"] or "[]")
            tags_list = json.loads(r["aspect_tags"] or "[]")
            age_label = self._format_age(r["updated_at"] or r["created_at"])
            item_desc = (
                f"'{r['title']}' ({r['release_year'] or 'Unknown'}, {r['media_type']}) - "
                f"Genres: {', '.join(genres_list)} | Director: {r['director_or_creator'] or 'N/A'} | "
                f"Rating: {r['score']}/6 ({age_label})"
            )
            if tags_list:
                item_desc += f" | Tags: {', '.join(tags_list)}"
            if r["notes"]:
                item_desc += f" | Note: '{r['notes']}'"

            if r["score"] in (5, 6):
                loved_titles.append(item_desc)
            elif r["score"] == 4:
                liked_titles.append(item_desc)
            elif r["score"] == 3:
                mediocre_titles.append(item_desc)
            else:
                disliked_titles.append(item_desc)

        prompt_content = f"""
You are a master film theorist, narrative analyst, and cinephile psychologist.
Analyze the viewing history and ratings below to synthesize this user's psychological entertainment DNA.
The rating scale is strictly 1 to 6 (1-2 = Hated/Disliked, 3 = Mediocre/Tolerated, 4 = Good, 5 = Great, 6 = Masterpiece/Favorite).
Each rating is annotated with how long ago it was logged (e.g. 'rated 3 days ago'). Weight recent ratings more heavily than older ones when analyzing preferences.

### User's Rating History:
[Masterpieces & Favorites (5-6 / 6)]:
{chr(10).join(loved_titles) if loved_titles else 'None yet'}

[Enjoyed (4 / 6)]:
{chr(10).join(liked_titles) if liked_titles else 'None yet'}

[Mediocre (3 / 6)]:
{chr(10).join(mediocre_titles) if mediocre_titles else 'None yet'}

[Disliked & Hated (1-2 / 6)]:
{chr(10).join(disliked_titles) if disliked_titles else 'None yet'}

Respond ONLY with a valid JSON object matching this schema:
{{
  "core_loves": ["specific narrative, stylistic, or thematic elements they consistently revere"],
  "deal_breakers": ["elements, pacing issues, or cliches that reliably ruin an experience for them"],
  "creator_affinities": ["directors, writers, or actors they gravitate toward"],
  "atmospheric_preferences": ["specific visual moods, color tones, musical textures, or settings"],
  "narrative_tropes": ["story structures or character archetypes they enjoy"],
  "full_summary": "A 2-paragraph evocative, sharp analysis written directly to the user ('Your taste leans toward...'). Highlight what bridges their highest-rated favorites and why their dislikes fell flat. Explicitly note in the summary if recent taste diverges from historical taste (e.g. recent drift toward new genres, directors, or tonal shifts)."
}}
"""

        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": "You analyze user film taste and output strictly valid JSON."},
                    {"role": "user", "content": prompt_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            raw_json = response.choices[0].message.content
            data = json.loads(raw_json)

            core_loves = data.get("core_loves", [])
            deal_breakers = data.get("deal_breakers", [])
            creator_affinities = data.get("creator_affinities", [])
            atmospheric_preferences = data.get("atmospheric_preferences", [])
            narrative_tropes = data.get("narrative_tropes", [])
            full_summary = data.get("full_summary", "")

            with get_db() as conn:
                conn.execute("""
                    INSERT INTO taste_dossiers (
                        user_id, core_loves, deal_breakers, creator_affinities,
                        atmospheric_preferences, narrative_tropes, full_summary,
                        ratings_count_at_synthesis, is_dirty, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        core_loves = excluded.core_loves,
                        deal_breakers = excluded.deal_breakers,
                        creator_affinities = excluded.creator_affinities,
                        atmospheric_preferences = excluded.atmospheric_preferences,
                        narrative_tropes = excluded.narrative_tropes,
                        full_summary = excluded.full_summary,
                        ratings_count_at_synthesis = excluded.ratings_count_at_synthesis,
                        is_dirty = 0,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    user_id,
                    json.dumps(core_loves),
                    json.dumps(deal_breakers),
                    json.dumps(creator_affinities),
                    json.dumps(atmospheric_preferences),
                    json.dumps(narrative_tropes),
                    full_summary,
                    ratings_count
                ))

            return {
                "user_id": user_id,
                "core_loves": core_loves,
                "deal_breakers": deal_breakers,
                "creator_affinities": creator_affinities,
                "atmospheric_preferences": atmospheric_preferences,
                "narrative_tropes": narrative_tropes,
                "full_summary": full_summary,
                "ratings_count_at_synthesis": ratings_count,
                "is_dirty": False,
                "updated_at": None
            }
        except Exception as e:
            logger.error(f"Error synthesizing taste dossier with GPT-4o: {e}")
            return {
                "user_id": user_id,
                "core_loves": [],
                "deal_breakers": [],
                "creator_affinities": [],
                "atmospheric_preferences": [],
                "narrative_tropes": [],
                "full_summary": f"Could not update taste dossier due to API error: {e}",
                "ratings_count_at_synthesis": ratings_count,
                "is_dirty": True,
                "updated_at": None
            }

taste_dossier_service = TasteDossierService()
