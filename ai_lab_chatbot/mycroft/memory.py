"""Conversation persistence and history assembly for Mycroft.

Keeps DB concerns out of the view: creating/fetching conversations (scoped to
their owner), writing messages, and assembling the sliding-window history sent
to Ollama. Phase 4 (tool messages), Phase 3 (embeddings, retrieval), and Phase 3b
(auto-learning) hook in here rather than in the HTTP layer.
"""
import json
import logging
import re

from django.conf import settings
from pgvector.django import CosineDistance

from ai_lab_chatbot.models import Conversation, HouseholdFact, Knowledge, Message
from ai_lab_chatbot.mycroft.client import complete_chat, embed_text
from ai_lab_chatbot.mycroft.prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


def _window_size():
    return getattr(settings, 'MYCROFT_HISTORY_WINDOW', 20)


def get_or_create_conversation(user, conversation_id):
    """Return the user's conversation.

    Falsy `conversation_id` creates a fresh one (lazy — an abandoned empty chat
    never hits the DB). Otherwise fetch it scoped to `user.id`; a conversation
    owned by someone else (or a bad id) raises Conversation.DoesNotExist, which
    the view turns into a 404.
    """
    if not conversation_id:
        return Conversation.objects.create(
            user_id=user.id, username=user.username
        )
    return Conversation.objects.get(id=conversation_id, user_id=user.id)


def history_for_prompt(conversation, limit=None):
    """The last `limit` messages, oldest-first, as {role, content} dicts — the
    sliding window handed to Ollama alongside the system prompt."""
    if limit is None:
        limit = _window_size()
    # Grab the most recent `limit` (tie-break on id), then restore chronology.
    recent = list(
        conversation.messages.order_by('-created_at', '-id')[:limit]
    )
    recent.reverse()
    return [{'role': m.role, 'content': m.content} for m in recent]


def add_message(conversation, role, content,
                prompt_tokens=None, completion_tokens=None, model=''):
    """Persist one turn. Token counts and `model` are set only on assistant turns
    backed by a real Ollama completion; they stay null/blank otherwise."""
    return Message.objects.create(
        conversation=conversation, role=role, content=content,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        model=model,
    )


def touch(conversation):
    """Bump updated_at so the conversation sorts to the top of history."""
    conversation.save(update_fields=['updated_at'])


def recent_conversations(user, limit=50):
    """The user's conversations, most-recent-activity first."""
    return Conversation.objects.filter(user_id=user.id)[:limit]


def favorite_conversations(user):
    """The user's favorited conversations (all of them — a curated set),
    most-recent-activity first."""
    return Conversation.objects.filter(user_id=user.id, is_favorite=True)


def other_conversations(user, limit=50):
    """The user's non-favorited conversations, most-recent-activity first."""
    return Conversation.objects.filter(user_id=user.id, is_favorite=False)[:limit]


def toggle_favorite(user, conversation_id):
    """Flip a conversation's favorite flag. Scoped to the owner (raises
    Conversation.DoesNotExist otherwise). Uses update_fields to leave updated_at
    untouched so favoriting doesn't reorder history."""
    conversation = Conversation.objects.get(id=conversation_id, user_id=user.id)
    conversation.is_favorite = not conversation.is_favorite
    conversation.save(update_fields=['is_favorite'])
    return conversation


def rename_conversation(user, conversation_id, title):
    """Set a conversation's title (owner-scoped; raises Conversation.DoesNotExist
    otherwise). A blank title is a no-op — the existing name is kept, so an
    accidental empty save can't wipe a title. Renaming isn't activity, so
    updated_at is left untouched via update_fields — same rationale as
    toggle_favorite."""
    conversation = Conversation.objects.get(id=conversation_id, user_id=user.id)
    new_title = (title or '').strip()[:200]
    if new_title:
        conversation.title = new_title
        conversation.save(update_fields=['title'])
    return conversation


def delete_conversation(user, conversation_id):
    """Delete the user's conversation (cascades to its messages). Raises
    Conversation.DoesNotExist if it isn't theirs."""
    conversation = Conversation.objects.get(id=conversation_id, user_id=user.id)
    conversation.delete()


# --- Semantic memory (Phase 3) ---------------------------------------------

def embed_message(message):
    """Embed a message and store the vector, best-effort. Returns the vector, or
    None if embedding failed — a stumble here (Ollama down, model missing) must
    never break the chat path, so the message just stays unretrievable."""
    try:
        vector = embed_text(message.content)
    except Exception:
        logger.exception("Mycroft message embedding failed")
        return None
    message.embedding = vector
    message.save(update_fields=['embedding'])
    return vector


def embed_knowledge(knowledge):
    """Embed a Knowledge row from its topic + content and store the vector,
    best-effort (see embed_message). Returns the vector or None."""
    try:
        vector = embed_text(f"{knowledge.topic}\n{knowledge.content}")
    except Exception:
        logger.exception("Mycroft knowledge embedding failed")
        return None
    knowledge.embedding = vector
    knowledge.save(update_fields=['embedding'])
    return vector


def retrieve_memories(user, query_vec, *, exclude_conversation_id=None):
    """Semantically relevant context for `query_vec`, as (knowledge, messages).

    Knowledge is global (curated, shared). Past messages are scoped to the
    requesting user and exclude the current conversation — its recent turns are
    already in the sliding window, so recall is about *other* conversations. Both
    drop matches beyond MYCROFT_RETRIEVAL_MAX_DISTANCE and rows without an
    embedding. Returns empty lists when there's no query vector to match on.
    """
    if query_vec is None:
        return [], []

    max_distance = settings.MYCROFT_RETRIEVAL_MAX_DISTANCE

    knowledge = list(
        Knowledge.objects
        .exclude(embedding__isnull=True)
        .annotate(distance=CosineDistance('embedding', query_vec))
        .filter(distance__lte=max_distance)
        .order_by('distance')[:settings.MYCROFT_RETRIEVAL_KNOWLEDGE]
    )

    messages_qs = (
        Message.objects
        .filter(conversation__user_id=user.id)
        .exclude(embedding__isnull=True)
    )
    if exclude_conversation_id is not None:
        messages_qs = messages_qs.exclude(conversation_id=exclude_conversation_id)
    messages = list(
        messages_qs
        .annotate(distance=CosineDistance('embedding', query_vec))
        .filter(distance__lte=max_distance)
        .order_by('distance')[:settings.MYCROFT_RETRIEVAL_MESSAGES]
    )

    return knowledge, messages


# --- Auto-learning (Phase 3b) ----------------------------------------------

# Cap on facts accepted from one exchange, so a runaway reply can't flood the
# shared table.
_MAX_LEARNED_PER_EXCHANGE = 5


def _normalize_fact(text):
    """Comparison key for exact-text dedup: lowercased, whitespace-collapsed,
    trailing period stripped. Two facts with the same key are treated as one."""
    return re.sub(r'\s+', ' ', text).strip().rstrip('.').lower()


def _parse_fact_list(raw):
    """Pull a list of fact strings out of the model's reply, defensively.

    The extraction prompt asks for a bare JSON array, but models wander — they
    wrap it in ```json fences or add a sentence of preamble. We slice from the
    first '[' to the last ']' and parse that, keeping only non-empty strings and
    capping the count. Any parse failure yields [] (learn nothing this turn)
    rather than raising.
    """
    if not raw:
        return []
    start = raw.find('[')
    end = raw.rfind(']')
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return []
    if not isinstance(items, list):
        return []
    facts = []
    for item in items:
        if isinstance(item, str) and item.strip():
            facts.append(item.strip())
        if len(facts) >= _MAX_LEARNED_PER_EXCHANGE:
            break
    return facts


def learn_from_exchange(conversation, user):
    """Mine durable HouseholdFacts from a conversation's latest exchange.

    Best-effort and off the streaming path (called from its own endpoint): asks
    Mycroft to extract durable facts from the last user+assistant turns, then
    writes each new one as a `source='learned'` HouseholdFact attributed to
    `user`. Exact-text duplicates of existing facts are skipped. Any failure
    (Ollama down, unparseable reply, DB hiccup) is logged and swallowed —
    auto-learning must never surface an error to the caller. Returns the list of
    newly created fact strings (possibly empty).
    """
    try:
        user_msg = conversation.messages.filter(role='user').last()
        assistant_msg = conversation.messages.filter(role='assistant').last()
        if not (user_msg and assistant_msg):
            return []

        raw = complete_chat(
            build_extraction_prompt(
                user_msg.content, assistant_msg.content, user.username
            ),
            model=conversation.model or None,
        )
        candidates = _parse_fact_list(raw)
        if not candidates:
            return []

        # Small table — load all existing facts and dedup in Python.
        seen = {_normalize_fact(f.content) for f in HouseholdFact.objects.all()}
        created = []
        for content in candidates:
            key = _normalize_fact(content)
            if key in seen:
                continue
            seen.add(key)
            HouseholdFact.objects.create(
                content=content,
                source='learned',
                source_user_id=user.id,
                source_username=user.username,
            )
            created.append(content)
        return created
    except Exception:
        logger.exception("Mycroft auto-learning failed")
        return []
