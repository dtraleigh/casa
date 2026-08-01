"""Conversation persistence and history assembly for Mycroft.

Keeps DB concerns out of the view: creating/fetching conversations (scoped to
their owner), writing messages, and assembling the sliding-window history sent
to Ollama. Phase 3 (tool messages) and Phase 4 (embeddings, auto-learning) hook
in here rather than in the HTTP layer.
"""
import logging

from django.conf import settings
from pgvector.django import CosineDistance

from ai_lab_chatbot.models import Conversation, Knowledge, Message
from ai_lab_chatbot.mycroft.client import embed_text

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
                prompt_tokens=None, completion_tokens=None):
    """Persist one turn. Token counts are set only on assistant turns backed by
    a real Ollama completion; they stay null otherwise."""
    return Message.objects.create(
        conversation=conversation, role=role, content=content,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
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
