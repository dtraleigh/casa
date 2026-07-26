"""Conversation persistence and history assembly for Mycroft.

Keeps DB concerns out of the view: creating/fetching conversations (scoped to
their owner), writing messages, and assembling the sliding-window history sent
to Ollama. Phase 3 (tool messages) and Phase 4 (embeddings, auto-learning) hook
in here rather than in the HTTP layer.
"""
from django.conf import settings

from ai_lab_chatbot.models import Conversation, Message


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


def add_message(conversation, role, content):
    """Persist one turn."""
    return Message.objects.create(
        conversation=conversation, role=role, content=content
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


def delete_conversation(user, conversation_id):
    """Delete the user's conversation (cascades to its messages). Raises
    Conversation.DoesNotExist if it isn't theirs."""
    conversation = Conversation.objects.get(id=conversation_id, user_id=user.id)
    conversation.delete()
