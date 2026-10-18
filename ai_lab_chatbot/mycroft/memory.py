"""Conversation persistence and history assembly for Mycroft.

Keeps DB concerns out of the view: creating/fetching conversations (scoped to
their owner), writing messages, and assembling the sliding-window history sent
to Ollama. Phase 4 (tool messages), Phase 3 (embeddings, retrieval), and Phase 3c
(guided learning) hook in here rather than in the HTTP layer.
"""
import json
import logging
import re

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from pgvector.django import CosineDistance

from ai_lab_chatbot.models import (
    Conversation, HouseholdFact, Knowledge, Message, MycroftConfig,
)
from ai_lab_chatbot.mycroft.client import complete_chat, embed_text
from ai_lab_chatbot.mycroft.prompts import (
    build_fact_update_prompt, build_system_prompt,
)

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
    # A window that starts mid tool exchange would open with tool results whose
    # call fell off the front, which is meaningless to the model, so drop them.
    while recent and recent[0].role == 'tool':
        recent.pop(0)
    return [prompt_message(m) for m in recent]


def prompt_message(message):
    """One stored turn in Ollama's message format. Tool turns (Phase 4) carry
    their calls / tool name so the model sees its earlier tool use."""
    out = {'role': message.role, 'content': message.content}
    if message.tool_calls:
        out['tool_calls'] = [
            {'function': {'name': c.get('name', ''),
                          'arguments': c.get('arguments') or {}}}
            for c in message.tool_calls
        ]
    if message.role == 'tool' and message.tool_name:
        out['tool_name'] = message.tool_name
    return out


def add_message(conversation, role, content,
                prompt_tokens=None, completion_tokens=None, model='',
                tool_calls=None, tool_name=''):
    """Persist one turn. Token counts and `model` are set only on assistant turns
    backed by a real Ollama completion; they stay null/blank otherwise.
    `tool_calls` goes on an assistant turn that requested tools and `tool_name`
    on a 'tool' result turn (Phase 4)."""
    return Message.objects.create(
        conversation=conversation, role=role, content=content,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        model=model, tool_calls=tool_calls or None, tool_name=tool_name,
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


# --- Guided learning (Phase 3c) -------------------------------------------

# Cap on each kind of change accepted from one answer, so a runaway reply can't
# flood the shared table.
_MAX_LEARNED_PER_EXCHANGE = 5

LEARNING_TITLE = 'Teaching Mycroft'
# Used when the opening question can't be generated (Ollama down, etc.).
FALLBACK_OPENER = "Let's start with the basics — who lives in your household?"
# Synthetic user turn that kicks off the interview. Never persisted.
_OPENER_PROMPT = (
    "Start the interview: greet me in one sentence and ask your first question."
)


def _empty_result():
    return {'added': [], 'updated': [], 'retired': []}


def _normalize_fact(text):
    """Comparison key for exact-text dedup: lowercased, whitespace-collapsed,
    trailing period stripped. Two facts with the same key are treated as one."""
    return re.sub(r'\s+', ' ', text).strip().rstrip('.').lower()


def _as_id(value):
    """An int id from the model's output (it may quote it), else None."""
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_fact_ops(raw):
    """Pull {"add", "update", "retire"} out of the model's reply, defensively.

    The prompt asks for a bare JSON object, but models wander — code fences,
    preamble. We slice from the first '{' to the last '}' and parse that,
    keeping only well-typed entries and capping each list. Any parse failure
    yields empty ops (change nothing this turn) rather than raising.
    """
    ops = {'add': [], 'update': [], 'retire': []}
    if not raw:
        return ops
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end <= start:
        return ops
    try:
        data = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return ops
    if not isinstance(data, dict):
        return ops

    def as_list(key):
        value = data.get(key)
        return value if isinstance(value, list) else []

    for item in as_list('add'):
        if isinstance(item, str) and item.strip():
            ops['add'].append(item.strip())
    for item in as_list('update'):
        if not isinstance(item, dict):
            continue
        fact_id = _as_id(item.get('id'))
        content = item.get('content')
        if fact_id is not None and isinstance(content, str) and content.strip():
            ops['update'].append({'id': fact_id, 'content': content.strip()})
    for item in as_list('retire'):
        fact_id = _as_id(item)
        if fact_id is not None:
            ops['retire'].append(fact_id)

    return {k: v[:_MAX_LEARNED_PER_EXCHANGE] for k, v in ops.items()}


def start_learning_conversation(user):
    """Create a "Teach Mycroft" conversation seeded with Mycroft's opening
    question, so the interview starts with Mycroft asking. Generating the opener
    is best-effort — on failure a canned question is used instead."""
    model = MycroftConfig.default_model()
    conversation = Conversation.objects.create(
        user_id=user.id, username=user.username,
        title=LEARNING_TITLE, model=model, mode='learning',
    )
    opener = ''
    try:
        opener = complete_chat(
            [
                {'role': 'system',
                 'content': build_system_prompt(user, mode='learning')},
                {'role': 'user', 'content': _OPENER_PROMPT},
            ],
            model=model,
        )
    except Exception:
        logger.exception("Mycroft learning opener failed")
    if opener:
        add_message(conversation, 'assistant', opener, model=model)
    else:
        add_message(conversation, 'assistant', FALLBACK_OPENER)
    return conversation


def apply_learning(conversation, user):
    """Turn the latest answer in a learning conversation into HouseholdFact
    changes: new facts are added, changed facts are superseded by their
    replacement, and facts no longer true are retired (kept for history, out of
    prompts).

    A no-op for ordinary chats — only learning conversations write facts. Only
    the user's message is a source of facts; the assistant message before it
    (the question) is context. Best-effort and off the streaming path: any
    failure is logged and swallowed. Returns
    {'added': [str], 'updated': [{'from', 'to'}], 'retired': [str]}.
    """
    result = _empty_result()
    if conversation.mode != 'learning':
        return result
    try:
        user_msg = conversation.messages.filter(role='user').last()
        if not user_msg:
            return result
        # The question being answered: the last assistant turn before it.
        question_msg = (
            conversation.messages
            .filter(role='assistant', id__lt=user_msg.id)
            .last()
        )

        # Small table — load all current facts.
        current = {
            f.id: f for f in HouseholdFact.objects.filter(is_current=True)
        }
        raw = complete_chat(
            build_fact_update_prompt(
                question_msg.content if question_msg else '',
                user_msg.content,
                list(current.values()),
                user.username,
            ),
            model=conversation.model or None,
        )
        ops = _parse_fact_ops(raw)

        attribution = {
            'source': 'learned',
            'source_user_id': user.id,
            'source_username': user.username,
        }
        seen = {_normalize_fact(f.content) for f in current.values()}
        now = timezone.now()

        with transaction.atomic(using=HouseholdFact.objects.db):
            for content in ops['add']:
                key = _normalize_fact(content)
                if key in seen:
                    continue
                seen.add(key)
                HouseholdFact.objects.create(content=content, **attribution)
                result['added'].append(content)

            for op in ops['update']:
                old = current.get(op['id'])
                if old is None or not old.is_current:
                    continue
                if _normalize_fact(op['content']) == _normalize_fact(old.content):
                    continue
                new = HouseholdFact.objects.create(
                    content=op['content'], **attribution
                )
                old.is_current = False
                old.superseded_by = new
                old.superseded_at = now
                old.save(update_fields=[
                    'is_current', 'superseded_by', 'superseded_at', 'updated_at',
                ])
                seen.add(_normalize_fact(op['content']))
                result['updated'].append({'from': old.content, 'to': new.content})

            for fact_id in ops['retire']:
                old = current.get(fact_id)
                if old is None or not old.is_current:
                    continue
                old.is_current = False
                old.superseded_at = now
                old.save(update_fields=[
                    'is_current', 'superseded_at', 'updated_at',
                ])
                result['retired'].append(old.content)

        return result
    except Exception:
        logger.exception("Mycroft learning failed")
        return _empty_result()
