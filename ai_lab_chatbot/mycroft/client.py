"""Ollama client wrapper for Mycroft.

Thin layer over the `ollama` package. Knows how to stream a chat completion;
it does not manage conversation history or prompts (that's the view's job).
"""
from django.conf import settings
from ollama import Client


def _client() -> Client:
    return Client(host=settings.OLLAMA_HOST)


def list_models():
    """Installed Ollama chat models, sorted, for the model selector.

    Enumerates `_client().list()`, then keeps only chat-capable models (see
    `_supports_chat`) so embedding-only models like nomic-embed-text never appear
    in the selector. Best-effort: any failure (Ollama down, unexpected shape)
    yields [] so page render never depends on Ollama being reachable.
    """
    try:
        resp = _client().list()
    except Exception:
        return []
    names = []
    for entry in getattr(resp, 'models', None) or []:
        name = getattr(entry, 'model', None)
        if name is None and isinstance(entry, dict):
            name = entry.get('model') or entry.get('name')
        if name:
            names.append(name)
    return sorted(n for n in names if _supports_chat(n))


def _base_name(name):
    """Model name without its `:tag` (llama3.1:8b -> llama3.1)."""
    return (name or '').split(':', 1)[0]


def _supports_chat(name):
    """Whether an installed model can do chat completion.

    Asks Ollama for the model's capabilities (`ollama show`): a chat model
    reports 'completion'; an embedding-only model reports just 'embedding'. If
    capabilities can't be determined (older Ollama, or show fails), fall back to
    excluding the configured embed model by base name (tag-tolerant) so nothing
    regresses.
    """
    try:
        info = _client().show(name)
        caps = getattr(info, 'capabilities', None)
        if caps is None and isinstance(info, dict):
            caps = info.get('capabilities')
    except Exception:
        caps = None
    if caps:
        return 'completion' in caps
    return _base_name(name) != _base_name(settings.OLLAMA_EMBED_MODEL)


def stream_chat(messages, stats_out=None, model=None):
    """Stream a chat completion from Ollama.

    `messages` is a list of {"role": ..., "content": ...} dicts, with the
    system prompt already prepended. `model` overrides the configured default
    (`OLLAMA_CHAT_MODEL`) — used by the per-conversation model selector. Yields
    response text chunks as they generate.

    If `stats_out` is a dict, it's populated from Ollama's final (`done=True`)
    chunk with the per-request metrics — prompt/completion token counts and
    durations (converted from Ollama's nanoseconds to milliseconds). The caller
    reads it after the generator is exhausted; the yield contract (text only) is
    unchanged.
    """
    stream = _client().chat(
        model=model or settings.OLLAMA_CHAT_MODEL,
        messages=messages,
        stream=True,
        options={'num_ctx': settings.MYCROFT_NUM_CTX},
    )
    for chunk in stream:
        piece = chunk.message.content
        if piece:
            yield piece
        if getattr(chunk, 'done', False) and stats_out is not None:
            stats_out.update({
                'prompt_tokens': chunk.prompt_eval_count,
                'completion_tokens': chunk.eval_count,
                'eval_duration_ms': (chunk.eval_duration or 0) / 1e6,
                'total_duration_ms': (chunk.total_duration or 0) / 1e6,
            })


def embed_text(text) -> list[float]:
    """Embed one string with the configured embedding model, returning the raw
    vector. Raises on failure — callers in `memory.py` make embedding
    best-effort so a stumble never breaks the chat path.
    """
    resp = _client().embed(model=settings.OLLAMA_EMBED_MODEL, input=text)
    return list(resp.embeddings[0])


def complete_chat(messages, model=None) -> str:
    """Non-streaming chat completion, returned as a single string.

    Used for short off-band calls (e.g. generating a conversation title,
    auto-learning extraction) that shouldn't tie up the streaming path. `model`
    overrides the configured default, so these calls can match the
    conversation's selected model.
    """
    resp = _client().chat(
        model=model or settings.OLLAMA_CHAT_MODEL,
        messages=messages,
        stream=False,
    )
    return (resp.message.content or '').strip()
