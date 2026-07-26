"""Ollama client wrapper for Mycroft.

Thin layer over the `ollama` package. Knows how to stream a chat completion;
it does not manage conversation history or prompts (that's the view's job).
"""
from django.conf import settings
from ollama import Client


def _client() -> Client:
    return Client(host=settings.OLLAMA_HOST)


def stream_chat(messages, stats_out=None):
    """Stream a chat completion from Ollama.

    `messages` is a list of {"role": ..., "content": ...} dicts, with the
    system prompt already prepended. Yields response text chunks as they
    generate.

    If `stats_out` is a dict, it's populated from Ollama's final (`done=True`)
    chunk with the per-request metrics — prompt/completion token counts and
    durations (converted from Ollama's nanoseconds to milliseconds). The caller
    reads it after the generator is exhausted; the yield contract (text only) is
    unchanged.
    """
    stream = _client().chat(
        model=settings.OLLAMA_CHAT_MODEL,
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


def complete_chat(messages) -> str:
    """Non-streaming chat completion, returned as a single string.

    Used for short off-band calls (e.g. generating a conversation title) that
    shouldn't tie up the streaming path.
    """
    resp = _client().chat(
        model=settings.OLLAMA_CHAT_MODEL,
        messages=messages,
        stream=False,
    )
    return (resp.message.content or '').strip()
