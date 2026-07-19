"""Ollama client wrapper for Mycroft.

Thin layer over the `ollama` package. Knows how to stream a chat completion;
it does not manage conversation history or prompts (that's the view's job).
"""
from django.conf import settings
from ollama import Client


def _client() -> Client:
    return Client(host=settings.OLLAMA_HOST)


def stream_chat(messages):
    """Stream a chat completion from Ollama.

    `messages` is a list of {"role": ..., "content": ...} dicts, with the
    system prompt already prepended. Yields response text chunks as they
    generate.
    """
    stream = _client().chat(
        model=settings.OLLAMA_CHAT_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        piece = chunk.message.content
        if piece:
            yield piece
