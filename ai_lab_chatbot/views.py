import json
import logging

from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from ai_lab_chatbot.mycroft.client import stream_chat
from ai_lab_chatbot.mycroft.prompts import build_system_prompt

logger = logging.getLogger(__name__)

# Roles the browser is allowed to send in the history.
_ALLOWED_ROLES = {'user', 'assistant'}


def _frame(type_, **fields):
    """One newline-delimited JSON frame of the response stream."""
    return json.dumps({'type': type_, **fields}) + '\n'


@login_required
def chat_view(request):
    """The chat page. Each page load starts a fresh conversation (no
    persistence in Phase 1)."""
    return render(request, 'ai_lab_chatbot/chat.html')


@login_required
@require_http_methods(["POST"])
def send_message(request):
    """Accept the running history, prepend the system prompt, and stream
    Mycroft's reply back token-by-token.

    Request body: {"messages": [{"role": "user"|"assistant", "content": str}, ...]}
    The browser sends the full history each turn; the server keeps no state.
    """
    try:
        payload = json.loads(request.body)
        history = payload.get('messages', [])
    except (json.JSONDecodeError, AttributeError):
        return HttpResponseBadRequest("Invalid JSON body.")

    # Sanitize the client-supplied history down to role/content pairs.
    messages = [
        {'role': m['role'], 'content': m['content']}
        for m in history
        if isinstance(m, dict) and m.get('role') in _ALLOWED_ROLES and m.get('content')
    ]

    if not messages:
        return HttpResponseBadRequest("No messages provided.")

    system_prompt = build_system_prompt(request.user)
    messages = [{'role': 'system', 'content': system_prompt}] + messages

    def token_stream():
        # Newline-delimited JSON frames. Framing lets the browser tell our
        # output apart from anything a proxy or a dying worker writes onto the
        # connection — an unframed byte stream is indistinguishable from an
        # error page, and gets appended to the chat as if the model said it.
        try:
            for piece in stream_chat(messages):
                yield _frame('token', content=piece)
        except Exception as exc:
            logger.exception("Mycroft stream failed")
            yield _frame('error', content=str(exc))
        else:
            # Absent if the worker is killed mid-stream (gunicorn's abort
            # raises SystemExit, which bypasses the except above). The client
            # treats a missing 'done' as a truncated response.
            yield _frame('done')

    response = StreamingHttpResponse(
        token_stream(), content_type='application/x-ndjson'
    )
    # Discourage proxies from buffering the stream.
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response
