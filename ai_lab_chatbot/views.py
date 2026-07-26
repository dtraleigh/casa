import json
import logging

from django.contrib.auth.decorators import login_required
from django.http import (
    StreamingHttpResponse, HttpResponseBadRequest, JsonResponse, Http404,
)
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from ai_lab_chatbot.models import Conversation
from ai_lab_chatbot.mycroft import memory
from ai_lab_chatbot.mycroft.client import stream_chat, complete_chat
from ai_lab_chatbot.mycroft.prompts import build_system_prompt, build_title_prompt

logger = logging.getLogger(__name__)


def _frame(type_, **fields):
    """One newline-delimited JSON frame of the response stream."""
    return json.dumps({'type': type_, **fields}) + '\n'


def _serialize_messages(conversation):
    """The full transcript for rendering a resumed conversation in the page."""
    return [
        {'role': m.role, 'content': m.content}
        for m in conversation.messages.all()
    ]


@login_required
def chat_view(request):
    """A fresh chat page. The conversation is created lazily on first send."""
    return render(request, 'ai_lab_chatbot/chat.html', {
        'bootstrap': {'conversation_id': None, 'title': '', 'messages': []},
    })


@login_required
def resume_view(request, conversation_id):
    """Resume an existing conversation — renders its full transcript."""
    try:
        conversation = Conversation.objects.get(
            id=conversation_id, user_id=request.user.id
        )
    except Conversation.DoesNotExist:
        raise Http404("No such conversation.")

    return render(request, 'ai_lab_chatbot/chat.html', {
        'bootstrap': {
            'conversation_id': str(conversation.id),
            'title': conversation.display_title(),
            'messages': _serialize_messages(conversation),
        },
    })


@login_required
def history_view(request):
    """List the user's conversations, favorites pinned in their own section on
    top, each most-recent-activity first."""
    return render(request, 'ai_lab_chatbot/history.html', {
        'favorites': memory.favorite_conversations(request.user),
        'conversations': memory.other_conversations(request.user),
    })


@login_required
@require_http_methods(["POST"])
def toggle_favorite_view(request, conversation_id):
    """Toggle a conversation's favorite flag; returns the new state."""
    try:
        conversation = memory.toggle_favorite(request.user, conversation_id)
    except Conversation.DoesNotExist:
        raise Http404("No such conversation.")
    return JsonResponse({'is_favorite': conversation.is_favorite})


@login_required
@require_http_methods(["POST"])
def delete_conversation_view(request, conversation_id):
    """Delete one of the user's conversations, then return to history."""
    try:
        memory.delete_conversation(request.user, conversation_id)
    except Conversation.DoesNotExist:
        raise Http404("No such conversation.")
    return redirect('ai_lab_chatbot:history')


@login_required
@require_http_methods(["POST"])
def send_message(request):
    """Persist the user's turn, stream Mycroft's reply, persist that too.

    Request body: {"conversation_id": "<uuid>"|null, "content": str}. The server
    owns history now — it reconstructs the sliding window from the DB rather than
    trusting a client-supplied array.
    """
    try:
        payload = json.loads(request.body)
        conversation_id = payload.get('conversation_id')
        content = (payload.get('content') or '').strip()
    except (json.JSONDecodeError, AttributeError):
        return HttpResponseBadRequest("Invalid JSON body.")

    if not content:
        return HttpResponseBadRequest("No message provided.")

    try:
        conversation = memory.get_or_create_conversation(
            request.user, conversation_id
        )
    except Conversation.DoesNotExist:
        raise Http404("No such conversation.")

    is_new = conversation.messages.count() == 0

    memory.add_message(conversation, 'user', content)
    system_prompt = build_system_prompt(request.user)
    messages = (
        [{'role': 'system', 'content': system_prompt}]
        + memory.history_for_prompt(conversation)
    )

    def token_stream():
        # Newline-delimited JSON frames. Framing lets the browser tell our
        # output apart from anything a proxy or a dying worker writes onto the
        # connection — an unframed byte stream is indistinguishable from an
        # error page, and gets appended to the chat as if the model said it.
        assistant_text = ''
        errored = False
        try:
            for piece in stream_chat(messages):
                assistant_text += piece
                yield _frame('token', content=piece)
        except Exception as exc:
            errored = True
            logger.exception("Mycroft stream failed")
            yield _frame(
                'error', content=str(exc),
                conversation_id=str(conversation.id), is_new=is_new,
            )

        # Persist whatever came back (partial included — transcript-visible) and
        # bump activity time. Runs before request_finished, so DB access is safe.
        if assistant_text:
            memory.add_message(conversation, 'assistant', assistant_text)
        memory.touch(conversation)

        if not errored:
            # Absent if the worker is killed mid-stream (gunicorn's abort raises
            # SystemExit, bypassing the except above). The client treats a
            # missing 'done' as a truncated response.
            yield _frame('done', conversation_id=str(conversation.id), is_new=is_new)

    response = StreamingHttpResponse(
        token_stream(), content_type='application/x-ndjson'
    )
    # Discourage proxies from buffering the stream.
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


@login_required
@require_http_methods(["POST"])
def generate_title_view(request, conversation_id):
    """Generate a short title from the first exchange (called once by the
    browser after the first reply). No-op if a title already exists; a
    generation failure leaves it blank so the UI falls back to a snippet."""
    try:
        conversation = Conversation.objects.get(
            id=conversation_id, user_id=request.user.id
        )
    except Conversation.DoesNotExist:
        raise Http404("No such conversation.")

    if conversation.title:
        return JsonResponse({'title': conversation.title})

    first_user = conversation.messages.filter(role='user').first()
    first_assistant = conversation.messages.filter(role='assistant').first()
    if not (first_user and first_assistant):
        return JsonResponse({'title': ''})

    try:
        title = complete_chat(
            build_title_prompt(first_user.content, first_assistant.content)
        )
        title = title.strip().strip('"').strip()[:200]
    except Exception:
        logger.exception("Mycroft title generation failed")
        title = ''

    if title:
        conversation.title = title
        conversation.save(update_fields=['title', 'updated_at'])

    return JsonResponse({'title': title})
