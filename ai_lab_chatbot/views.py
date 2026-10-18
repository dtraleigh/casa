import json
import logging

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import (
    StreamingHttpResponse, HttpResponseBadRequest, JsonResponse, Http404,
)
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from ai_lab_chatbot.models import Conversation, MycroftConfig
from ai_lab_chatbot.mycroft import memory, tools
from ai_lab_chatbot.mycroft.client import (
    stream_chat, complete_chat, list_models, supports_tools,
)
from ai_lab_chatbot.mycroft.prompts import build_system_prompt, build_title_prompt

logger = logging.getLogger(__name__)


def _frame(type_, **fields):
    """One newline-delimited JSON frame of the response stream."""
    return json.dumps({'type': type_, **fields}) + '\n'


def _serialize_messages(conversation):
    """The full transcript for rendering a resumed conversation in the page."""
    return [
        {'role': m.role, 'content': m.content, 'model': m.model,
         'tool_calls': m.tool_calls, 'tool_name': m.tool_name}
        for m in conversation.messages.all()
    ]


def _last_context(conversation):
    """Token counts of the most recent assistant turn, for the context footer on
    resume. None if there is none, or the last one predates the feature (its
    counts are null) — the client renders `Context: —` in that case."""
    last = conversation.messages.filter(role='assistant').last()
    if not last or last.prompt_tokens is None:
        return None
    return {
        'prompt_tokens': last.prompt_tokens,
        'completion_tokens': last.completion_tokens,
    }


def _model_options(selected):
    """Installed chat models for the selector, guaranteed to include `selected`
    (so a conversation on a since-removed model still shows it), sorted."""
    options = set(list_models())
    if selected:
        options.add(selected)
    return sorted(options)


@login_required
def chat_view(request):
    """A fresh chat page. The conversation is created lazily on first send."""
    default_model = MycroftConfig.default_model()
    return render(request, 'ai_lab_chatbot/chat.html', {
        'bootstrap': {
            'conversation_id': None, 'title': '', 'messages': [],
            'context': None,
            'models': _model_options(default_model),
            'selected_model': default_model,
            'mode': 'chat',
        },
        'mycroft_num_ctx': settings.MYCROFT_NUM_CTX,
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

    selected_model = conversation.model or MycroftConfig.default_model()
    return render(request, 'ai_lab_chatbot/chat.html', {
        'bootstrap': {
            'conversation_id': str(conversation.id),
            'title': conversation.display_title(),
            'messages': _serialize_messages(conversation),
            'context': _last_context(conversation),
            'models': _model_options(selected_model),
            'selected_model': selected_model,
            'mode': conversation.mode,
        },
        'mycroft_num_ctx': settings.MYCROFT_NUM_CTX,
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
def rename_conversation_view(request, conversation_id):
    """Rename a conversation to a user-supplied title. An empty/blank title
    clears back to the auto-generated fallback; the response carries both the
    stored title (for the edit field) and the resolved display_title (for the
    label the user sees)."""
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON body.")

    try:
        conversation = memory.rename_conversation(
            request.user, conversation_id, payload.get('title', '')
        )
    except Conversation.DoesNotExist:
        raise Http404("No such conversation.")

    return JsonResponse({
        'title': conversation.title,
        'display_title': conversation.display_title(),
    })


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
        requested_model = (payload.get('model') or '').strip()
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

    # Resolve and remember the model for this turn: an explicit dropdown choice
    # wins, else the conversation's stored model, else the global default. The
    # dropdown only offers installed models, so a bad pick just surfaces via the
    # stream error frame — no pre-send validation round-trip to Ollama.
    chosen_model = (
        requested_model or conversation.model or MycroftConfig.default_model()
    )
    if conversation.model != chosen_model:
        conversation.model = chosen_model
        conversation.save(update_fields=['model'])

    user_msg = memory.add_message(conversation, 'user', content)
    # Embed the incoming turn (best-effort) and reuse that vector to pull in
    # semantically relevant knowledge + past messages from other conversations.
    query_vec = memory.embed_message(user_msg)
    knowledge, past = memory.retrieve_memories(
        request.user, query_vec, exclude_conversation_id=conversation.id
    )
    history = memory.history_for_prompt(conversation)
    # Tools (Phase 4) are offered only in ordinary chats (a learning interview
    # has no use for them), only to models that can take them, and only when
    # the router says this message needs them. Offering tools on every turn
    # makes llama3.1 try to call one even for "what's 2+2?".
    tool_names = []
    if conversation.mode == 'chat' and supports_tools(chosen_model):
        recent = [
            (m['role'], m['content']) for m in history[:-1]
            if m['role'] in ('user', 'assistant') and m['content']
        ]
        tool_names = tools.select_tools(recent, content, model=chosen_model)
    system_prompt = build_system_prompt(
        request.user, knowledge=knowledge, past=past, mode=conversation.mode,
        tool_names=tool_names,
    )
    messages = [{'role': 'system', 'content': system_prompt}] + history
    tool_defs = tools.tool_definitions(tool_names) if tool_names else None
    max_rounds = settings.MYCROFT_TOOL_MAX_ROUNDS

    def token_stream():
        # Newline-delimited JSON frames. Framing lets the browser tell our
        # output apart from anything a proxy or a dying worker writes onto the
        # connection — an unframed byte stream is indistinguishable from an
        # error page, and gets appended to the chat as if the model said it.
        assistant_text = ''
        errored = False
        stats = {}  # populated from Ollama's final chunk (token counts, timings)
        try:
            # Tool loop: while the model asks for tools, run them, persist the
            # call + result turns, and ask again with the results appended.
            # The last round withholds tools so the model has to answer.
            available = tool_defs
            for round_no in range(max_rounds + 1):
                offered = available if round_no < max_rounds else None
                calls = []
                stats.clear()
                assistant_text = ''
                # While tools are offered, hold back a reply that starts like
                # JSON: it may be a call written as text rather than an answer.
                holding = bool(offered)
                for piece in stream_chat(messages, stats_out=stats,
                                         model=chosen_model, tools=offered,
                                         tool_calls_out=calls):
                    assistant_text += piece
                    if holding:
                        head = assistant_text.lstrip()
                        if not head or head[0] in '{`':
                            continue
                        holding = False
                        piece = assistant_text  # release what was held
                    yield _frame('token', content=piece)
                if holding and assistant_text and not calls:
                    text_call = tools.parse_text_tool_call(assistant_text)
                    if text_call is None:
                        # Just an answer that began with a brace; send it on.
                        yield _frame('token', content=assistant_text)
                    elif text_call['name'] in tool_names:
                        calls = [text_call]
                        assistant_text = ''
                    else:
                        # Junk like {"name": "None"}. Drop it and answer again
                        # without tools.
                        logger.info("Discarded text tool call %r", assistant_text)
                        assistant_text = ''
                        available = None
                        continue
                if not calls:
                    break
                calls = [
                    {'name': c['name'],
                     'arguments': tools.coerce_arguments(c['arguments'])}
                    for c in calls
                ]
                call_msg = memory.add_message(
                    conversation, 'assistant', assistant_text,
                    prompt_tokens=stats.get('prompt_tokens'),
                    completion_tokens=stats.get('completion_tokens'),
                    model=chosen_model, tool_calls=calls,
                )
                messages.append(memory.prompt_message(call_msg))
                # Any text that came with the call belongs to that turn.
                assistant_text = ''
                for call in calls:
                    yield _frame('tool_call', name=call['name'],
                                 arguments=call['arguments'])
                    result = tools.run_tool(call['name'], call['arguments'])
                    result_msg = memory.add_message(
                        conversation, 'tool', result, tool_name=call['name'])
                    messages.append(memory.prompt_message(result_msg))
                    yield _frame('tool_result', name=call['name'],
                                 content=result)
        except Exception as exc:
            errored = True
            logger.exception("Mycroft stream failed")
            yield _frame(
                'error', content=str(exc),
                conversation_id=str(conversation.id), is_new=is_new,
            )

        # Persist whatever came back (partial included — transcript-visible) and
        # bump activity time. Runs before request_finished, so DB access is safe.
        # Token counts land only on a clean completion; a partial/errored reply
        # leaves them null (no final chunk means an empty `stats`).
        if assistant_text:
            assistant_msg = memory.add_message(
                conversation, 'assistant', assistant_text,
                prompt_tokens=stats.get('prompt_tokens'),
                completion_tokens=stats.get('completion_tokens'),
                model=chosen_model,
            )
            # Embed the reply too (best-effort) so Mycroft's own answers are
            # recallable later, not just the user's turns.
            memory.embed_message(assistant_msg)
        memory.touch(conversation)

        if not errored:
            # The per-turn context metrics, ahead of the terminal frame. Only on
            # a clean stream (an error path has no reliable counts/timings).
            if stats:
                yield _frame('stats', **stats)
            # 'done' is absent if the worker is killed mid-stream (gunicorn's
            # abort raises SystemExit, bypassing the except above). The client
            # treats a missing 'done' as a truncated response.
            yield _frame('done', conversation_id=str(conversation.id),
                         is_new=is_new, model=chosen_model)

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
    # Skip tool-call turns (Phase 4); they usually have no text to title from.
    first_assistant = (
        conversation.messages.filter(role='assistant').exclude(content='').first()
    )
    if not (first_user and first_assistant):
        return JsonResponse({'title': ''})

    try:
        title = complete_chat(
            build_title_prompt(first_user.content, first_assistant.content),
            model=conversation.model or None,
        )
        title = title.strip().strip('"').strip()[:200]
    except Exception:
        logger.exception("Mycroft title generation failed")
        title = ''

    if title:
        conversation.title = title
        conversation.save(update_fields=['title', 'updated_at'])

    return JsonResponse({'title': title})


@login_required
@require_http_methods(["POST"])
def start_learning_view(request):
    """Start a "Teach Mycroft" learning conversation (Phase 3c) — seeded with
    Mycroft's opening question — and open it."""
    conversation = memory.start_learning_conversation(request.user)
    return redirect('ai_lab_chatbot:conversation', conversation_id=conversation.id)


@login_required
@require_http_methods(["POST"])
def learn_from_exchange_view(request, conversation_id):
    """Apply the latest answer in a learning conversation to HouseholdFacts
    (Phase 3c). Called by the browser after each reply in a learning chat, off
    the streaming path; returns the changes so the chat can show them. Ordinary
    chats get no changes. Best-effort — a failure returns empty lists rather
    than an error, so the chat is never affected."""
    try:
        conversation = Conversation.objects.get(
            id=conversation_id, user_id=request.user.id
        )
    except Conversation.DoesNotExist:
        raise Http404("No such conversation.")

    return JsonResponse(memory.apply_learning(conversation, request.user))
