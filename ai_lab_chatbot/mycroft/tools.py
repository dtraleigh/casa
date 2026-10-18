"""Tool definitions and dispatch for Mycroft (Phase 4).

A tool is a schema the model sees plus a Python function Django runs when the
model asks for it. To add a capability: write the function, `register()` a Tool
for it below, and you're done. The schema goes to Ollama, the system prompt
picks up the tool's hint, and `run_tool` dispatches calls to it.

Tools are registered statically at import time; a dynamic registry can come
later if it's ever needed.
"""
import json
import logging
from dataclasses import dataclass
from typing import Callable

from ai_lab_chatbot.mycroft import weather
from ai_lab_chatbot.mycroft.client import complete_chat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tool:
    name: str
    # What the tool does, sent to the model in the function schema.
    description: str
    # JSON Schema for the arguments (an "object" schema).
    parameters: dict
    # Called with the model's arguments as keyword args; returns something
    # JSON-serialisable. Raising `user_errors` yields a message the model can
    # relay; any other exception is logged and reported generically.
    func: Callable
    # How to use it, for the system prompt's tools section.
    prompt_hint: str
    # When it's needed, for the router that decides per message whether to
    # offer tools at all (`select_tools`).
    router_hint: str
    user_errors: tuple = ()


_REGISTRY: dict[str, Tool] = {}


def register(tool):
    _REGISTRY[tool.name] = tool
    return tool


register(Tool(
    name='get_current_weather',
    description=(
        "Get current weather conditions, an hourly outlook for the next 12 "
        "hours and a 7-day forecast for a location."
    ),
    parameters={
        'type': 'object',
        'properties': {
            'location': {
                'type': 'string',
                'description': (
                    'City to check, e.g. "Asheville, NC" or "Paris, France". '
                    "Leave empty for the household's home (Raleigh, NC)."
                ),
            },
        },
        'required': [],
    },
    func=weather.get_current_weather,
    prompt_hint=(
        "get_current_weather: use it whenever the user asks about the weather "
        "or about something that depends on it (what to wear, umbrellas, "
        "outdoor plans). Never guess the weather. If no place is mentioned, "
        "leave the location empty; the household is in Raleigh, NC. Weather "
        "results from earlier in the conversation may be out of date, so call "
        "it again for a fresh check."
    ),
    router_hint=(
        "get_current_weather: current weather and the forecast for the coming "
        "week, for any place. Needed for questions about the weather or that "
        "depend on it (what to wear, umbrellas, outdoor plans, travel "
        "conditions)."
    ),
    user_errors=(weather.WeatherError,),
))


def registered_tools():
    return list(_REGISTRY.values())


def _selected(names):
    """Registered tools, limited to `names` when given."""
    if names is None:
        return list(_REGISTRY.values())
    return [_REGISTRY[n] for n in names if n in _REGISTRY]


def tool_definitions(names=None):
    """The function schemas in Ollama's `tools=` format, for all tools or just
    `names`."""
    return [
        {
            'type': 'function',
            'function': {
                'name': t.name,
                'description': t.description,
                'parameters': t.parameters,
            },
        }
        for t in _selected(names)
    ]


def describe_registered_tools(names=None) -> str:
    """The system prompt's tools section, generated from the registry so what
    Mycroft claims he can do always matches the code. Covers all tools or just
    `names`; empty when there are none."""
    selected = _selected(names)
    if not selected:
        return ""
    lines = [f"- {t.prompt_hint}" for t in selected]
    return (
        "You have these tools for live information:\n" + "\n".join(lines)
        + "\nOnly call a tool when the answer needs it. Answer everything "
        "else (general knowledge, maths, conversation) directly. After a tool "
        "returns, answer in plain conversational language and don't repeat "
        "the raw data."
    )


def coerce_arguments(arguments):
    """Arguments as a dict. Models sometimes send a JSON string, or nothing."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return {}
    return dict(arguments) if isinstance(arguments, dict) else {}


def run_tool(name, arguments) -> str:
    """Execute one tool call and return its result as a JSON string for a
    `role='tool'` message. Never raises: an unknown tool or a failure becomes an
    `{"error": ...}` result, so the model can tell the user what went wrong
    instead of the chat breaking."""
    tool = _REGISTRY.get(name)
    if tool is None:
        return json.dumps({'error': f"There is no tool called {name!r}."})

    # Drop arguments the schema doesn't declare (a small model's improvisation
    # shouldn't become a TypeError). Values are stringified for string params.
    props = tool.parameters.get('properties', {})
    kwargs = {}
    for key, value in coerce_arguments(arguments).items():
        if key not in props:
            continue
        if props[key].get('type') == 'string' and value is not None:
            value = str(value)
        kwargs[key] = value

    try:
        result = tool.func(**kwargs)
    except tool.user_errors as exc:
        return json.dumps({'error': str(exc)})
    except Exception:
        logger.exception("Mycroft tool %s failed", name)
        return json.dumps({'error': f"The {name} tool failed unexpectedly."})
    return json.dumps(result, ensure_ascii=False)


# How many earlier text turns the router sees, so a follow-up ("and
# tomorrow?") is routed like the question it follows.
_ROUTER_CONTEXT_TURNS = 4


def select_tools(recent, latest, model=None):
    """The tools the latest user message needs, as a list of names, and
    usually [].

    One short, deterministic completion before the real reply. Tools are
    offered only when this picks some, because llama3.1's template pushes the
    model toward a function call on every turn that offers tools. `recent` is
    the conversation so far as (role, content) pairs; only the tail is used.
    Best-effort: any failure means no tools.
    """
    if not _REGISTRY:
        return []
    # Imported here because prompts imports this module.
    from ai_lab_chatbot.mycroft.prompts import build_tool_router_prompt

    try:
        raw = complete_chat(
            build_tool_router_prompt(
                recent[-_ROUTER_CONTEXT_TURNS:], latest,
                [t.router_hint for t in _REGISTRY.values()],
            ),
            model=model, options={'temperature': 0},
        )
    except Exception:
        logger.exception("Mycroft tool routing failed")
        return []
    return _parse_tool_names(raw)


def _parse_tool_names(raw):
    """Registered tool names from the router's reply. Expects a JSON array,
    and falls back to spotting bare tool names in prose."""
    start, end = raw.find('['), raw.rfind(']')
    names = None
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, list):
                names = [n for n in parsed if isinstance(n, str)]
        except ValueError:
            pass
    if names is None:
        names = [n for n in _REGISTRY if n in raw]
    return [n for n in dict.fromkeys(names) if n in _REGISTRY]


def parse_text_tool_call(text):
    """A tool call written out as plain text, or None.

    llama3.1 sometimes writes a call as text instead of making a real one, e.g.
    `{"name": "get_current_weather", "parameters": {...}}`, or
    `{"name": "None", "parameters": {}}` when nothing fits. Returns
    {"name", "arguments"} for anything shaped like that, whatever the name, so
    the caller can run a real tool or discard junk. Code fences are tolerated.
    """
    s = text.strip()
    if s.startswith('```'):
        s = s.strip('`').strip()
        if s.lower().startswith('json'):
            s = s[4:].strip()
    if not (s.startswith('{') and s.endswith('}')):
        return None
    try:
        obj = json.loads(s)
    except ValueError:
        return None
    if (not isinstance(obj, dict) or not isinstance(obj.get('name'), str)
            or set(obj) - {'name', 'parameters', 'arguments'}):
        return None
    args = obj.get('parameters', obj.get('arguments'))
    return {'name': obj['name'], 'arguments': coerce_arguments(args)}
