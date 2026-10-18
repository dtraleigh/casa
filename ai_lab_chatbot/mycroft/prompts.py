"""Assembles the system prompt sent to Ollama for each request.

The final prompt is never stored — it's built fresh every request from the
active Personality, all current HouseholdFacts, and the requesting user's UserContext,
plus descriptions of the tools offered on this request (Phase 4) and hardcoded
guardrails.
"""
from ai_lab_chatbot.models import Personality, HouseholdFact, UserContext
from ai_lab_chatbot.mycroft.tools import describe_registered_tools

# Guardrails always apply, regardless of the active personality. They live in
# code (not the admin-editable Personality) so editing personality can never
# accidentally remove them.
STANDARD_GUARDRAILS = (
    "Never claim to have capabilities you don't have. If a question needs "
    "real-time information you have no tool for (live scores, recent news, "
    "anything not covered by a tool listed above), say so plainly rather than "
    "guessing. You are a conversational "
    "assistant, not a research agent — don't attempt long multi-step research "
    "tasks."
)

# Appended in "Teach Mycroft" learning conversations. Lives in code for the same
# reason as the guardrails. The user's answers are mined into HouseholdFacts
# off-band (memory.apply_learning); this only shapes the interview itself.
LEARNING_INSTRUCTIONS = (
    "This is a learning conversation: your job right now is to get to know the "
    "household so you can help better later. The facts you already know are "
    "listed under \"About the household\" — don't ask about those again unless "
    "checking whether something has changed. Ask ONE short, friendly question "
    "at a time about something you don't know yet. Good topics: who lives here "
    "and how they're related, pets, the home and address, vehicles, work and "
    "school schedules, routines, likes and dislikes, important dates, and "
    "service providers (doctors, vets, utilities). Briefly acknowledge each "
    "answer before moving on. If the user says something has changed, confirm "
    "the change in a sentence. If they want to tell you about something "
    "specific, follow their lead. Never invent or assume facts, and if they "
    "don't want to answer, move on."
)

# Minimal identity used only if no Personality is marked active, so a
# misconfigured admin state degrades instead of producing an empty prompt.
_FALLBACK_IDENTITY = (
    "You are Mycroft, a locally-hosted assistant running on Leo's home server."
)


def build_system_prompt(user, knowledge=None, past=None, mode='chat',
                        tool_names=None) -> str:
    """Assemble the system prompt for `user`. Empty sections are omitted.

    `knowledge` and `past` are optional semantically-retrieved context (Phase 3):
    Knowledge rows and past Message rows respectively. They ride in the system
    prompt like personality/facts — assembled fresh, never stored. `mode` is the
    conversation's mode; 'learning' adds the interviewer instructions.
    `tool_names` are the tools offered on this request (Phase 4), and only
    those are described, so Mycroft never claims a tool he can't call.
    """
    personality = Personality.get_active()
    # Small table, load all current facts (superseded ones are history only).
    facts = HouseholdFact.objects.filter(is_current=True)
    context = UserContext.for_user(user)
    tool_descriptions = (
        describe_registered_tools(tool_names) if tool_names else ''
    )

    sections = []

    if personality is not None:
        sections.append(personality.description.strip())
        if personality.instructions.strip():
            sections.append(personality.instructions.strip())
    else:
        sections.append(_FALLBACK_IDENTITY)

    fact_lines = [f"- {f.content.strip()}" for f in facts if f.content.strip()]
    if fact_lines:
        sections.append("About the household:\n" + "\n".join(fact_lines))

    if context.content.strip():
        sections.append("About the current user:\n" + context.content.strip())

    knowledge_lines = [
        f"- {k.content.strip()}" for k in (knowledge or []) if k.content.strip()
    ]
    if knowledge_lines:
        sections.append("Relevant knowledge:\n" + "\n".join(knowledge_lines))

    past_lines = [
        f"- {m.role}: {m.content.strip()}" for m in (past or []) if m.content.strip()
    ]
    if past_lines:
        sections.append(
            "Relevant past conversation:\n" + "\n".join(past_lines)
        )

    if tool_descriptions.strip():
        sections.append(tool_descriptions.strip())

    if mode == 'learning':
        sections.append(LEARNING_INSTRUCTIONS)

    sections.append(STANDARD_GUARDRAILS)

    return "\n\n".join(sections)


def build_title_prompt(first_user, first_assistant):
    """Messages asking Mycroft to name a conversation from its first exchange.

    Returns a small message list for a non-streaming completion. Kept terse and
    self-contained (no personality) so the title stays a plain label.
    """
    exchange = f"User: {first_user}\n\nAssistant: {first_assistant}"
    return [
        {
            'role': 'system',
            'content': (
                "You write short conversation titles. Given the first exchange "
                "of a chat, reply with a title of at most 5 words that captures "
                "the topic. Reply with the title only — no quotes, no punctuation "
                "at the end, no preamble."
            ),
        },
        {'role': 'user', 'content': exchange},
    ]


def build_fact_update_prompt(question, answer, current_facts, username):
    """Messages asking Mycroft to turn one learning-chat answer into fact edits.

    `question` is Mycroft's preceding message (context only, so a bare answer
    like "Biscuit" can be resolved); `answer` is the user's reply and the ONLY
    source of facts. `current_facts` are the current HouseholdFacts, shown with
    ids so the model can update or retire them. The reply is expected to be a
    JSON object {"add": [...], "update": [{"id", "content"}], "retire": [...]};
    `memory.py` parses it defensively.
    """
    speaker = username or "the user"
    fact_lines = "\n".join(
        f"[{f.id}] {f.content.strip()}" for f in current_facts
    ) or "(none yet)"
    body = (
        f"Known facts:\n{fact_lines}\n\n"
        f"Mycroft asked: {question or '(nothing - the user volunteered this)'}\n\n"
        f"{speaker} answered: {answer}"
    )
    return [
        {
            'role': 'system',
            'content': (
                "You maintain a list of durable facts about a household. Given "
                "the known facts and the user's latest answer, decide what to "
                "change. Only the user's answer is a source of facts — Mycroft's "
                "question is context, never a fact. Ignore general knowledge, "
                "remarks about the conversation itself, and anything transient "
                "(moods, what someone is doing right now).\n\n"
                "- add: new facts the answer states that are not already known.\n"
                "- update: a known fact the answer says has changed; give its id "
                "and the full new sentence.\n"
                "- retire: ids of known facts the answer says are no longer true "
                "and have no replacement.\n\n"
                "Write each fact as one short standalone sentence naming the "
                f"person (the speaker is {speaker}), e.g. \"{speaker}'s dog is "
                "named Biscuit.\"\n\n"
                "Examples:\n"
                'Answer "We moved to 12 Oak St" with known fact [4] "The '
                'household lives at 9 Elm Rd." -> {"add": [], "update": '
                '[{"id": 4, "content": "The household lives at 12 Oak St."}], '
                '"retire": []}\n'
                'Answer "Not sure, ask me later" -> {"add": [], "update": [], '
                '"retire": []}\n\n'
                "Reply with ONLY the JSON object — no prose, no code fences."
            ),
        },
        {'role': 'user', 'content': body},
    ]


def build_tool_router_prompt(recent, latest, tool_hints):
    """Messages asking whether the latest user message needs any tools.

    Routing happens before the real reply because llama3.1's Ollama template
    pushes the model toward a function call on any turn where tools are
    offered, including "what's 2+2?". Tools are offered only when this says
    so. `recent` is a few earlier text turns as (role, content) pairs, so a
    follow-up like "and tomorrow?" resolves. `tool_hints` are
    "name: when to use it" lines. The reply should be a JSON array of tool
    names, and `tools.select_tools` parses it defensively.
    """
    history = "\n".join(f"{role}: {content}" for role, content in recent)
    body = (
        (f"Earlier in the conversation:\n{history}\n\n" if history else "")
        + f"Latest message: {latest}"
    )
    return [
        {
            'role': 'system',
            'content': (
                "You decide whether answering a user's latest message needs a "
                "live-data tool. Available tools:\n"
                + "\n".join(f"- {h}" for h in tool_hints)
                + "\n\nMost messages need no tool. General knowledge, history, "
                "maths, advice, opinions and chit-chat all need none. Pick a "
                "tool only when the answer depends on the live data it "
                "provides.\n\n"
                "Examples:\n"
                '"What\'s the weather tomorrow?" -> ["get_current_weather"]\n'
                '"Should I bring a jacket tonight?" -> ["get_current_weather"]\n'
                '"What is 2+2?" -> []\n'
                '"When was Raleigh founded?" -> []\n\n'
                "Reply with ONLY a JSON array of tool names, or [] for none."
            ),
        },
        {'role': 'user', 'content': body},
    ]
