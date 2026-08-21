"""Assembles the system prompt sent to Ollama for each request.

The final prompt is never stored — it's built fresh every request from the
active Personality, all HouseholdFacts, and the requesting user's UserContext,
plus tool descriptions (empty in Phase 1) and hardcoded guardrails.
"""
from ai_lab_chatbot.models import Personality, HouseholdFact, UserContext
from ai_lab_chatbot.mycroft.tools import describe_registered_tools

# Guardrails always apply, regardless of the active personality. They live in
# code (not the admin-editable Personality) so editing personality can never
# accidentally remove them.
STANDARD_GUARDRAILS = (
    "Never claim to have capabilities you don't have. If a question needs "
    "real-time information you don't have (current weather, live scores, recent "
    "news), say so plainly rather than guessing. You are a conversational "
    "assistant, not a research agent — don't attempt long multi-step research "
    "tasks."
)

# Minimal identity used only if no Personality is marked active, so a
# misconfigured admin state degrades instead of producing an empty prompt.
_FALLBACK_IDENTITY = (
    "You are Mycroft, a locally-hosted assistant running on Leo's home server."
)


def build_system_prompt(user, knowledge=None, past=None) -> str:
    """Assemble the system prompt for `user`. Empty sections are omitted.

    `knowledge` and `past` are optional semantically-retrieved context (Phase 3):
    Knowledge rows and past Message rows respectively. They ride in the system
    prompt like personality/facts — assembled fresh, never stored.
    """
    personality = Personality.get_active()
    facts = HouseholdFact.objects.all()  # small table, load all
    context = UserContext.for_user(user)
    tool_descriptions = describe_registered_tools()

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


def build_extraction_prompt(user_text, assistant_text, username):
    """Messages asking Mycroft to mine durable household facts from one exchange.

    Returns a small message list for a non-streaming completion (Phase 3b
    auto-learning). Kept terse and self-contained — no personality — so the model
    stays focused on extraction, not conversation. The reply is expected to be a
    JSON array of fact strings (empty when nothing qualifies); `memory.py` parses
    it defensively.
    """
    exchange = f"User: {user_text}\n\nAssistant: {assistant_text}"
    speaker = username or "the user"
    return [
        {
            'role': 'system',
            'content': (
                "You extract durable facts about a household and its members from "
                "a single chat exchange, to be remembered long-term. Include only "
                "clear, lasting facts worth keeping: names, relationships, pets, "
                "vehicles, addresses, jobs, and ongoing preferences or routines. "
                "Exclude anything transient (moods, what someone is doing right "
                "now, today's weather), questions, speculation, and anything that "
                "sounds like it was shared in confidence. Be conservative — when "
                "in doubt, leave it out. Write each fact as one short standalone "
                f"sentence, naming the person where known (the speaker is "
                f"{speaker}), e.g. \"Leo's dog is named Biscuit.\" Reply with ONLY "
                "a JSON array of fact strings, and [] when nothing qualifies. No "
                "prose, no code fences, no commentary."
            ),
        },
        {'role': 'user', 'content': exchange},
    ]
