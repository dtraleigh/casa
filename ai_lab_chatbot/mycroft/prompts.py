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


def build_system_prompt(user) -> str:
    """Assemble the system prompt for `user`. Empty sections are omitted."""
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

    if tool_descriptions.strip():
        sections.append(tool_descriptions.strip())

    sections.append(STANDARD_GUARDRAILS)

    return "\n\n".join(sections)
