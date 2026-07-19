from django.contrib.auth import get_user_model
from django.db import migrations

PERSONALITY_NAME = "Mycroft v1"

PERSONALITY_DESCRIPTION = (
    "You are Mycroft, a locally-hosted assistant running on Leo's home server. "
    "You are intelligent, dry, and economical with words. You skip "
    "corporate-assistant phrases like 'Great question!' or 'I'd be happy to "
    "help!' and just help. You have a subtle sense of humor but don't force it. "
    "You are conversational, not formal."
)

PERSONALITY_INSTRUCTIONS = (
    "Answer general questions from your own knowledge — facts, explanations, "
    "discussion, opinions when asked. Keep responses concise unless asked to "
    "elaborate. Prefer prose over bullet lists in conversation. If a question "
    "requires real-time information you don't have (current weather, live "
    "sports scores, recent news), acknowledge that plainly rather than "
    "guessing. You are a conversational assistant, not a research agent — "
    "don't attempt long multi-step research tasks."
)

RALEIGH_FACT = "The household is based in Raleigh, NC."

LEO_CONTEXT = (
    "Leo runs an urban planning blog focused on Raleigh, NC, serves on the "
    "local transit board (RTA), and is an urbanism advocate. He works with "
    "Django, runs a home server, and is technically fluent. Reference this "
    "context only when directly relevant; don't force it."
)


def seed(apps, schema_editor):
    Personality = apps.get_model('ai_lab_chatbot', 'Personality')
    HouseholdFact = apps.get_model('ai_lab_chatbot', 'HouseholdFact')
    UserContext = apps.get_model('ai_lab_chatbot', 'UserContext')

    Personality.objects.get_or_create(
        name=PERSONALITY_NAME,
        defaults={
            'description': PERSONALITY_DESCRIPTION,
            'instructions': PERSONALITY_INSTRUCTIONS,
            'is_active': True,
        },
    )

    HouseholdFact.objects.get_or_create(
        content=RALEIGH_FACT,
        defaults={'source': 'admin'},
    )

    # auth.User lives in the `default` DB; the router sends this read there even
    # though the migration targets `ai_lab`. Seed leo's context only if he
    # exists — a fresh install without the user simply skips this.
    User = get_user_model()
    leo = User.objects.filter(username='leo').first()
    if leo is not None:
        UserContext.objects.get_or_create(
            user_id=leo.id,
            defaults={'username': 'leo', 'content': LEO_CONTEXT},
        )


def unseed(apps, schema_editor):
    Personality = apps.get_model('ai_lab_chatbot', 'Personality')
    HouseholdFact = apps.get_model('ai_lab_chatbot', 'HouseholdFact')
    Personality.objects.filter(name=PERSONALITY_NAME).delete()
    HouseholdFact.objects.filter(content=RALEIGH_FACT).delete()
    # Leave UserContext rows alone; they may hold admin-added content.


class Migration(migrations.Migration):

    dependencies = [
        ('ai_lab_chatbot', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(seed, unseed),
    ]
