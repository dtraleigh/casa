import uuid

from django.conf import settings
from django.db import models
from pgvector.django import VectorField


class Personality(models.Model):
    """Mycroft's voice and rules. Exactly one row is active at a time; the
    active personality is shared across all users."""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(
        help_text="Personality and voice — how Mycroft speaks."
    )
    instructions = models.TextField(
        help_text="Rules and considerations for every response."
    )
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "personalities"

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if self.is_active:
            # Only one personality may be active at a time.
            Personality.objects.exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)

    @classmethod
    def get_active(cls):
        return cls.objects.filter(is_active=True).first()


class HouseholdFact(models.Model):
    """Facts about the household, shared across all users' conversations.

    Phase 1 is admin-curated only; Phase 3b adds auto-learning from
    conversations, which is why source attribution fields already exist.
    """
    SOURCE_CHOICES = [
        ('admin', 'Admin-curated'),
        ('learned', 'Learned from conversation'),
    ]

    content = models.TextField(
        help_text=(
            "One specific, durable fact about the household, written as a "
            "standalone sentence. Add a separate row for each fact rather than "
            "listing several here. "
            'Examples: "The household is based in Raleigh, NC." — '
            '"Leo\'s wife is named Jennifer." — "Trash pickup is Tuesday mornings."'
        )
    )
    source = models.CharField(
        max_length=20, choices=SOURCE_CHOICES, default='admin',
        help_text=(
            'Leave as "Admin-curated" for facts you add by hand. '
            '"Learned from conversation" is set automatically by auto-learning '
            "(Phase 3b) — you won't normally choose it yourself."
        ),
    )
    # Decoupled user reference: auth.User lives in the `default` DB while this
    # model lives in `ai_lab`, so no cross-DB ForeignKey. Unused in Phase 1;
    # populated by Phase 3b auto-learning.
    source_user_id = models.IntegerField(
        null=True, blank=True,
        help_text=(
            "Auto-learning only: the id of the user who was talking when this "
            "fact was extracted. Leave blank for hand-entered facts."
        ),
    )
    source_username = models.CharField(
        max_length=150, blank=True,
        help_text=(
            "Auto-learning only: the username who was talking when this fact "
            "was extracted. Leave blank for hand-entered facts."
        ),
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.content[:80]


class UserContext(models.Model):
    """Per-user context: what Mycroft knows about the person he's talking to.

    One row per Django user, referenced by id (decoupled — see HouseholdFact).
    """
    user_id = models.IntegerField(unique=True)
    username = models.CharField(max_length=150, blank=True)
    content = models.TextField(
        blank=True,
        help_text="What Mycroft should know about this user specifically."
    )
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Mycroft's context for {self.username or self.user_id}"

    @classmethod
    def for_user(cls, user):
        """Get-or-create-empty; never returns None."""
        obj, _ = cls.objects.get_or_create(
            user_id=user.id,
            defaults={'username': user.username},
        )
        return obj


class Conversation(models.Model):
    """A persisted chat between one user and Mycroft.

    Private per-user. Owner is referenced by a decoupled integer id (auth.User
    lives in the `default` DB, this model in `ai_lab`, so no cross-DB FK — same
    pattern as HouseholdFact / UserContext). The UUID PK keeps conversation URLs
    unguessable, but access is still enforced by filtering on user_id.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.IntegerField(db_index=True)
    username = models.CharField(max_length=150, blank=True)
    title = models.CharField(max_length=200, blank=True)
    # Ollama chat model this conversation is using (the model selector). Blank
    # resolves to MycroftConfig.default_model() at send time — so pre-feature
    # conversations and fresh ones both fall back to the current default.
    model = models.CharField(max_length=100, blank=True)
    # Pinned by the user to a Favorites section on the History page. Toggling it
    # deliberately does NOT bump updated_at (favoriting isn't activity).
    is_favorite = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # Bumped explicitly on each exchange (a related Message insert does not
    # trigger auto_now here), so history sorts by most-recent-activity.
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return self.display_title()

    def display_title(self):
        """The title, or a first-message snippet fallback until the async title
        lands (or if title generation failed)."""
        if self.title:
            return self.title
        first = self.messages.filter(role='user').first()
        return first.content[:60] if first else 'New conversation'


class Message(models.Model):
    """One turn in a Conversation. Only 'user' and 'assistant' roles are stored
    in Phase 2; the assembled system prompt is never persisted. `role` stays a
    plain CharField so Phase 4 can add 'tool' without a migration."""
    id = models.BigAutoField(primary_key=True)
    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name='messages'
    )
    role = models.CharField(max_length=20)
    content = models.TextField()
    # Per-request context metrics from Ollama, set only on 'assistant' turns
    # produced by a real completion. Null for user turns, partial/errored
    # replies (no final chunk arrived), and every pre-feature row.
    prompt_tokens = models.IntegerField(
        null=True, blank=True,
        help_text="Ollama prompt_eval_count for the request that produced this message.",
    )
    completion_tokens = models.IntegerField(
        null=True, blank=True,
        help_text="Ollama eval_count for this message.",
    )
    # Chat model that produced this turn — set on assistant turns for the model
    # selector, so replies stay attributable when the model is switched mid-chat.
    # Blank for user turns and every pre-feature row.
    model = models.CharField(max_length=100, blank=True)
    # nomic-embed-text vector for semantic recall (Phase 3), set best-effort just
    # after the row is written. Null for pre-feature rows and any turn whose embed
    # call failed — such rows simply aren't retrievable.
    embedding = VectorField(dimensions=768, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # Tie-break on the monotonic id so same-microsecond inserts stay ordered.
        ordering = ['created_at', 'id']

    def __str__(self):
        return f"{self.role}: {self.content[:60]}"


class Knowledge(models.Model):
    """Curated facts Mycroft can recall, independent of any conversation. Global
    (shared across users), admin-authored, and embedded on save so semantic
    retrieval can surface the relevant ones per turn (Phase 3)."""
    topic = models.CharField(max_length=200)
    content = models.TextField(help_text="A fact or note Mycroft should be able to recall.")
    # Embedded from topic + content on save. Null only if the embed call failed
    # (the row still exists; it just won't be retrieved until re-saved).
    embedding = VectorField(dimensions=768, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "knowledge"

    def __str__(self):
        return self.topic


class MycroftConfig(models.Model):
    """Singleton runtime config for Mycroft, editable in admin. Always row pk=1.

    Currently holds just the default chat model new conversations start on;
    kept as a model (not a setting) so it's changeable without a deploy.
    """
    default_chat_model = models.CharField(
        max_length=100, blank=True,
        help_text=(
            "Model new chats start on. Pick from the installed models; leave "
            "blank to fall back to the OLLAMA_CHAT_MODEL setting."
        ),
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Mycroft configuration"
        verbose_name_plural = "Mycroft configuration"

    def __str__(self):
        return "Mycroft configuration"

    def save(self, *args, **kwargs):
        # Pin to a single row so there's exactly one config.
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    @classmethod
    def default_model(cls):
        """The configured default chat model, or the OLLAMA_CHAT_MODEL setting
        when unset — never returns empty."""
        return cls.get_solo().default_chat_model or settings.OLLAMA_CHAT_MODEL
