import uuid

from django.db import models


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

    Phase 1 is admin-curated only; Phase 4 adds auto-learning from
    conversations, which is why source attribution fields already exist.
    """
    SOURCE_CHOICES = [
        ('admin', 'Admin-curated'),
        ('learned', 'Learned from conversation'),
    ]

    content = models.TextField(help_text="A single fact about the household.")
    source = models.CharField(
        max_length=20, choices=SOURCE_CHOICES, default='admin'
    )
    # Decoupled user reference: auth.User lives in the `default` DB while this
    # model lives in `ai_lab`, so no cross-DB ForeignKey. Unused in Phase 1;
    # populated by Phase 4 auto-learning.
    source_user_id = models.IntegerField(null=True, blank=True)
    source_username = models.CharField(max_length=150, blank=True)
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
    plain CharField so Phase 3 can add 'tool' without a migration."""
    id = models.BigAutoField(primary_key=True)
    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name='messages'
    )
    role = models.CharField(max_length=20)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # Tie-break on the monotonic id so same-microsecond inserts stay ordered.
        ordering = ['created_at', 'id']

    def __str__(self):
        return f"{self.role}: {self.content[:60]}"
