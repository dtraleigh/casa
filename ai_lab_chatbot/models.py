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
