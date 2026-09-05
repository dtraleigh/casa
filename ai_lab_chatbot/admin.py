from django import forms
from django.contrib import admin

from ai_lab_chatbot.models import (
    Personality, HouseholdFact, UserContext, Conversation, Message, Knowledge,
    MycroftConfig,
)
from ai_lab_chatbot.mycroft import memory
from ai_lab_chatbot.mycroft.client import list_models


@admin.register(Personality)
class PersonalityAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_active', 'updated_at')
    list_filter = ('is_active',)
    search_fields = ('name', 'description', 'instructions')


@admin.register(HouseholdFact)
class HouseholdFactAdmin(admin.ModelAdmin):
    # source + source_username surface auto-learned facts (Phase 3b) at a glance:
    # filter source='learned' to review what Mycroft picked up and prune as needed.
    list_display = ('content_preview', 'source', 'source_username', 'updated_at')
    list_filter = ('source',)
    search_fields = ('content', 'source_username')
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        (None, {
            'description': (
                "Facts here are shared across every user's conversations with "
                "Mycroft. Enter one clear, durable fact per row — for a "
                "hand-added fact, fill in the content and leave the rest as-is."
            ),
            'fields': ('content', 'source'),
        }),
        ('Source attribution (set automatically by auto-learning)', {
            'classes': ('collapse',),
            'description': "Leave blank when adding a fact by hand.",
            'fields': ('source_user_id', 'source_username'),
        }),
        ('Timestamps', {
            'classes': ('collapse',),
            'fields': ('created_at', 'updated_at'),
        }),
    )

    @admin.display(description='Content')
    def content_preview(self, obj):
        return obj.content[:80]


@admin.register(UserContext)
class UserContextAdmin(admin.ModelAdmin):
    list_display = ('username', 'user_id', 'updated_at')
    search_fields = ('username', 'content')


class MessageInline(admin.TabularInline):
    """Read-only transcript view — conversations are audit records, not edited
    by hand."""
    model = Message
    extra = 0
    can_delete = False
    fields = ('role', 'content', 'model', 'created_at')
    readonly_fields = ('role', 'content', 'model', 'created_at')

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('display_title', 'username', 'user_id', 'model', 'is_favorite', 'updated_at')
    list_filter = ('is_favorite', 'model')
    search_fields = ('title', 'username', 'messages__content')
    readonly_fields = ('id', 'user_id', 'username', 'created_at', 'updated_at')
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'role', 'content_preview', 'model', 'created_at')
    list_filter = ('role', 'model')
    search_fields = ('content',)

    @admin.display(description='Content')
    def content_preview(self, obj):
        return obj.content[:80]


@admin.register(Knowledge)
class KnowledgeAdmin(admin.ModelAdmin):
    list_display = ('topic', 'content_preview', 'updated_at')
    search_fields = ('topic', 'content')
    # Embedding is managed on save, not hand-edited.
    exclude = ('embedding',)
    readonly_fields = ('created_at', 'updated_at')

    @admin.display(description='Content')
    def content_preview(self, obj):
        return obj.content[:80]

    def save_model(self, request, obj, form, change):
        """Persist, then (re-)embed from the saved topic + content so admin
        edits stay searchable. Embedding is best-effort — a failure leaves the
        row saved with a null vector rather than blocking the edit."""
        super().save_model(request, obj, form, change)
        memory.embed_knowledge(obj)


class MycroftConfigForm(forms.ModelForm):
    """Renders `default_chat_model` as a dropdown of installed Ollama models so
    the default is picked, not typed. The current value and a blank
    "use the setting default" option are always included, and if Ollama is
    unreachable the field degrades to just those so the config still saves."""

    class Meta:
        model = MycroftConfig
        fields = ['default_chat_model']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        current = self.instance.default_chat_model
        # dict.fromkeys de-dups while preserving order (installed first).
        names = list(dict.fromkeys(
            [*list_models(), *( [current] if current else [] )]
        ))
        choices = [('', '— use OLLAMA_CHAT_MODEL default —')]
        choices += [(n, n) for n in names]
        self.fields['default_chat_model'] = forms.ChoiceField(
            choices=choices, required=False,
            help_text=MycroftConfig._meta.get_field('default_chat_model').help_text,
        )


@admin.register(MycroftConfig)
class MycroftConfigAdmin(admin.ModelAdmin):
    form = MycroftConfigForm
    list_display = ('__str__', 'default_chat_model', 'updated_at')
    readonly_fields = ('updated_at',)

    def has_add_permission(self, request):
        # Singleton — allow creating the one row only if it doesn't exist yet.
        return not MycroftConfig.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False
