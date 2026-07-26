from django.contrib import admin

from ai_lab_chatbot.models import (
    Personality, HouseholdFact, UserContext, Conversation, Message,
)


@admin.register(Personality)
class PersonalityAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_active', 'updated_at')
    list_filter = ('is_active',)
    search_fields = ('name', 'description', 'instructions')


@admin.register(HouseholdFact)
class HouseholdFactAdmin(admin.ModelAdmin):
    list_display = ('content_preview', 'source', 'updated_at')
    list_filter = ('source',)
    search_fields = ('content',)

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
    fields = ('role', 'content', 'created_at')
    readonly_fields = ('role', 'content', 'created_at')

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('display_title', 'username', 'user_id', 'is_favorite', 'updated_at')
    list_filter = ('is_favorite',)
    search_fields = ('title', 'username', 'messages__content')
    readonly_fields = ('id', 'user_id', 'username', 'created_at', 'updated_at')
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'role', 'content_preview', 'created_at')
    list_filter = ('role',)
    search_fields = ('content',)

    @admin.display(description='Content')
    def content_preview(self, obj):
        return obj.content[:80]
