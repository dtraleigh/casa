from django.contrib import admin

from ai_lab_chatbot.models import Personality, HouseholdFact, UserContext


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
