from django.urls import path

from ai_lab_chatbot.views import (
    chat_view, send_message, history_view, resume_view,
    generate_title_view, delete_conversation_view, toggle_favorite_view,
    rename_conversation_view,
)

app_name = 'ai_lab_chatbot'

urlpatterns = [
    path('', chat_view, name='chat'),
    path('send/', send_message, name='send'),
    path('history/', history_view, name='history'),
    path('conversation/<uuid:conversation_id>/', resume_view, name='conversation'),
    path('conversation/<uuid:conversation_id>/title/', generate_title_view, name='title'),
    path('conversation/<uuid:conversation_id>/rename/', rename_conversation_view, name='rename'),
    path('conversation/<uuid:conversation_id>/delete/', delete_conversation_view, name='delete'),
    path('conversation/<uuid:conversation_id>/favorite/', toggle_favorite_view, name='favorite'),
]
