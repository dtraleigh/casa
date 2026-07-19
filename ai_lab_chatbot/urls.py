from django.urls import path

from ai_lab_chatbot.views import chat_view, send_message

app_name = 'ai_lab_chatbot'

urlpatterns = [
    path('', chat_view, name='chat'),
    path('send/', send_message, name='send'),
]
