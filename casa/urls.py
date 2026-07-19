from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    path('wemo/', include('wemo.urls', namespace='wemo')),
    path('mycroft/', include('ai_lab_chatbot.urls', namespace='ai_lab_chatbot')),
]
