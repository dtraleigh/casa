from django.urls import path
from .views import wemo_main

app_name = 'wemo'

urlpatterns = [
    path('', wemo_main, name='main')
]
