from django.urls import path
from .views import wemo_main, wemo_toggle, wemo_refresh_status, wemo_discover, away_mode_status, away_mode_toggle, \
    event_history

app_name = 'wemo'

urlpatterns = [
    path('', wemo_main, name='main'),
    path('toggle/<int:switch_id>/', wemo_toggle, name='toggle'),
    path('refresh/<int:switch_id>/', wemo_refresh_status, name='refresh'),
    path('discover/', wemo_discover, name='discover'),
    path('away-mode/status/', away_mode_status, name='away_mode_status'),
    path('away-mode/toggle/', away_mode_toggle, name='away_mode_toggle'),
    path('events/', event_history, name='event_history'),
]
