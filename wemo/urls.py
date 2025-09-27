from django.urls import path
from .views import wemo_main, wemo_toggle, wemo_refresh_status, wemo_discover

app_name = 'wemo'

urlpatterns = [
    path('', wemo_main, name='main'),
    path('toggle/<int:switch_id>/', wemo_toggle, name='toggle'),
    path('refresh/<int:switch_id>/', wemo_refresh_status, name='refresh'),
    path('discover/', wemo_discover, name='discover'),
]
