from django.urls import path

from .views import casa_login, casa_logout, dashboard_view, log_download_view, logs_view

urlpatterns = [
    path('login/', casa_login, name='login'),
    path("logout/", casa_logout, name='logout'),
    path('logs/', logs_view, name='logs'),
    path('logs/download/', log_download_view, name='log_download'),
    path('', dashboard_view, name='dashboard')
]
