from django.urls import path

from .views import casa_login, casa_logout, dashboard_view

urlpatterns = [
    path('login/', casa_login, name='login'),
    path("logout/", casa_logout, name='logout'),
    path('', dashboard_view, name='dashboard')
]
