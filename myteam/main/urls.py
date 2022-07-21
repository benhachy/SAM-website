from django.urls import path
from . import views
from django.contrib.auth.models import User


app_name = "main"

urlpatterns= [
    path('', views.base, name="default"),
    path('register/', views.register),
    path('recuperation/', views.recuperation),
    path('home/', views.home),
    path('home/password/', views.PasswordsChangeView.as_view(template_name='registration/change-password.html')),
    path('home/success-password/', views.success_password),
    path('home/tarification/', views.tarification),
]