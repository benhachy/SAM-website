from django.urls import path
from . import views
#from django.contrib.auth.models import User


app_name = "main"

urlpatterns= [
    path('', views.base, name="default"),
    path('register/', views.register, name="register"),
    path('recuperation/', views.recuperation, name ="recuperation"),
    path('home/', views.home, name="home"),
    path('home/password/', views.PasswordsChangeView.as_view(template_name='registration/change-password.html'), name="change-password"),
    path('home/success-password/', views.success_password, name="password-changed"),
    path('home/tarification/', views.tarification, name="tarification")
]