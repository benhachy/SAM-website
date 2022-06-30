from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from .forms import RegisterForm, RecuperationForm, PasswordChangingForm

from django.core.mail import send_mail

from .models import Fichiers
from django.views.generic import FormView, DetailView, ListView

from django.contrib.auth.models import User

import sys
sys.path.append('c:\\Users\\Gebruiker\\OneDrive\\Bureau\\SAM')

from ML.ml import naivemodel

from datetime import datetime

from django.contrib.auth import update_session_auth_hash
from django.contrib import messages


from django.contrib.auth.views import PasswordChangeView
from django.urls import reverse_lazy

# Create your views here.

# Modifications du mot de passe
class PasswordsChangeView(PasswordChangeView):
    form_class = PasswordChangingForm
    success_url = '../../home/success-password/'

'''
La vue en dessous permet de rediriger l'utilisateur vers la page d'authentification en rentrant au site 
'''
def base(request):
    return redirect('/login')

def register(request):
    # Le cas de POST
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            form = RegisterForm()
            return render(request, "registration/success-registration.html/", {"form_register":form})
        else:
            show = 1
            form = RegisterForm()
            return render(request, "registration/registration-error.html",{"form_register":form})
    
    # Le cas de GET
    else :
        show = 1
        form = RegisterForm()
        return render(request, "registration/register.html", {"form_register":form})



def recuperation(request):
    if request.method=="POST":
        email = request.POST["email"]
        # Here we need to handle sending mails
        send_mail(
        'Recuperation compte SAM-MYTEAM',
        'Here is the message.',
        'myteam.youssef@outlook.fr',
        [email],
        fail_silently=False,
        )
        form = RecuperationForm({"email":email})
        return render(request, "registration/recuperation-sent.html", {"form_recuperation":form})
   
    form = RecuperationForm()
    return render(request, "registration/recuperation.html", {"form_recuperation":form})

'''
Cette vue permet principalement à l'utilisateur d'utiliser l'outil NLP et offre d'autre fonctionnalités: 

- Elle gère l'enregistrement des fichiers
- Elle est liée à la vue qui permet la modification du mot de passe
- Elle permets à l'utilisateur de se déconnecter
'''
def home(request):
    # Le cas où l'utilisateur est identifié
    if request.user.is_authenticated :
        if request.method=="POST":
            # Le cas où la requête provient du bouton <<UPLOAD>>
            if request.POST.get("upload"):
                # Pour s'assurer qu'un fichier est bien séléctionné par l'utilisateur
                if len(request.FILES) > 0 :
                    #file = request.FILES['file']
                    files = request.FILES.getlist('file')
                    for file in files:
                        # Types de fichiers acceptés
                        if file.content_type in ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                            doc = Fichiers(file = file, username = request.user.username)
                            doc.save()
                        
                        # Le cas où le type de fichier n'est pas accepté 
                        else:
                            return render(request,"main/tochange.html")
                    
                    # Traitement des fichiers par l'application Machine Learning
                    dic = {}

                    for file in files: 
                            now = datetime.now()
                            date = now.strftime("%Y/%m/%d")
                            chemin = "media/%s/%s/%s" %(date,request.user.username,file.name)
                            # Traitement du fichier par le model
                            resultat = naivemodel(chemin)
                            # Dictionnaire contenant le résultat de chaque fichier
                            dic[file.name] = resultat
                    return render(request, "main/result.html", {"dic": dic, "nbr":len(files)})

                # Le cas où aucun fichier n'est sélécionné
                else:
                    return render(request, "main/toupload.html")
                    
        # Relie la vue et le gabari main/home.html après l'authentification d'un utilisateur, autrement elle répond à la requête GET envoyé par ce dernier
        return render(request, "main/home.html")

    # Le cas où l'utilisateur n'est pas identifé : redirirection à la page d'authentification
    else :
        return redirect("/login")

def success_password(request):
    return render(request, "registration/success-password.html")
