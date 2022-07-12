from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from .forms import RegisterForm, RecuperationForm, PasswordChangingForm

from django.core.mail import send_mail

from .models import Fichiers
from django.views.generic import FormView, DetailView, ListView

from django.contrib.auth.models import User

import sys
sys.path.append('..\\')

from ML.ml import naivemodel

from datetime import datetime

from django.contrib.auth import update_session_auth_hash

from django.contrib.auth.views import PasswordChangeView




def base(request):
    '''
    Cette vue permet de rediriger l'utilisateur vers la page d'authentification en accédant au site 
    '''
    return redirect('/login')


def register(request):
    '''
    C'est cette vue qui gère la création d'un compte utilisateur
    '''

    # Le cas d'une requête POST
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
    
    # Le cas d'une requête GET
    else :
        show = 1
        form = RegisterForm()
        return render(request, "registration/register.html", {"form_register":form})



def recuperation(request):
    '''
    Cette vue permet la récupération du compte de l'utilisateur en envoyant un mail de récuperation à sa boite mail
    '''

    # Le cas d'une requête d'une POST
    if request.method=="POST":
        email = request.POST["email"]

        # La fonction send_mail permet l'envoi de mail à l'aide des paramètres décrits dans le fichiers myteam/settings.py
        send_mail(
        'Recuperation compte SAM-MYTEAM',
        'Here is the message.',
        'myteam.youssef@outlook.fr',
        [email],
        fail_silently=False,
        )

        form = RecuperationForm({"email":email})
        return render(request, "registration/recuperation-sent.html", {"form_recuperation":form})

    # Le cas d'une requête GET
    else : 
        form = RecuperationForm()
        return render(request, "registration/recuperation.html", {"form_recuperation":form})



class PasswordsChangeView(PasswordChangeView):
    ''' 
    Cette vue traite la modification du mot de passe de l'utilisateur
    '''
    form_class = PasswordChangingForm
    success_url = '../../home/success-password/'



def success_password(request):
    '''
    C'est cette vue qui s'affiche à l'utilisateur lorsque son mot de passe est modifié avec succès
    '''
    return render(request, "registration/success-password.html")


def home(request):
    '''
    Cette vue permet principalement à l'utilisateur d'utiliser l'outil NLP et offre d'autre fonctionnalités: 

    - Elle gère l'enregistrement des fichiers
    - Elle est liée à la vue qui permet la modification du mot de passe
    - Elle permets à l'utilisateur de se déconnecter
    '''
    
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
                    global dic_files
                    dic_files = {}

                    for file in files: 
                            now = datetime.now()
                            date = now.strftime("%Y/%m/%d")
                            nom = file.name.replace(' ','_')
                            chemin = "media/%s/%s/%s" %(date,request.user.username,nom)
                            
                            # Traitement du fichier par le model
                            resultat = naivemodel(chemin)
                            
                            # dic_filestionnaire contenant le résultat de chaque fichier
                            dic_files[file.name] = resultat
                    return render(request, "main/result.html", {"dic_files": dic_files, "nbr":len(files)})

                # Le cas où aucun fichier n'est sélécionné
                else:
                    return render(request, "main/toupload.html")
        

        ## Ce block sera probablement supprimé (on utilisera plutôt JS)
        else:
            # Le cas où la requête GET provient du bouton <<Évaluer la cohérence du montant>>        
            if request.GET.get("evaluer"):
                # Vérification si le montant est cohérent ou pas
                coherent = 0
                euros = int(request.GET["montant"])
                # Ici le critère de la cohérence, ---- cnnx NLP ----
                if  euros > 2000 : 
                    coherent = 1

                response =  render(request, "main/result.html", {"coherent":coherent, "dic_files":dic_files, "card":request.GET["evaluer"]})
                return response
                
            # Relie la vue et le gabari main/home.html après l'authentification d'un utilisateur, autrement elle répond à la requête GET envoyé par ce dernier
            return render(request, "main/home.html")

        ##

    # Le cas où l'utilisateur n'est pas identifé : redirirection à la page d'authentification
    else :
        return redirect("/login")


