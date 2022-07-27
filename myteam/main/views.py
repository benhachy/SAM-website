from time import perf_counter_ns
from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from .forms import RegisterForm, RecuperationForm, PasswordChangingForm

from django.core.mail import send_mail

from .models import Fichiers
from django.views.generic import FormView, DetailView, ListView

from django.contrib.auth.models import User

import sys
sys.path.append('..\\')
sys.path.append('..\\ML\\Full\\')

import os 
import ML.Full.ml as ml

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
                        file.name = request.user.username + "_" + file.name

                        # Types de fichiers acceptés
                        if file.content_type in ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                            doc = Fichiers(file = file, username = request.user.username)
                            doc.save()
                        
                        # Le cas où le type de fichier n'est pas accepté 
                        else:
                            return render(request,"main/tochange.html")
                    
                    # Traitement des fichiers par l'application Machine Learning
                    
                    # contient la nature prédite de chaque projet détécté
                    global dic_files
                    dic_files = {}

                    # contient le montant prédit pour chaque projet
                    global dic_montant
                    dic_montant= {}

                    # contient le nombre de projet que chaque fichier contient
                    global dic_projects
                    dic_projects = {}

                    # contient le nom des projets dont la synthèse est trop courte pour être évaluer
                    global dic_small
                    dic_small = {}


                    for file in files: 
                        # Le cas d'un fichier petit
                        if(len(file) < 1500):
                            dic_small[ml.delete_username(file.name, request.user.username)] = 1
                            continue
                        '''
                        # Récupération du fichier
                        now = datetime.now()
                        date = now.strftime("%Y/%m/%d")
                        nom = file.name.replace(' ','_')
                        chemin = "media/%s/%s/%s" %(date,request.user.username,nom)
                        '''

                        # Nombre de projets détéctés dans le fichier
                        num_projects = ml.read_file(file)[1]

                        if num_projects == 0:
                            dic_projects[ml.delete_username(file.name, request.user.username)] = 0
                            continue
                       
                        # Traitement du fichier par le model
                        resultat = ml.model(file)

                        dic_projects[ml.delete_username(file.name, request.user.username)] = num_projects

                        # Traite aussi le cas où le fichier contient plusieurs projets
                        if num_projects == 1:
                            dic_files[ml.delete_username(file.name, request.user.username)] = resultat[0][0]
                            dic_montant[ml.delete_username(file.name, request.user.username)] = resultat[0][1]
                        
                        else:
                            for i in range(num_projects):
                                # dic_filestionnaire contient le résultat de chaque fichier
                                dic_files[ml.delete_username(file.name, request.user.username)+ "-  Projet " + str(i+1)] = resultat[i][0]
                                dic_montant[ml.delete_username(file.name, request.user.username)+ "-  Projet " + str(i+1)] = resultat[i][1]


                    # Génération des rapports
                    now = datetime.now()
                    nom = file.name.replace(' ','_')
                    path_to_report = f"media/Reports/{request.user.username}/{request.user.username}{now.hour}{now.minute}{now.second}.txt" 

                    # Création du dossier associé à chaque utilisateur
                    try:
                        os.mkdir(f"media/Reports/{request.user.username}")
                    except:
                        pass

                    # Création du rapport
                    with open(path_to_report, 'a+') as report:      
                        for doc_name, nbr_detected_projects in dic_projects.items():
                            if nbr_detected_projects == 0:
                                report.write("Attention ! Nous ne sommes pas parvenus à detecter la description technique des travaux dans le "
                                             f"document {doc_name}. \nAssurez-vous que votre fichier ne contient que cette dernière, ou procédez à ce "
                                             " découpage manuellement avant de continuer ! \n" )
                            elif nbr_detected_projects == 1 :
                                report.write(f"Nous avons détecté un seul projet dans le document {doc_name} \n")
                            else:
                                report.write(f"Nous avons détecté {nbr_detected_projects} projets dans le document {doc_name}\n")
                       
                        
                        ## Fonction en dévelopment
                        for doc_name in dic_small:
                            report.write("\n\n" + doc_name + "\n")
                            report.write(f"La synthèse soumise par le fichier {doc_name} est trop courte pour être évaluée.\n")
                            report.write("_"*120 + "\n")
                        for doc_name, nature in dic_files.items():
                            report.write("\n\n" + doc_name + "\n")
                            report.write(f"NATURE DE PROJET: {nature} \n")
                            report.write(f"MONTANT PRÉDIT: {dic_montant[doc_name]} euros \n" + "_"*120 + "\n")
                            

                    return render(request, "main/result.html", {"dic_files": dic_files, "nbr":len(files), "dic_small":dic_small, "dic_projects":dic_projects, "path_to_report":path_to_report})

                # Le cas où aucun fichier n'est sélécionné
                else:
                    return render(request, "main/toupload.html")
        

        ## Ce block est à améliorer (on utilisera plutôt JS)
        else:
            # Le cas où la requête GET provient du bouton <<Évaluer la cohérence du montant>>        
            if request.GET.get("evaluer"):
                # Vérification si le montant est cohérent ou pas
                coherent = 0
                montant = int(request.GET["montant"])

                # Montant prédit par l'app NLP pour la fichier correspondant à la carte
                nom = request.GET["evaluer"]
                montant_predit = dic_montant[nom]

                # Ici le critère de la cohérence, ---- cnnx NLP ----
                if  ml.process_montant_pred(montant_predit, montant) : 
                    coherent = 1

                response =  render(request, "main/result.html", {"coherent":coherent, "dic_files":dic_files, "card":request.GET["evaluer"], "dic_small":dic_small, "dic_projects":dic_projects})
                return response

            # Relie la vue et le gabari main/home.html après l'authentification d'un utilisateur, autrement elle répond à la requête GET envoyé par ce dernier
            return render(request, "main/home.html")

        ##

    # Le cas où l'utilisateur n'est pas identifé : redirirection à la page d'authentification
    else :
        return redirect("/login")

  
def tarification(request):
    '''
        La vue qui gère la page de tarification
    '''
    return render(request, "main/tarification.html")


from rest_framework.views import APIView

from main.views import Fichiers

from main.serializers import FichierSerializer, UserSerializer

from rest_framework.response import Response

from rest_framework.viewsets import ReadOnlyModelViewSet, ModelViewSet

class FichierViewset(ReadOnlyModelViewSet):
    
    serializer_class = FichierSerializer

    def get_queryset(self):
        return Fichiers.objects.all()
    
    '''
    def get(self, *args, **kwargs):
        queryset = Fichiers.objects.all()
        serializer = FichierSerializer(queryset, many=True)
        return Response(serializer.data)
    '''
class UserViewset(ModelViewSet):

    serializer_class = UserSerializer

    def get_queryset(self):
        return User.objects.all()

    

