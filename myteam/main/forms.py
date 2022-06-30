from django import forms
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.contrib.auth.models import User 

class RegisterForm(UserCreationForm):
    '''
    Cette classe implémente le formulaire d'inscription, il s'agit d'une classe fille de UserCreationForm une classe déjà implémentée en Django

    Attributs propores à RegisterForm:
    ---------------------------------
    email : 
        représente l'adresse email par laquelle l'utilisateur veut s'inscrir
    '''
    email = forms.EmailField()

    '''
    Meta est une sous classe par laquelle on précise le modèle qui sera utilisé pour enregistrer les données de l'utilisateur, et aussi les champs que l'utilisateur devrait remplir
    '''
    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

class RecuperationForm(forms.Form):
    '''
    Il s'agit d'une classe fille de forms.Forms qui implémente un formulaire pour la récupération d'un compte utilisateur. 
    Au contraire de la classe RegisterForm, cette classe ne contient pas une sous classe Meta parce qu'elle n'enregistre pas de données.

    Attributs propores à la classe RecuperationForm:
    ------------------------------------------------
    email : 
        représente l'adresse mail de l'utilisateur qui veut récuperer son compte
    '''
    email = forms.EmailField()


class DocumentForm(forms.Form):
    '''
    C'est cette classe qui permet l'affichage du bouton de téléchargement des documents utilisateur.

    Attributs :
    ----------
    doc : 
        représente le champ fichier qui sera téléchargé par l'utilisateur
    '''
    doc = forms.FileField(label="Please, upload your document")

class PasswordChangingForm(PasswordChangeForm):
    '''
    Formulaire de changement de mot de passe

    Attributs propre à cette classe :
    ---------------------------------
        old_password :
            Le champ qui récupèrera l'ancien mot de passe de l'utilisateur
        new_password1 :
            Le champ qui récupèrera le nouveau mot de passe de l'utilisateur
        new_password2 :
            Champ de confirmation du nouveau mot de passe
        
    '''
    old_password = forms.PasswordInput(attrs={'class':'form-control','type':'password'})
    new_password1 = forms.PasswordInput(attrs={'class':'form-control','type':'password'})
    new_password2 = forms.PasswordInput(attrs={'class':'form-control','type':'password'})

    '''
    Meta est une sous classe par laquelle on précise le modèle qui sera utilisé pour enregistrer les données de l'utilisateur, et aussi les champs que l'utilisateur devrait remplir
    '''
    class Meta:
        model = User
        fields = ('old_password', 'new_password1', 'new_password2')
