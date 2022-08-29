from django.db import models
import os 

from datetime import datetime



def update_filename(instance, filename):
    '''
    C'est cette fonction qui gère l'enregistrement des fichiers dans des dossiers spécifiques à chaque utilisateur

    Paramètres
    ----------

    instance :
            Instance de la class Fichiers
    filename :

    '''
    # Récupération du temps courant  
    now = datetime.now()
    date = now.strftime("%Y\\%m\\%d\\")

    format = instance.username
    nom = instance.file.name.replace(' ', '_')
    toadd =  os.path.join(date, format, nom)

    # Le chemin du fichier
    toremove = os.path.join("media\\", toadd)
 
    if os.path.exists(toremove):
        os.remove(toremove)
    
    
    return toadd
    
        


class Fichiers(models.Model):
    '''
    Ce modèle permet l'enregistrement des fichiers envoyés par l'utilisateur
    
    Attributs:
    ----------
    file :
        le fichier téléchargé
    username :
        le nom d'utilisateur

    '''
    file = models.FileField(upload_to=update_filename)
    username = models.CharField(max_length = 200, default="")
