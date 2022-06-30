from django.db import models
import os 

from datetime import datetime

# Create your models here.

def update_filename(instance, filename):
    now = datetime.now()
    date = now.strftime("%Y\\%m\\%d\\")
    format = instance.username
    toadd =  os.path.join(date, format, instance.file.name)
    toremove = os.path.join("media\\", toadd)
 
    #print(os.path.exists(toremove))
    #print(toremove)
    
    if os.path.exists(toremove):
        os.remove(toremove)
    
    
    return toadd
    
        


class Fichiers(models.Model):
    file = models.FileField(upload_to=update_filename)
    username = models.CharField(max_length = 200, default="")
