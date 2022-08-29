from rest_framework.serializers import ModelSerializer

from main.models import Fichiers

from django.contrib.auth.models import User 


class FichierSerializer(ModelSerializer):

    class Meta:
        model = Fichiers
        fields = ['file', 'username']

      

class UserSerializer(ModelSerializer):

    class Meta:
        model = User
        fields = ['username', 'email', 'last_login', 'date_joined']