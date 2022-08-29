from django.urls import reverse_lazy
from rest_framework.test import APITestCase

from django.contrib.auth.models import User 


class TestUser(APITestCase):
    # Nous stockons l’url de l'endpoint dans un attribut de classe pour pouvoir l’utiliser plus facilement dans chacun de nos tests

    url = reverse_lazy('user-list')
    def format_datetime(self, value):
        # Cette méthode est un helper permettant de formater une date en chaine de caractères sous le même format que celui de l’api
        return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def test_list(self):
        # Créons deux catégories dont une seule est active
        utilisateur = User.objects.create(username='John', first_name='John', last_name='Kenedy', password='123abcABC$$', email="john.kenedy@us.us")
        User.objects.create(username='Donald', first_name='Donald', last_name='Trump', password='123abcABC$$', email="donald@trumpy.us")

        # On réalise l’appel en GET en utilisant le client de la classe de test
        response = self.client.get(self.url)
      
        # Nous vérifions que le status code est bien 200
        # et que les valeurs retournées sont bien celles attendues
        self.assertEqual(response.status_code, 200)
        excepted = [
            {
                'username': utilisateur.username,
                'email': utilisateur.email
            }
        ]

        self.assertEqual(excepted[0]["username"], response.json()[0].get("username"))
        self.assertEqual(excepted[0]["email"], response.json()[0].get("email"))
