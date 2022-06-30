from django import forms
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.contrib.auth.models import User 

class RegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

class RecuperationForm(forms.Form):
    email = forms.EmailField()


class DocumentForm(forms.Form):
    doc = forms.FileField(label="Please, upload your document")

class PasswordChangingForm(PasswordChangeForm):
    old_password = forms.PasswordInput(attrs={'class':'form-control','type':'password'})
    new_password1 = forms.PasswordInput(attrs={'class':'form-control','type':'password'})
    new_password2 = forms.PasswordInput(attrs={'class':'form-control','type':'password'})

    class Meta:
        model = User
        fields = ('old_password', 'new_password1', 'new_password2')
