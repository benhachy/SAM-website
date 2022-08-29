# Generated by Django 4.0.5 on 2022-06-29 09:43

from django.db import migrations, models
import main.models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fichiers',
            name='file',
            field=models.FileField(upload_to=main.models.update_filename),
        ),
        migrations.AlterField(
            model_name='fichiers',
            name='username',
            field=models.CharField(default='', max_length=200),
        ),
    ]