# Generated by Django 5.0.6 on 2024-08-18 14:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('normativa', '0009_remove_domande_rating_domande_rating_domanda_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='domande',
            name='rating_domanda',
            field=models.FloatField(default=None, null=True, verbose_name='Rating domanda'),
        ),
        migrations.AlterField(
            model_name='domande',
            name='rating_risposta',
            field=models.FloatField(default=None, null=True, verbose_name='Rating risposta'),
        ),
    ]
