# Generated by Django 2.1.15 on 2020-12-31 05:59

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('test', '0015_auto_20201231_0242'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patient',
            name='fecha_ingreso',
            field=models.DateField(default=datetime.datetime(2020, 12, 31, 2, 59, 31, 972379), verbose_name='Fecha de registro'),
        ),
    ]
