# Generated by Django 2.1.15 on 2020-12-31 05:42

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('test', '0014_auto_20201231_0240'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patient',
            name='fecha_ingreso',
            field=models.DateField(default=datetime.datetime(2020, 12, 31, 2, 42, 49, 596313), verbose_name='Fecha de registro'),
        ),
    ]
