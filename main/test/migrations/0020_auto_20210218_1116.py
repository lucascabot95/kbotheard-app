# Generated by Django 2.1.15 on 2021-02-18 14:16

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('test', '0019_auto_20210206_0006'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patient',
            name='fecha_ingreso',
            field=models.DateField(default=datetime.datetime(2021, 2, 18, 11, 16, 57, 464635), editable=False, verbose_name='Fecha de registro'),
        ),
    ]
