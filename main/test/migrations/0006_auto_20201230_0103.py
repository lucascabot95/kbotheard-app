# Generated by Django 2.1.15 on 2020-12-30 04:03

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('test', '0005_auto_20201230_0101'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patient',
            name='fecha_ingreso',
            field=models.DateTimeField(blank=True, default=datetime.date(2020, 12, 30)),
        ),
    ]
