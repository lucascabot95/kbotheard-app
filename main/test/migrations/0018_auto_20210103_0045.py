# Generated by Django 2.1.15 on 2021-01-03 03:45

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('test', '0017_auto_20201231_0356'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patient',
            name='departamento',
            field=models.CharField(blank=True, max_length=2, null=True, verbose_name='Departamento (Opcional)'),
        ),
        migrations.AlterField(
            model_name='patient',
            name='fecha_ingreso',
            field=models.DateField(default=datetime.datetime(2021, 1, 3, 0, 45, 9, 26022), editable=False, verbose_name='Fecha de registro'),
        ),
        migrations.AlterField(
            model_name='patient',
            name='piso',
            field=models.IntegerField(blank=True, null=True, verbose_name='Piso (Opcional)'),
        ),
        migrations.AlterField(
            model_name='patient',
            name='retinografia_left',
            field=models.ImageField(blank=True, null=True, upload_to='', verbose_name='Retinografia izquierda (Opcional)'),
        ),
        migrations.AlterField(
            model_name='patient',
            name='retinografia_rigth',
            field=models.ImageField(blank=True, null=True, upload_to='', verbose_name='Retinografia derecha (Opcional)'),
        ),
    ]