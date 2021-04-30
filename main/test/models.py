from django.db import models
from django.utils import timezone
# Create your models here.

class Patient(models.Model):
    
    nombres = models.CharField(max_length=100, verbose_name="Nombres")
    apellidos = models.CharField(max_length=100, verbose_name="Apellidos")
    dni = models.CharField(max_length=10, unique=True, verbose_name="Dni")
    
    dirreccion = models.CharField(max_length=150, verbose_name="Dirección")
    piso =  models.IntegerField(blank=True, null=True, verbose_name="Piso (Opcional)")
    departamento = models.CharField(max_length=2, blank=True, null=True, verbose_name="Departamento (Opcional)")
    ciudad = models.CharField(max_length=100, verbose_name="Ciudad")
    
    
    codigo_postal = models.CharField(max_length=8, verbose_name="Codigo_Postal")
    telefono = models.CharField(max_length=15, verbose_name="Telefono")
    mail = models.CharField(max_length=100, verbose_name="Mail")
    antecedentes = models.TextField(verbose_name="Antecedentes")
    motivo = models.TextField(verbose_name="Motivo")
    
    retinografia_left = models.ImageField(null=True, blank=True, verbose_name="Retinografia izquierda (Opcional)")
    retinografia_rigth = models.ImageField(null=True, blank=True, verbose_name="Retinografia derecha (Opcional)")

    fecha_ingreso = models.DateField(default=timezone.now, verbose_name="Fecha de registro", editable=False)
    fecha_modificacion = models.DateTimeField(auto_now_add=True, verbose_name="Fecha de modificación")

    
    class Meta:
        verbose_name = ("Paciente")
        verbose_name_plural = ("Pacientes")
        db_table = "paciente"
        ordering = ['id']

    def __str__(self):
        return self.apellidos + ", " + self.nombres

