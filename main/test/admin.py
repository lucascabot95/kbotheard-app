from django.contrib import admin
from main.test.models import Patient

# Register your models here.

class PatientAdmin(admin.ModelAdmin):
    pass

admin.site.register(Patient)