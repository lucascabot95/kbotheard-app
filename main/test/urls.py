from django.urls import path
from main.test.views import introduction, dashboards, PatientListView, PatientCreateView, PatientUpdateView, help, about, ReportView, informe
from django.conf.urls import url

urlpatterns = [
    path('introduction/', introduction, name="index"),
    path('dashboards/', dashboards, name="dashboards"),
    path('search/', PatientListView.as_view(), name="patient_list"),
    path('add/', PatientCreateView.as_view(), name="add_patient"),
    path('edit/<int:pk>/', PatientUpdateView.as_view(), name="edit_patient"),
    path('report/<int:pk>/', ReportView.as_view(), name="report_patient"),
    path('help/', help, name="help"),
    path('about/', about, name="about"),
    path('informe/<int:pk>/', informe, name="informe"),
]

handler404 = 'test.views.error_404_view'
