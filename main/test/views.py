#Django
from django.shortcuts import render
from main.test.models import Patient
from django.shortcuts import HttpResponse
from django.views.generic import ListView, CreateView, UpdateView, DeleteView, DetailView
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator

from main.test.forms import PatientForm
from django.urls import reverse_lazy, reverse
from django.shortcuts import redirect

from django.contrib.auth.views import LoginView
from pathlib import Path

#Python
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import date
from datetime import datetime
import cv2
import matplotlib.pyplot as plt

# vistas basadas en funciones
@login_required
def introduction(request):
    return render(request,'index.html')

@login_required
def dashboards(request):
    return render(request,'dashboards.html')

@login_required
def buscar_paciente(request):
    data = {
        'title': 'Listado de Pacientes',
        'pacientes': Patient.objects.all()
    }

    return render(request,'search_patient.html', data)

@login_required
def add_paciente(request):
    return render(request,'add_patient.html')

@login_required
def help(request):
    return render(request,'help.html')

@login_required
def about(request):
    return render(request,'about.html')

def error_404_view(request, exception):
    return render('404.html')

#Vistas basadas en clases
@method_decorator(login_required, name='dispatch')
class PatientListView(ListView):
    model = Patient
    template_name = 'search_patient.html'

    #@method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def get_queryset(self):
        return Patient.objects.all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = 'Listado de busqueda de Pacientes'
        #print(reverse_lazy('search_patient.html'))
        return context

@method_decorator(login_required, name='dispatch')
class PatientCreateView(CreateView):
    model = Patient
    form_class = PatientForm
    template_name = 'add_patient.html'
    success_url = reverse_lazy('patient_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = 'Agregar paciente'
        return context

@method_decorator(login_required, name='dispatch')
class PatientUpdateView(UpdateView):
    model = Patient
    form_class = PatientForm
    template_name = 'add_patient.html'
    success_url = reverse_lazy('patient_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = 'Edición paciente'
        context['entity'] = 'Pacientes'
        context['list_url'] = reverse_lazy('patient_list')
        context['action'] = 'edit'
        print(context)
        return context

@method_decorator(login_required, name='dispatch')
class ReportView(DeleteView):
    model = Patient
    template_name = 'report.html'

    #@method_decorator(login_required)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = 'Reporte de un paciente'
        context['list_url'] = reverse_lazy('patient_list')
        print("context")
        print(context)
        return context


def informe(request, pk):
    '''
    #Libraries local
    from static.py import cnn

    #Paciente
    category_list = Patient.objects.all()
    context = {'object_list': category_list}
    paciente = category_list.filter(pk=pk).values()
    retinografia_left = paciente.values_list('retinografia_left', flat=True)[0]
    retinografia_rigth = paciente.values_list('retinografia_rigth', flat=True)[0]

    ######
    demo_path_lef = 'media/'+retinografia_left
    demo_filter_lef = cnn.load_image(demo_path_lef)

    img_lef_sinFiltrar = cv2.imread('media/'+retinografia_left)
    img_lef_sinFiltrar = cv2.cvtColor(img_lef_sinFiltrar, cv2.COLOR_BGR2RGB)
    circle_lef = cnn.circle_crop(img_lef_sinFiltrar)

    f = plt.figure(figsize=(20,20))

    f.add_subplot(2,3, 1)
    plt.axis('off')
    plt.imshow(circle_lef, cmap=plt.cm.binary)

    f.add_subplot(2,3, 2)
    plt.imshow(demo_filter_lef, cmap=plt.cm.binary)
    plt.axis('off')
    plt.savefig("static/py/graficos/pacientes/ret_lef.png", transparent=True, bbox_inches='tight')


    ######
    demo_path_rig = 'media/'+retinografia_rigth
    demo_filter_rig = cnn.load_image(demo_path_rig)

    img_rig_sinFiltrar = cv2.imread('media/'+retinografia_rigth)
    img_rig_sinFiltrar = cv2.cvtColor(img_rig_sinFiltrar, cv2.COLOR_BGR2RGB)
    circle_rig = cnn.circle_crop(img_rig_sinFiltrar)

    f = plt.figure(figsize=(20,20))

    f.add_subplot(2,3, 1)
    plt.axis('off')
    plt.imshow(circle_rig, cmap=plt.cm.binary)

    f.add_subplot(2,3, 2)
    plt.imshow(demo_filter_rig, cmap=plt.cm.binary)
    plt.axis('off')
    plt.savefig("static/py/graficos/pacientes/ret_rig.png", transparent=True, bbox_inches='tight')



    #Reporte PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename=informe-paciente.pdf'
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setTitle("Reporte Médico")

    #header
    c.setLineWidth(.3)
    c.setFont('Helvetica-Bold', 22)
    c.drawString(200, 750, 'Historia Clinica')
    c.setFont('Helvetica-Bold', 16)
    c.drawString(45, 720, 'Reporte médico')

    c.drawImage('static/images/logo_undav.png', 30, 700, width=80,preserveAspectRatio=True, mask='auto')
    c.drawImage('static/images/iconfinder_users-8_984122.png', 450, 400, width=100,preserveAspectRatio=True, mask='auto')

    c.setFont('Helvetica', 12)
    c.drawString(60, 690, 'Nombres: ' + str(paciente.values_list('nombres', flat=True)[0]))
    c.drawString(60, 670, 'Apellidos: ' + str(paciente.values_list('apellidos', flat=True)[0]))
    c.drawString(60, 650, 'DNI: ' + str(paciente.values_list('dni', flat=True)[0]))
    c.drawString(60, 630, 'Dirección: ' + str(paciente.values_list('dirreccion', flat=True)[0]))
    c.drawString(60, 610, 'Piso: ' + str(paciente.values_list('piso', flat=True)[0]))
    c.drawString(60, 590, 'Departamento: ' + str(paciente.values_list('departamento', flat=True)[0]))

    c.drawString(60, 570, 'Ciudad: ' + str(paciente.values_list('ciudad', flat=True)[0]))
    c.drawString(60, 550, 'Codigo Postal: ' + str(paciente.values_list('codigo_postal', flat=True)[0]))
    c.drawString(60, 530, 'Telefono: ' + str(paciente.values_list('telefono', flat=True)[0]))
    c.drawString(60, 510, 'Mail: ' + str(paciente.values_list('mail', flat=True)[0]))
    c.drawString(60, 490, 'Fecha de ingreso: ' + str(paciente.values_list('fecha_ingreso', flat=True)[0]))
    c.drawString(60, 470, 'Última visita: ' + str(paciente.values_list('fecha_modificacion', flat=True)[0])[:10])
    c.drawString(60, 420, 'Antecedentes: ' + str(paciente.values_list('antecedentes', flat=True)[0]))
    c.drawString(60, 400, 'Motivo: ' + str(paciente.values_list('motivo', flat=True)[0]))

    #c.setLineWidth(.5)
    c.setFont('Helvetica-Bold', 12)
    c.drawString(80, 320, 'Retinografía ojo izquierdo')
    c.drawImage('media/'+retinografia_left, 80, 130, width=200, height=200, preserveAspectRatio=True)
    c.drawString(300, 320, 'Retinografía ojo derecho')
    c.drawImage('media/'+retinografia_rigth, 300, 130, width=200, height=200, preserveAspectRatio=True)

    c.setFont('Helvetica-Bold', 12)
    c.drawString(480, 750, str(datetime.now().date()))
    c.line(460,747,560,747)

    c.showPage()
    #header
    c.drawImage('static/images/logo_undav.png', 30, 700, width=80,preserveAspectRatio=True, mask='auto')
    c.setFont('Helvetica-Bold', 16)
    c.drawString(150, 750, 'Retinopatía Diabetica')

    c.drawImage('static/py/graficos/lesiones RD.png', 80, 470, width=450, height=250, preserveAspectRatio=True)
    c.drawImage('static/py/graficos/Cuadro RD.png', 80, 170, width=450, height=250, preserveAspectRatio=True)

    c.showPage()

    #header
    c.drawImage('static/images/logo_undav.png', 30, 700, width=80,preserveAspectRatio=True, mask='auto')
    c.setFont('Helvetica-Bold', 16)
    c.drawString(150, 750, 'Reporte de Inteligencia Artificial')

    c.setFont('Helvetica-Bold', 10)
    c.drawString(80, 720, 'Retinografía ojo izquierdo')
    c.drawImage('static/py/graficos/pacientes/ret_lef.png', 80, 450, width=450, height=300, preserveAspectRatio=True)

    c.setFont('Helvetica-Bold', 10)
    c.drawString(80, 470, 'Retinografía ojo Derecho')
    c.drawImage('static/py/graficos/pacientes/ret_rig.png', 80, 200, width=450, height=300, preserveAspectRatio=True)

    #PREDICCIONES
    f = plt.figure(figsize=(20,20))
    plt.imshow(demo_filter_lef, cmap=plt.cm.binary)
    plt.axis('off')
    plt.savefig("static/py/graficos/pacientes/predicciones/pred_ret_lef.png", transparent=True, bbox_inches='tight')

    f = plt.figure(figsize=(20,20))
    plt.imshow(demo_filter_rig, cmap=plt.cm.binary)
    plt.axis('off')
    plt.savefig("static/py/graficos/pacientes/predicciones/pred_ret_rig.png", transparent=True, bbox_inches='tight')

    test1_path = 'static/py/graficos/pacientes/predicciones/'
    test1 = cnn.load_images_from_folder(test1_path)
    clasesPredichas = cnn.loaded_model.predict_classes(test1)
    class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    print(class_names[clasesPredichas[0]])
    print(class_names[clasesPredichas[1]])

    c.setFont('Helvetica-Bold', 10)
    c.drawString(80, 200, 'Clasificación de Retinopatía Diabética: ')
    c.drawString(80, 180, '\n Ojo izquiedo: ' + str(class_names[clasesPredichas[0]]))
    c.drawString(80, 170, '\n Ojo derecho: ' + str(class_names[clasesPredichas[1]]))

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    response.write(pdf)
    return response
    '''
    return request
