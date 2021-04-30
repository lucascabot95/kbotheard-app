from django.forms import ModelForm, TextInput
from main.test.models import Patient

class PatientForm(ModelForm):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for form in self.visible_fields():
            form.field.widget.attrs['class']='form-control'
            form.field.widget.attrs['autocomplete']='off'

        self.fields['nombres'].widget.attrs['autofocus'] = True
    
    class Meta:
        model = Patient
        fields = '__all__'
        exclude = ['fecha_ingreso', 'fecha_modificacion']

