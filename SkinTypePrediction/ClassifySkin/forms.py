from django import forms
from ClassifySkin.models import Skin

class SkinForm(forms.ModelForm):
    class Meta:
        model=Skin
        fields='__all__'
