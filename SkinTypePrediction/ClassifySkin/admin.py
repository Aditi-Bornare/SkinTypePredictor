from django.contrib import admin
from ClassifySkin.models import Skin
# Register your models here.
class SkinAdmin(admin.ModelAdmin):
    list_display=['id','image']

admin.site.register(Skin,SkinAdmin)
