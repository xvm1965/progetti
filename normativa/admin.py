from django.contrib import admin
from .models import Documenti, Categorie, Processi, Disposizioni, UnitaOrganizzative, LegalEntity, TipiUnitaOrganizzative, Domande
from rangefilter.filters import DateRangeFilter
import os
from django.conf import settings
from django.contrib.auth.models import User, Group
from django.contrib.auth.admin import UserAdmin, GroupAdmin
from django.utils.html import format_html
from django.urls import reverse
from .my_admin import my_admin_site

from django.urls import path
from . import views
#from django.http import JsonResponse
import json
import datetime

from django.shortcuts import redirect
from .createqa import createqas
from .trainmodel import train_qgmodel, train_agmodel
from .config import CHAT_TEMP_DIR

class YearFilter(admin.SimpleListFilter):
    title = 'Anno di emissione'
    parameter_name = 'anno_emissione'

    def lookups(self, request, model_admin):
        # Creare un set degli anni disponibili nel campo data_emissione
        years = set([d.data_disposizione.year for d in model_admin.model.objects.all()])
        return [(year, year) for year in years]

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(data_disposizione__year=self.value())
        return queryset

class GenericDropdownFilter(admin.SimpleListFilter):
    title = ''
    parameter_name = ''
    related_model = None
    related_field_name = ''

    def lookups(self, request, model_admin):
        if self.related_model and self.related_field_name:
            related_objects = set([getattr(d, self.related_field_name) for d in model_admin.model.objects.all()])
            return [(obj.id, str(obj)) for obj in related_objects]
        return []

    def queryset(self, request, queryset):
        if self.value():
            filter_kwargs = {f'{self.related_field_name}_id': self.value()}
            return queryset.filter(**filter_kwargs)
        return queryset

class DisposizioniOwnerDropdownFilter(GenericDropdownFilter):
    title = 'Owner'
    parameter_name = 'owner'
    related_model = UnitaOrganizzative
    related_field_name = 'owner'

class UnitaOrganizzativeLegalEntityDropdownFilter(GenericDropdownFilter):
    title = 'Legal Entity'
    parameter_name = 'Legal Entity'
    related_model = LegalEntity
    related_field_name = 'LegalEntity'

class BaseAdmin(admin.ModelAdmin):
    list_display_links = ["descrizione"]
    search_fields = ["descrizione"]
    list_display = ['descrizione']
    

from django import forms
class DocumentiAdminForm(forms.ModelForm):
    class Meta:
        model = Documenti
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        file_directory = settings.DOCS_DIR
        used_files = Documenti.objects.values_list('file_name', flat=True)
        file_choices = []

        if os.path.exists(file_directory):
            file_choices = [
                (f, f) for f in os.listdir(file_directory)
                if os.path.isfile(os.path.join(file_directory, f)) and f not in used_files
            ]
         # Aggiungi il file attualmente selezionato (se esiste) alle scelte
        if self.instance and self.instance.pk:
            current_file = self.instance.file_name
            if current_file:
               file_choices.insert(0, (self.instance.file_name, self.instance.file_name))

        self.fields['file_name'] = forms.ChoiceField(choices=file_choices, required=False)
        self.update_owner_fields()
        
        
    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data.get('disposizione') and cleaned_data['modo'] == Documenti.Modalita.RECEPITA:
            if not cleaned_data.get('data_pubblicazione'):
                raise ValidationError("La data di pubblicazione deve essere specificata quando la modalità è 'Recepita'")
        return cleaned_data
    
   
        
    
    class Media:
        css = {
            'all': ('css/admin_custom.css',)
        }
    
   
    
    

    def update_owner_fields(self):
        self.update_queryset('legal_entity_controllante', 'owner_controllante')
        self.update_queryset('legal_entity_controllata', 'owner_controllata')

    def update_queryset(self, legal_entity_field, owner_field):
        legal_entity = getattr(self.instance, legal_entity_field)
        if legal_entity:
            self.fields[owner_field].queryset = UnitaOrganizzative.objects.filter(LegalEntity=legal_entity)
        else:
            self.fields[owner_field].queryset = UnitaOrganizzative.objects.none()

from pathlib import Path
def save_dataset_to_temp_dir(dataset):
    # Crea la directory CHAT_TEMP_DIR se non esiste
    Path(CHAT_TEMP_DIR).mkdir(parents=True, exist_ok=True)
    
    # Genera il nome del file basato sul timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    file_name = f"{timestamp}.tmp"
    
    # Percorso completo del file nella directory CHAT_TEMP_DIR
    file_path = Path(CHAT_TEMP_DIR) / file_name
    
    # Salva il dataset in formato JSON
    with open(file_path, 'w') as file:
        json.dump(dataset, file)
    
    # Restituisce il percorso completo del file creato (path + file_name)
    print (f"creato il file: {file_path}")
    return str(file_path)



from .utils import costruisci_dataset_per_agmodel
#from django.http import JsonResponse
from django.core.cache import cache


class DocumentiAdmin(BaseAdmin):
    form = DocumentiAdminForm
    change_form_template = 'admin/documenti_change_form.html'
    change_list_template = 'admin/documenti_change_list.html'

           
    def download_link(self, obj):
        if obj.id:
            url = reverse('normativa:download_document', args=[obj.id])
            return format_html('<a href="{}">Download</a>', url)
        return "-"

    def view_link(self, obj):
        if obj.id:
            file_url = f'{settings.MEDIA_URL}{obj.file_name}'  # URL pubblico per accedere al file
            return format_html('<a href="{}" target="_blank">View</a>', file_url)
        return "-"

    def rif_disposizione(self, obj):
        if obj.disposizione:
           data = obj.disposizione.data_disposizione.strftime('%d/%m/%Y')
           numero = obj.disposizione.numero
            
           return f"{numero} – {data}"
        return "-"
    
    def save_model(self, request, obj, form, change):
        if change:  # Se è una modifica
            # Recupera l'oggetto esistente dal database
            old_instance = Documenti.objects.get(pk=obj.pk)
            if old_instance.file_name != obj.file_name:
                # Solo se file_name è cambiato
                obj.qg_trained=False
                obj.ag_trained=False
                # pass
        



    download_link.short_description = ''
    view_link.short_description = ''
    rif_disposizione.short_description = 'Disposizione'
        
    list_display=BaseAdmin.list_display + ["categoria", "rif_disposizione", "file_name", "qg_trained", "ag_trained", "download_link", "view_link"]
    list_filter=["categoria", "qg_trained", "ag_trained"]
    
    readonly_fields = ('download_link', 'view_link', 'rif_disposizione', "qg_trained", "ag_trained")

    fieldsets = (
        (None, {
            'fields': (
                ('descrizione', 'categoria', "qg_trained", "ag_trained"),
                ('file_name', 'download_link', 'view_link'),
            )
        }),
        ('Ownership', {
            'fields': (
                ('legal_entity_controllante', 'owner_controllante'),
                ('legal_entity_controllata', 'owner_controllata', 'processo'),
            )
        }),
        ('Disposizione', {
            'fields': (
                ('disposizione',
                #'data_disposizione' if 'disposizione' in [field.name for field in Documenti._meta.fields] else (),
                'rif_disposizione') ,
                )
        }),
        ('Pubblicazione e modalità di approvazione', {
            'fields': (
                ('data_pubblicazione', 'data_approvazione', 'modo'),
            )
        }),
    )

    def save_model(self, request, obj, form, change):
         if 'file_name' in form.changed_data:
            obj.qg_trained = False
            obj.ag_trained = False
         super().save_model(request, obj, form, change)

        
    def has_change_permission(self, request, obj=None):
        # Verifica se l'utente è superuser
        if request.user.is_superuser:
            return True
        
        # Verifica se l'utente appartiene al gruppo 'admin'
        if request.user.groups.filter(name='admin').exists():
            return True
        
        # Altrimenti, nega il permesso di modificare
        return False
   
    def has_training_permissions (self, request, obj=None):
        return self.has_change_permission(request)
     
   
   
   
    @admin.action(description="Chat")
    def action_chat(self, dummy, request, queryset):
        # Ottieni gli ID dei documenti selezionati
        document_ids = list(queryset.values_list('id', flat=True))
        
        # Costruisci il dataset
        dataset = costruisci_dataset_per_agmodel(document_ids)
        json_dataset = json.dumps(dataset)
        
        # Genera una chiave univoca per il dataset
        dataset_key = f"chat_dataset_{request.session.session_key}"

        # Salva il dataset nella cache
        cache.set(dataset_key, json_dataset, timeout=3600)  # Timeout di 1 ora

         # Salva la chiave del dataset nella sessione
        request.session['chat_dataset_key'] = dataset_key
        request.session['show_modal'] = True  # Aggiungi un flag per mostrare la modale

     
        print (f"in action chat dataset_key: {dataset_key}")

        # Reindirizza semplicemente alla pagina senza query params (documenti_change_list)
        changelist_url = reverse(f'admin:{self.model._meta.app_label}_{self.model._meta.model_name}_changelist')
     
        return redirect(changelist_url)

        

    @admin.action(description="Train question generator")
    def action_train_qgmodel(self, dummy, request, queryset):
        document_ids = list(queryset.values_list('id', flat=True))
        request._admin_instance = self  # Imposta l'istanza di admin nel contesto della richiesta
        train_qgmodel(request, document_ids)

        return
    
    @admin.action(description="Train answer generator")
    def action_train_agmodel(self, dummy, request, queryset):
        document_ids = list(queryset.values_list('id', flat=True))
        request._admin_instance = self  # Imposta l'istanza di admin nel contesto della richiesta
        train_agmodel(request, document_ids)
        return

    @admin.action(description="Generate questions")
    def generate_questions(self, dummy, request, queryset):
        document_ids = list(queryset.values_list('id', flat=True))
        request._admin_instance = self  # Imposta l'istanza di admin nel contesto della richiesta
        createqas(request, document_ids=document_ids)
        return 
   
    def get_actions(self, request):
        actions = super().get_actions(request) or {}
        if 'action_chat' not in actions:
            actions['action_chat'] = (self.action_chat, 'action_chat', self.action_chat.short_description)
        if self.has_training_permissions(request):
            if 'generate_questions' not in actions:
                actions['generate_questions'] = (self.generate_questions, 'generate_questions', self.generate_questions.short_description)
            if 'action_train_qgmodel' not in actions:
                actions['action_train_qgmodel'] = (self.action_train_qgmodel, 'action_train_qgmodel', self.action_train_qgmodel.short_description)
            if 'action_train_agmodel' not in actions:
                actions['action_train_agmodel'] = (self.action_train_agmodel, 'action_train_agmodel', self.action_train_agmodel.short_description)

        return actions   

class DomandeAdmin(admin.ModelAdmin):
    readonly_fields = ('documento', 'risposta', 'data_creazione', 'auto_generated')
    fields = ('documento', 'domanda', 'risposta', 'rating_domanda', 'rating_risposta', 'data_creazione', 'auto_generated')
    list_display = ('domanda', 'data_creazione', 'auto_generated', 'rating_domanda_f', 'rating_risposta_f')
    list_filter=['auto_generated', 'documento']
    
   


    def change_view(self, request, object_id, form_url='', extra_context=None):
        return super().change_view(request, object_id, form_url, extra_context=extra_context)

    def add_view(self, request, form_url='', extra_context=None):
        return super().add_view(request, form_url, extra_context=extra_context)
    
    def save_model(self, request, obj, form, change):
        if change:  # Se è una modifica (non una nuova istanza)
            # Imposta il campo auto_generated a False
            obj.auto_generated = False
            obj.save()

            # Imposta il campo qg_trained del documento associato a False
            obj.documento.qg_trained = False
            obj.documento.ag_trained = False
            obj.documento.save()

    def delete_model(self, request, obj):
        # Prima di eliminare la domanda, imposta il campo qg_trained del documento associato a False
        obj.documento.qg_trained = False
        obj.documento.ag_trained = False
        obj.documento.save()
        super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        # Prima di eliminare il queryset di domande, imposta il campo qg_trained dei documenti associati a False
        documenti_ids = queryset.values_list('documento__id', flat=True).distinct()
        Documenti.objects.filter(id__in=documenti_ids).update(qg_trained=False)
        documenti = Documenti.objects.filter(id__in=documenti_ids)
        for documento in documenti:
            documento.qg_trained = False
            documento.save()  # Chiama il metodo save() per ogni oggetto

        super().delete_queryset(request, queryset)


class CategorieAdmin(BaseAdmin):
    list_display=BaseAdmin.list_display + ['processo', 'owner']
    def has_module_permission(self, request):
        return request.user.is_superuser
    
class ProcessiAdmin(BaseAdmin):
    list_display=BaseAdmin.list_display + ['padre', 'LegalEntity', 'owner']
    list_filter=["padre", "owner"]
    change_form_template = 'admin/processi_change_form.html'
    def has_module_permission(self, request):
        return request.user.is_superuser

class DisposizioniAdmin(BaseAdmin):
    list_display=BaseAdmin.list_display + ['data_disposizione', 'numero', 'owner', 'stato']
    list_filter = [YearFilter, ('data_disposizione', DateRangeFilter), DisposizioniOwnerDropdownFilter, 'stato']
    def has_module_permission(self, request):
        return request.user.is_superuser
    
class UnitaOrganizzativeAdmin(BaseAdmin):
    list_display=BaseAdmin.list_display + ['padre', 'LegalEntity', 'tipo']
    list_filter = [UnitaOrganizzativeLegalEntityDropdownFilter]
    def has_module_permission(self, request):
        return request.user.is_superuser

class LegalEntityAdmin(BaseAdmin):
    list_display=BaseAdmin.list_display +  ['padre', 'quota_controllata']
    def has_module_permission(self, request):
        return request.user.is_superuser

class TipiUnitaOrganizzativeAdmin(BaseAdmin):
    def has_module_permission(self, request):
        return request.user.is_superuser


    
my_admin_site.register (Documenti, DocumentiAdmin)
my_admin_site.register(Categorie, CategorieAdmin)
my_admin_site.register(Processi, ProcessiAdmin)
my_admin_site.register(Disposizioni, DisposizioniAdmin)
my_admin_site.register(UnitaOrganizzative, UnitaOrganizzativeAdmin)
my_admin_site.register(LegalEntity, LegalEntityAdmin)
my_admin_site.register(TipiUnitaOrganizzative, TipiUnitaOrganizzativeAdmin)
my_admin_site.register(User, UserAdmin)
my_admin_site.register(Group, GroupAdmin)
my_admin_site.register(Domande, DomandeAdmin)
