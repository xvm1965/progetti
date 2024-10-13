from django.db import models
from .utils import check_for_cycles, hierarchy_error
from django.core.exceptions import ValidationError
from django.core.files.storage import FileSystemStorage
from django.utils.html import format_html




# Create your models here.

class TipiUnitaOrganizzative(models.Model):
    descrizione = models.CharField(max_length=30, unique=True)
    
    def __str__(self):
        return self.descrizione

    class Meta:
        verbose_name = "Tipo Unità Organizzativa"
        verbose_name_plural = "Tipi Unità Organizzative"

class LegalEntity(models.Model):
    descrizione = models.CharField(max_length=100, unique=True)
    padre = models.ForeignKey('self', verbose_name="Controllante", on_delete=models.PROTECT, blank=True, null=True)
    quota_controllata = models.DecimalField ('controllo', max_digits=5, decimal_places=2, default=0, blank=False, null=False )
    
    def __str__(self):
        return self.descrizione

    class Meta:
        verbose_name = "Legal Entity"
        verbose_name_plural = "Legal Entity"    

    def clean(self):
        if self.padre is None:
            #è la capogruppo
            if LegalEntity.objects.exclude(pk=self.pk).filter(padre__isnull=True).exists():
                #esiste già una capogruppo
                raise ValidationError("La capogruppo è già stata indicata")
            else:
                #la capogruppo non esiste ancora verifica la quota di cointrollo
                if self.quota_controllata != 0:
                    raise ValidationError("La capogruppo non può avere quote in controllo")
        else:
            #ha indicato una controllante
            if self.padre == self:
                # ha indicato se stessa come controllante
                raise ValidationError("La legal entity non può controllare se stessa")
            else:
                # ha indicato una controllante esistente diversa da se stessa, verifico che non ci siano cicli
                if check_for_cycles (self):
                    raise ValidationError ('indicando questa controllante si crea un ciclo')
            # controlla che la quota di controllo sia tra 0 e 100
            if self.quota_controllata <= 0 or self.quota_controllata > 100:
                raise ValidationError("la quota di controllo deve essere tra 0 e 100 ")
            
       
    def save(self, *args, **kwargs):
        # Chiama il metodo clean per eseguire le validazioni
        self.clean()
        # Salva l'istanza del modello
        super().save(*args, **kwargs)


class UnitaOrganizzative(models.Model):
    descrizione = models.CharField(max_length=100, unique=True)
    tipo = models.ForeignKey(TipiUnitaOrganizzative, on_delete=models.PROTECT, blank=False, null=False)
    LegalEntity = models.ForeignKey(LegalEntity, on_delete=models.PROTECT, blank=False, null=False)
    padre = models.ForeignKey('self', verbose_name="Unita superiore", on_delete=models.PROTECT, blank=True, null=True)
    
    def __str__(self):
        return self.descrizione
    
    def is_dependent(self, other_unita_organizzativa):
        """
        Verifica se questa unità organizzativa dipende direttamente o indirettamente
        dall'unità organizzativa specificata.
        """
        # Inizializza una lista vuota per tracciare le unità organizzative visitate
        visited = []
        # Esegui una ricerca ricorsiva per trovare l'unità organizzativa specificata
        return self._is_dependent_recursive(other_unita_organizzativa, visited)

    def _is_dependent_recursive(self, other_unita_organizzativa, visited):
        """
        Funzione di supporto ricorsiva per la verifica della dipendenza.
        """
        # Aggiungi questa unità organizzativa alla lista delle visite
        visited.append(self)
        # Controlla se l'unità organizzativa corrente è uguale a quella specificata
        if self == other_unita_organizzativa:
            return True
        # Se l'unità organizzativa corrente ha un padre, controlla la dipendenza dal padre
        if self.padre and self.padre not in visited:
            return self.padre._is_dependent_recursive(other_unita_organizzativa, visited)
        # Se l'unità organizzativa corrente ha una legal entity diversa da quella del padre, non dipende direttamente
        if self.padre and self.LegalEntity != self.padre.LegalEntity:
            return False
        # Se l'unità organizzativa corrente non ha un padre, non dipende direttamente
        if not self.padre:
            return False
        # Se l'unità organizzativa corrente ha un padre già visitato, ci sono cicli e non dipende direttamente
        return False




    def clean(self):
        if self.padre == self:
            # ha indicato se stessa come controllante
            raise ValidationError("l'unità organizzativa non può dipendere da se stessa")
        else:
            if self.padre is None:
                # non ha indicato un'unità organizzativa di riferimento dovrebbe essere l'unità root
                if self.__class__.objects.exclude(pk=self.pk).filter(padre__isnull=True, LegalEntity=self.LegalEntity).exists():
                    # ha già indicato un'unità root per questa legal entity
                    raise ValidationError("è già stata definita un unità organizzativa root per questa legal entity")
            else:
                # ha indicato un'unità organizzativa verifica che non ci siano cicli
                if check_for_cycles (self): 
                    # ha trovato il ciclo
                    raise ValidationError ('indicando questa controllante si crea un ciclo')
                else:
                    # non ci sono cicli verifica che l'unità organizzativa sia della stessa legal entity della controllante
                    if self.LegalEntity != self.padre.LegalEntity:
                        # è di un'altra legal entity
                        raise ValidationError ("L'unità organizzativa padre deve essere della stessa Legal Entity")

    def save(self, *args, **kwargs):
        # Chiama il metodo clean per eseguire le validazioni
        self.clean()
        # Salva l'istanza del modello
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Unità Organizzativa"
        verbose_name_plural = "Unità Organizzative"

import datetime
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

class Disposizioni(models.Model):
    class stati_disposizioni(models.TextChoices):
        DA_RECEPIRE = "DR", _("Da recepire")
        RECEPITA = "RE", _("Recepita")
        APPROVATA = "AP", _("Approvata")
        INFORMATIVA = "IN", _("Informativa")
        NON_APPLICABILE = "NA", _("Non applicabile")

    def __str__(self):
        return self.descrizione
    
    descrizione = models.CharField(max_length=100)
    data_disposizione = models.DateField(verbose_name='Data disposizione', default=timezone.now, blank=False, null=False)
    numero = models.PositiveSmallIntegerField (verbose_name='Numero disposizione', blank=False, null=False, unique_for_year='data_disposizione')
    stato = models.CharField(
        max_length=2,
        choices=stati_disposizioni.choices,
        default=stati_disposizioni.DA_RECEPIRE,
    ) 
    owner = models.ForeignKey(UnitaOrganizzative, on_delete=models.PROTECT, blank=False, null=False)

    class Meta:
        verbose_name = "Disposizione"
        verbose_name_plural = "Disposizioni"


class Processi(models.Model):
    descrizione = models.CharField(max_length=100)
    LegalEntity = models.ForeignKey(LegalEntity, on_delete=models.PROTECT, blank=False, null=False)
    padre = models.ForeignKey('self', verbose_name="Processo padre", on_delete=models.PROTECT, blank=True, null=True)
    owner = models.ForeignKey(UnitaOrganizzative, on_delete=models.PROTECT, blank=False, null=False)

    
    def clean(self):
        if self.padre == self:
            # ha indicato se stessa come controllante
            raise ValidationError("il processo non può dipendere da se stesso")
        else:
            # ha indicato un padre verifica che non ci siano cicli
            if check_for_cycles (self): 
                # ha trovato il ciclo
                raise ValidationError ('indicando questo processo padre  si crea un ciclo')
            else:
                # non ci sono cicli verifica che l'unità organizzativa owner del processo dipenda da quella del processo padre
                if self.padre:
                   if not self.owner.is_dependent (self.padre.owner):
                        # l'ownner del processo non dipende dall'owner del processo padre
                        raise ValidationError ("l'ownner del processo non dipende dall'owner del processo padre")

    
    def __str__(self):
        return self.descrizione

    def save(self, *args, **kwargs):
        # Chiama il metodo clean per eseguire le validazioni
        self.clean()
        # Salva l'istanza del modello
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Processo"
        verbose_name_plural = "Processi"


class Categorie(models.Model):
    descrizione = models.CharField(max_length=100)
    processo = models.BooleanField(verbose_name="Processo obbligatorio")
    owner = models.BooleanField(verbose_name="Owner obbligatorio")

    def __str__(self):
        return self.descrizione


    class Meta:
        verbose_name = "Categoria"
        verbose_name_plural = "Categorie"

from .utils import estrai_testo_da_file
from pathlib import Path
from django.conf import settings
class Documenti(models.Model):
    class Modalita(models.TextChoices):
        RECEPITA = "RE", "Recepita"
        APPROVATA = "AP", "Approvata"

    descrizione = models.CharField(max_length=100)
    categoria = models.ForeignKey(Categorie, on_delete=models.PROTECT)
    data_pubblicazione = models.DateField(blank=True, null=True)
    disposizione = models.ForeignKey(Disposizioni, on_delete=models.PROTECT, blank=True, null=True)
    file_name = models.CharField(max_length=255, blank=True, null=True)
    legal_entity_controllante = models.ForeignKey(LegalEntity, on_delete=models.PROTECT, verbose_name="controllante", blank=False, null=False, related_name='legal_entity_controllante', default=1)
    owner_controllante = models.ForeignKey(UnitaOrganizzative, on_delete=models.PROTECT, blank=True, null=True, related_name='owner_controllante')
    owner_controllata = models.ForeignKey(UnitaOrganizzative, on_delete=models.PROTECT, blank=True, null=True, related_name='owner_controllata')
    legal_entity_controllata = models.ForeignKey(LegalEntity, on_delete=models.PROTECT, verbose_name="controllata", blank=False, null=False, related_name='legal_entity_controllata', default=2)
    processo = models.ForeignKey(Processi, on_delete=models.PROTECT, blank=True, null=True)
    data_approvazione = models.DateField(blank=True, null=True)
    #trained = models.BooleanField (verbose_name="Addestrato", default=False)
    qg_trained=models.BooleanField (verbose_name="QG trained", default=False)
    ag_trained=models.BooleanField (verbose_name="AG trained", default=False)
    modo = models.CharField(
        max_length=2,
        choices=Modalita.choices,
        default=Modalita.RECEPITA,
        blank=True,
        null=True
    )
    
    
    def file_path(self):
        return Path(settings.DOCS_DIR) / self.file_name


    def __str__(self):
        return self.descrizione

    class Meta:
        verbose_name = "Documento"
        verbose_name_plural = "Documenti"

     
    def check_controllo(self, controllante, controllata):
        if controllante == controllata:
            return True
        if controllata.padre:
            return self.check_controllo(controllante, controllata.padre)
        return False

    def clean(self):
        super().clean()
        if self.categoria.owner:
            if not self.owner_controllante or not self.owner_controllata:
                raise ValidationError("Owner controllante e owner controllata devono essere specificati")
        else:
            if self.owner_controllante or self.owner_controllata:
                raise ValidationError("Owner controllante e owner controllata non devono essere specificati")
        if self.categoria.processo:
            if not self.processo:
                raise ValidationError("Il processo deve essere specificato")
        else:
            if self.processo:
                raise ValidationError("Il processo NON deve essere specificato")
        if self.modo == self.Modalita.RECEPITA and not self.disposizione:
            raise ValidationError("La disposizione deve essere specificata quando la modalità è 'Recepita'")
        if self.owner_controllante and self.owner_controllata:
            if not self.check_controllo(self.owner_controllante.LegalEntity, self.owner_controllata.LegalEntity):
                raise ValidationError("La Legal Entity della controllante deve controllare direttamente o indirettamente la Legal Entity della controllata")


# from .qa_model import QAModel

def Domande_float_format (n):
    return format (n, '6.3f')

class Domande(models.Model):
    domanda = models.TextField()
    risposta = models.TextField(blank=True, null=True)
    data_creazione = models.DateTimeField(auto_now_add=True)
    
    documento=models.ForeignKey(Documenti, on_delete=models.PROTECT, verbose_name="documento", blank=False, null=False)
    auto_generated= models.BooleanField (verbose_name="Generato", default=True)
    #rating=models.IntegerField(verbose_name='rating', blank=False, null=False)
    rating_domanda=models.FloatField(verbose_name='Rating domanda', default=0, null=False)
    rating_risposta=models.FloatField(verbose_name='Rating risposta', default=0, null=False)
    context_id =models.IntegerField()
    
    def __str__(self):
        return self.domanda
 
    
    def rating_domanda_f(self):
        return Domande_float_format (self.rating_domanda)
    rating_domanda_f.short_description='Rating q'
    
    def rating_risposta_f(self):
        return Domande_float_format (self.rating_risposta)
    rating_risposta_f.short_description='Rating a'



    def save(self, *args, **kwargs):
        # print ("sono nella save del modello Domande")
        # print ("Genero la risposta")
        # qa_model = QAModel()
        # for doc in Documenti.objects.all():
        #     testo = estrai_testo_da_file(doc.file_name)            
        #     self.risposta = qa_model.answer_question(self.domanda, testo)
        #     if self.risposta:
        #         break
        
        # if not self.risposta:
        #     self.risposta = "Risposta alla domanda " + self.domanda
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Domanda"
        verbose_name_plural = "Domande"

