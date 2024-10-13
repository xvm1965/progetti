from django.test import TestCase
from django.core.exceptions import ValidationError
from normativa.models import Documenti, Categorie, LegalEntity, UnitaOrganizzative, TipiUnitaOrganizzative, Processi

class DocumentiTestCase(TestCase):
    def setUp(self):
        
        # crea una categoria fittizia 
        self.categoria = Categorie.objects.create(descrizione='Categoria test', processo=False, owner=False)

        # crea tre legal entity fittizie
        self.legal_entity = LegalEntity.objects.create (descrizione='Legal Entity test', quota_controllata=0)
        self.legal_entity_alfa = LegalEntity.objects.create (descrizione='Legal Entity test alfa', padre= self.legal_entity, quota_controllata=100)
        self.legal_entity_beta = LegalEntity.objects.create (descrizione='Legal Entity test beta', padre= self.legal_entity, quota_controllata=100)

        # crea una tipo unità organizzativa fittizia
        self.tipo_unita_organizzativa = TipiUnitaOrganizzative.objects.create (descrizione = 'Tipo UO test')

        # Crea tre unità organizzative fittizie
        self.unita_organizzativa_root = UnitaOrganizzative.objects.create(descrizione="Unita Organizzativa Test root", LegalEntity=self.legal_entity, tipo=self.tipo_unita_organizzativa)
        self.unita_organizzativa_alfa = UnitaOrganizzative.objects.create(descrizione="Unita Organizzativa Test Alfa", LegalEntity=self.legal_entity_alfa, tipo=self.tipo_unita_organizzativa)
        self.unita_organizzativa_beta = UnitaOrganizzative.objects.create(descrizione="Unita Organizzativa Test Beta", LegalEntity=self.legal_entity_beta, tipo=self.tipo_unita_organizzativa)
        
        # Crea un processo fittizio
        self.processo = Processi.objects.create (descrizione="Processo Test", owner=self.unita_organizzativa_alfa)
     
        # Crea un documento di test
        self.documento = Documenti.objects.create(descrizione="Documento test", categoria=self.categoria, file='FileName', modo=None, owner_controllante=None, owner_controllata=None, processo=None)

    
    def log_and_assert(self, test_name, condition):
        if condition:
            print(f"{test_name}: ok")
        else:
            print(f"{test_name}: ko")
            self.fail(test_name)

    def test_documenti_no_owner_controllante(self):
         # Header
        test_name = "verifica che non possa indicare l'owner della controllante per documenti appartenenti ad una categoria che non lo richiedono"
        #imposto a false il flag sulla categoria
        self.categoria.owner=False
        self.documento.owner_controllante = self.unita_organizzativa_alfa
        try:
            self.documento.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_documenti_no_owner_controllata(self):
         # Header
        test_name = "verifica che non possa indicare l'owner della controllata per documenti appartenenti ad una categoria che non lo richiedono"
        #imposto a false il flag sulla categoria
        self.categoria.owner=False
        self.documento.owner_controllata = self.unita_organizzativa_alfa
        try:
            self.documento.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)
 
    def test_documenti_owner_controllante(self):
         # Header
        test_name = "verifica che indichi l'owner della controllante per documenti appartenenti ad una categoria che lo richiede"
        #imposto a True il flag sulla categoria
        self.categoria.owner=True
        self.documento.owner_controllante = None
        self.documento.owner_controllata = self.unita_organizzativa_alfa
        try:
            self.documento.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_documenti_owner_controllata(self):
         # Header
        test_name = "verifica che indichi l'owner della controllata per documenti appartenenti ad una categoria che lo richiede"
        #imposto a True il flag sulla categoria
        self.categoria.owner=True
        self.documento.owner_controllante = self.unita_organizzativa_alfa
        self.documento.owner_controllata = None
        try:
            self.documento.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)
    
    def test_documenti_no_processo(self):
         # Header
        test_name = "verifica che non possa indicare il processo per documenti appartenenti ad una categoria che non lo richiede"
        #imposto a false il flag sulla categoria
        self.categoria.processo=False
        self.documento.processo = self.processo
        try:
            self.documento.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_documenti_processo(self):
         # Header
        test_name = "verifica che indichi il processo per documenti appartenenti ad una categoria che lo richiede"
        #imposto a false il flag sulla categoria
        self.categoria.processo=True
        self.documento.processo = None
        try:
            self.documento.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_modo(self):
         # Header
        test_name = "verifica che se la modalità è 'RECEPITA' abbia indicato il numero di disposizione"
        #imposto a false il flag sulla categoria
        self.documento.disposizione=None
        self.documento.modo = Documenti.modalita.RECEPITA
        try:
            self.documento.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_consistenza_unita_organizzative(self):
         # Header
        test_name = "verifica che La Legal Entity della controllante controlli direttamente o indirettamente la Legal Entity della controllata"
        #imposto a true il flag owner sulla categoria
        self.categoria.owner = True
        #imposto controllante e controllata a due UO appartententi a LE indipendenti
        self.documento.owner_controllante=self.unita_organizzativa_alfa
        self.documento.owner_controllata=self.unita_organizzativa_beta
        try:
            self.documento.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)
    
    def test_consistenza_unita_organizzative_positivo(self):
         # Header
        test_name = "verifica che La Legal Entity della controllante controlli direttamente o indirettamente la Legal Entity della controllata ... positivo"
        #imposto a true il flag owner sulla categoria
        self.categoria.owner = True
        #imposto controllante e controllata a due UO appartententi a LE dipendenti
        self.documento.owner_controllante=self.unita_organizzativa_root
        self.documento.owner_controllata=self.unita_organizzativa_beta
        try:
            self.documento.clean()
            self.log_and_assert(test_name, True)
        except ValidationError:
            self.log_and_assert(test_name, False)

    def test_consistenza_unita_organizzative_invertite(self):
         # Header
        test_name = "verifica che La Legal Entity della controllante controlli direttamente o indirettamente la Legal Entity della controllata"
        #imposto a true il flag owner sulla categoria
        self.categoria.owner = True
        #imposto controllante e controllata al contrario 
        self.documento.owner_controllata=self.unita_organizzativa_root
        self.documento.owner_controllante=self.unita_organizzativa_beta
        try:
            self.documento.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)
    