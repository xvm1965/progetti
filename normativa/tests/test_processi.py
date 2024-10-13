from django.test import TestCase
from django.core.exceptions import ValidationError
from normativa.models import Processi, LegalEntity, UnitaOrganizzative, TipiUnitaOrganizzative

class ProcessiTestCase(TestCase):
    def setUp(self):
       
        # crea una tipo unità organizzativa fittizia
        self.tipo_unita_organizzativa = TipiUnitaOrganizzative.objects.create (descrizione = 'Tipo UO test')
       
        # Crea una Legal Entity root principale per i test se non esiste già una
        if not LegalEntity.objects.filter(padre__isnull=True).exists():
            self.legal_entity_root = LegalEntity.objects.create(descrizione="Legal Entity Root", quota_controllata=0)
        else:
            self.legal_entity_root = LegalEntity.objects.filter(padre__isnull=True).first()
        
        # Crea un'altra Legal Entity con il root come padre
        self.other_legal_entity = LegalEntity.objects.create(descrizione="Other Legal Entity", padre=self.legal_entity_root, quota_controllata=50)
        #self.other_legal_entity = LegalEntity.objects.create(descrizione="Other Legal Entity")

        # Crea un'unità organizzativa root per la legal entity principale
        self.unita_organizzativa_root = UnitaOrganizzative.objects.create(descrizione="Root Unit", LegalEntity=self.legal_entity_root, tipo=self.tipo_unita_organizzativa)
        self.other_unita_organizzativa_root = UnitaOrganizzative.objects.create(descrizione="Other Unit", LegalEntity=self.other_legal_entity, tipo=self.tipo_unita_organizzativa)
        self.unita_organizzativa_alfa = UnitaOrganizzative.objects.create(descrizione="Alfa Unit", LegalEntity=self.legal_entity_root, padre=self.unita_organizzativa_root, tipo=self.tipo_unita_organizzativa)
        self.unita_organizzativa_beta = UnitaOrganizzative.objects.create(descrizione="Beta Unit", LegalEntity=self.legal_entity_root, padre=self.unita_organizzativa_root, tipo=self.tipo_unita_organizzativa)

        # Crea un processo root per l'unità organizzativa root
        self.processo_root = Processi.objects.create(descrizione="Root Unit", owner=self.unita_organizzativa_alfa)

    
    def log_and_assert(self, test_name, condition):
        if condition:
            print(f"{test_name}: ok")
        else:
            print(f"{test_name}: ko")
            self.fail(test_name)

    def test_processi_self_parent(self):
         # Header
        test_name = "Test che un processo non possa avere come padre se stessa"
        self.processo_root.padre = self.processo_root
        try:
            self.processo_root.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    
    def test_processo_cycle(self):
        test_name = "Test che venga segnalato un errore se risalendo si ritorna allo stesso processo (ciclo)"
        child_unit = Processi.objects.create(descrizione="Child Unit", owner=self.unita_organizzativa_alfa, padre=self.processo_root)
        self.processo_root.padre = child_unit
        try:
            self.processo_root.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_processo_consistency_1(self):
        test_name = "Test che se specificato il padre abbia un owner dipendente dal figlio -- legal entity diverse"
        #crea un processo figlio che ha come owner un'unità organizzativa che appartiene ad un'altra legal entity 
        try:
            child_unit = Processi.objects.create(descrizione="Child Unit", owner=self.other_unita_organizzativa_root, padre=self.processo_root)
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_processo_consistency_2(self):
        test_name = "Test che se specificato il padre abbia un owner dipendente dal figlio -- stessa legal entity"
        #crea un processo figlio che ha come owner un'unità organizzativa della stessa legal entity ma indipendente dall'owenr del processo padre
        try:
            child_unit = Processi.objects.create(descrizione="Child Unit", owner=self.unita_organizzativa_beta, padre=self.processo_root)
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)
   
        