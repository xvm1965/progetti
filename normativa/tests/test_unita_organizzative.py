from django.test import TestCase
from django.core.exceptions import ValidationError
from normativa.models import LegalEntity, UnitaOrganizzative, TipiUnitaOrganizzative

class UnitaOrganizzativeTestCase(TestCase):
    def setUp(self):
        # Crea una Legal Entity root principale per i test se non esiste già una
        if not LegalEntity.objects.filter(padre__isnull=True).exists():
            self.legal_entity_root = LegalEntity.objects.create(descrizione="Legal Entity Root", quota_controllata=0)
        else:
            self.legal_entity_root = LegalEntity.objects.filter(padre__isnull=True).first()
        
        # Crea un'altra Legal Entity con il root come padre
        self.other_legal_entity = LegalEntity.objects.create(descrizione="Other Legal Entity", padre=self.legal_entity_root, quota_controllata=50)

         # crea una tipo unità organizzativa fittizia
        self.tipo_unita_organizzativa = TipiUnitaOrganizzative.objects.create (descrizione = 'Tipo UO test')
        
        # Crea un'unità organizzativa root per la legal entity principale
        self.unita_organizzativa_root = UnitaOrganizzative.objects.create(descrizione="Root Unit", LegalEntity=self.legal_entity_root, tipo=self.tipo_unita_organizzativa)

    def log_and_assert(self, test_name, condition):
        if condition:
            print(f"{test_name}: ok")
        else:
            print(f"{test_name}: ko")
            self.fail(test_name)

    def test_unita_organizzativa_self_parent(self):
        test_name = "Test che un'unità organizzativa non possa avere come padre se stessa"
        self.unita_organizzativa_root.padre = self.unita_organizzativa_root
        try:
            self.unita_organizzativa_root.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_unita_organizzativa_single_root(self):
        test_name = "Test che esista solo una root per ogni Legal Entity"
        try:
            UnitaOrganizzative.objects.create(descrizione="Another Root Unit", LegalEntity=self.legal_entity_root, tipo=self.tipo_unita_organizzativa)
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_unita_organizzativa_cycle(self):
        test_name = "Test che venga segnalato un errore se risalendo si ritorna alla stessa unità organizzativa (ciclo)"
        child_unit = UnitaOrganizzative.objects.create(descrizione="Child Unit", LegalEntity=self.legal_entity_root, padre=self.unita_organizzativa_root, tipo=self.tipo_unita_organizzativa)
        self.unita_organizzativa_root.padre = child_unit
        try:
            self.unita_organizzativa_root.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_unita_organizzativa_different_legal_entity(self):
        test_name = "Test che un'unità organizzativa non possa avere come padre un'unità di un'altra Legal Entity"
        another_unit = UnitaOrganizzative.objects.create(descrizione="Other Entity Unit", LegalEntity=self.other_legal_entity, tipo=self.tipo_unita_organizzativa)
        self.unita_organizzativa_root.padre = another_unit
        try:
            self.unita_organizzativa_root.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

   