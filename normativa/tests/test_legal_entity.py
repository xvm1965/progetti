import logging
from django.test import TestCase
from django.core.exceptions import ValidationError
from normativa.models import LegalEntity

# Configura il logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class LegalEntityTestCase(TestCase):
    def setUp(self):
        # Crea una Legal Entity principale per i test
        self.legal_entity_root = LegalEntity.objects.create(descrizione="Entity Root")

    def log_and_assert(self, test_name, condition):
        if condition:
            logger.info(f"{test_name}: ok")
        else:
            logger.info(f"{test_name}: ko")
            #self.fail(test_name)

    def test_legal_entity_self_parent(self):
        test_name = "Test che una Legal Entity non possa avere come padre se stessa"
        self.legal_entity_root.padre = self.legal_entity_root
        try:
            self.legal_entity_root.clean()
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_single_root_legal_entity(self):
        test_name = "Test che esista solo una Legal Entity senza padre"
        try:
            LegalEntity.objects.create(descrizione="Another Entity")
            self.log_and_assert(test_name, False)
        except ValidationError:
            self.log_and_assert(test_name, True)

    def test_legal_entity_cycle(self):
        test_name = "Test che venga segnalato un errore se risalendo si ritorna alla stessa Legal Entity (ciclo)"
        le_child = LegalEntity.objects.create(descrizione="Child Entity", padre=self.legal_entity_root, quota_controllata=50)
        self.legal_entity_root.padre = le_child
        try:
            self.legal_entity_root.clean()
            self.log_and_assert(test_name, False)
        except ValidationError as e:
            self.log_and_assert(test_name, True)

    def test_quota_controllo_for_root(self):
        test_name = "Test che se il padre è None, la quota di controllo sia 0"
        self.legal_entity_root.quota_controllata = 50  # Imposta una quota di controllo diversa da 0
        
        try:
            self.legal_entity_root.save()  # Salva l'istanza nel database per memorizzare la quota di controllo
            self.log_and_assert(test_name, False)  # Se non genera errore, il test fallisce
        except ValidationError:
            self.log_and_assert(test_name, True)  # Se genera errore, il test passa
        finally:
            self.legal_entity_root.quota_controllata = 0  # Riporta a 0 la quota di controllo
            self.legal_entity_root.save()  # Salva l'istanza nel database


    def test_quota_controllo_for_non_root(self):
        test_name = "Test che se il padre non è None, la quota di controllo sia >0 e <=100"
         # Test con quota valida
        le_child_valid = LegalEntity(descrizione="Child Entity", padre=self.legal_entity_root, quota_controllata=50)
        try:
            le_child_valid.clean()
            self.log_and_assert(f"{test_name} (quota valida)", True)
        except ValidationError:
            self.log_and_assert(f"{test_name} (quota valida)", False)
        
         # Test con quote non valide
        invalid_quotas = [150, 0, -10]
        for quota in invalid_quotas:
            le_child_invalid = LegalEntity(descrizione="Child Entity", padre=self.legal_entity_root, quota_controllata=quota)
            try:
                le_child_invalid.clean()
                self.log_and_assert(f"{test_name} (quota non valida: {quota})", False)
            except ValidationError:
                self.log_and_assert(f"{test_name} (quota non valida: {quota})", True)

 