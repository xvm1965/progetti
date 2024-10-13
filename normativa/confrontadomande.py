from normativa.models import Domande, Documenti
from itertools import groupby
from operator import itemgetter
domande = Domande.objects.all().order_by('context_id')
gruppi = groupby(domande, key=lambda x: x.context_id)
for context_id, domande_gruppo in gruppi:
     domande_gruppo = list(domande_gruppo)
     documenti = {domanda.documento.id for domanda in domande_gruppo}
     if len(documenti) > 1:
        print(f"\nContext ID: {context_id}")
        for domanda in domande_gruppo:
            print(f"  - Domanda: {domanda.domanda}")
            print(f"    Documento ID: {domanda.documento.id}")
        input("\nPremi un tasto per continuare al context_id successivo")
