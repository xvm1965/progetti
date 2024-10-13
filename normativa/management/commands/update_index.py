# file: yourapp/management/commands/update_index.py

from django.core.management.base import BaseCommand
from normativa.create_doc_index import update_faiss_index

class Command(BaseCommand):
    help = 'Aggiorna l\'indice FAISS con i dati dei documenti'

    def handle(self, *args, **options):
        update_faiss_index()
        self.stdout.write(self.style.SUCCESS('Indice FAISS aggiornato con successo.'))
