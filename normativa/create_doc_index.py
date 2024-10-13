# file: yourapp/utils.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from django.conf import settings
from .models import Documenti
from .utils import estrai_testo_da_file, normalizza_testo  # Importa la tua funzione di preprocessing
from .config import DOCUMENT_INDEX

def update_faiss_index():
    # Carica il modello di embedding
    model = SentenceTransformer('distiluse-base-multilingual-cased')

    # Recupera tutte le istanze di Documenti
    documents = Documenti.objects.all()

    # Preprocessa e calcola gli embeddings dei documenti
    texts = [preprocess_document(doc) for doc in documents]
    document_embeddings = model.encode(texts)

    # Crea un nuovo indice FAISS
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(np.array(document_embeddings))

    # Salva l'indice
    faiss.write_index(index, DOCUMENT_INDEX)


def preprocess_document (document):
    file_name = document.file_path()  
    text = estrai_testo_da_file(file_name)
    text=normalizza_testo(text)
    return text 