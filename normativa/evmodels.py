from sentence_transformers import SentenceTransformer, util
import importlib
import os
import json
import torch

from .config import CONTEXTS_FILENAME, CHUNKS_PER_SAMPLE
from .utils import load_agmodel


def calculate_similarity_scores(question, qa_environment):
    #"""Calcola i punteggi di similarità coseno tra la domanda e gli embeddings dei chunk."""
    question_embedding = qa_environment['embedding_model'].encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, torch.stack(qa_environment['embeddings']))[0]
    return scores

def select_top_chunks(question, qa_environment):
    """Seleziona i migliori chunks in base ai punteggi di similarità coseno."""
    scores = calculate_similarity_scores(question, qa_environment['embeddings'], qa_environment ['embedding_model'])
    top_results = torch.topk(scores, CHUNKS_PER_SAMPLE)
    top_chunks = [qa_environment['contexts'][idx] for idx in top_results.indices]
    return top_chunks

def load_context_from_file():
    """Carica i chunk e gli embedding dal file JSON, restituendo strutture vuote se il file non esiste."""
    
    if not os.path.exists(CONTEXTS_FILENAME):
        print(f"File {CONTEXTS_FILENAME} non trovato. Restituisco strutture vuote.")
        return [], []
    
    with open(CONTEXTS_FILENAME, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    context_chunks = [item['chunks'] for item in data]
    
    chunk_embeddings = []
    for item in data:
        embedded_chunk = item['embeddings']
        embeddings_shape = tuple(item['embeddings_shape'])  # Recupera la forma
        data_type = item.get('type')  # Recupera il tipo di dato

        if data_type == "tensor":
            # Converti la lista in un tensore e ripristina la forma originale
            tensor_embedding = torch.tensor(embedded_chunk)
            tensor_embedding = tensor_embedding.view(embeddings_shape)  # Ripristina la forma
            chunk_embeddings.append(tensor_embedding)
        else:
            print(f"Formato embedding non valido per il chunk: {embedded_chunk}")
            input("...errore, premi un tasto")
    
    return context_chunks, chunk_embeddings



from .utils import calculate_answer_score, load_agmodel

from transformers import pipeline
from django.core.cache import cache

from .models import Documenti  # Importa il modello Documenti
import json

def get_answer(question, request, dataset_file):
    # Variabili iniziali per la migliore risposta e i relativi dettagli
    best_answer = "Non sono riuscito a trovare una risposta adeguata."
    best_document_id = None
    best_document_name = None
    best_chunk_id = None

    dataset_key = request.session.get('chat_dataset_key')
    if not dataset_key:
        print(f"dataset key non specificata")
        return {
            'best_answer': best_answer,
            'document_id': best_document_id,
            'document_name': best_document_name,
            'chunk_id': best_chunk_id
        }

    # Recupera il dataset dalla cache
    dataset = cache.get(dataset_key)
    if not dataset:
        print(f"dataset non disponibile nella cache")
        return {
            'best_answer': best_answer,
            'document_id': best_document_id,
            'document_name': best_document_name,
            'chunk_id': best_chunk_id
        }

    print("Caricamento del modello e del tokenizer")
    qa_model, qa_tokenizer = load_agmodel()  # Carica il modello personalizzato o standard

    # Inizializza la pipeline di Question Answering
    print("Inizializzazione della pipeline di Question Answering")
    qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

    # Converti il dataset da JSON string a dizionario Python
    dataset = json.loads(dataset)

    best_score = -float('inf')  # Inizializza con un punteggio molto basso
   
    # Itera sui chunks e applica la pipeline QA
    for document_id, document_chunks in dataset.items():
        for i, chunk in enumerate(document_chunks):
            print(f"Elaborazione del chunk {i + 1}/{len(document_chunks)} dal documento {document_id}")

            # Applica la pipeline di QA su ciascun chunk di testo
            result = qa_pipeline(question=question, context=chunk)
            score = calculate_answer_score(chunk, question, result['answer'], result['score'])

            print(f"Risposta generata: {result['answer']} con score {score}")

            # Se il punteggio della risposta è migliore del precedente, aggiorna la risposta
            if score > best_score:
                best_score = score
                best_answer = result['answer']
                best_chunk_id = i  # Imposta l'ID del chunk corrente

                

                # Recupera la descrizione del documento associato utilizzando il suo ID
                try:
                    document = Documenti.objects.get(id=document_id)
                    best_document_id = document.id
                    best_document_name = document.descrizione
                except Documenti.DoesNotExist:
                    best_document_name = "Descrizione non disponibile"
                    best_document_id = document_id

    # Restituisce il risultato strutturato
    print ("\n\nin get_answer: ")
    print (f"domanda  : {question}")
    print (f"documento: {best_document_name}")
    print (f"chunk id : {best_chunk_id}")
    return {
        'best_answer': best_answer,
        'document_id': best_document_id,
        'document_name': best_document_name,
        'chunk_id': best_chunk_id
    }






