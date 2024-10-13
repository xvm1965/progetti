import json
from pathlib import Path
from .models import Domande, Documenti
from .config import TRAINED_QUESTION_GENERATOR, STANDARD_QUESTION_GENERATOR, CONTEXTS_FILENAME
from .config import TRAINED_ANSWER_GENERATOR, STANDARD_ANSWER_GENERATOR, CONTEXTS_FILENAME
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_scheduler
import torch.nn as nn
from django.core.management.base import BaseCommand
from sklearn.model_selection import train_test_split
from .utils import load_agmodel, load_qgmodel
import torch
import time
import math


def costruisci_dataset_per_training_qgmodel(documents):
    # Estrai tutte le domande dal database
    domande = Domande.objects.filter(documento__in=documents) #subset delle domande nel dataset associate ai documenti passati come parametro
    dataset = []
    
    # Carica l'intero file JSON che contiene tutti i documenti e i chunk
    json_file = Path(CONTEXTS_FILENAME)  # CONTEXTS_FILENAME è la variabile che contiene il percorso del file JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    """
    il file json è composto da una serie di dizionari con questa struttura
    document_id --> id del docuemnto
    chunks      --> testo del documento suddiviso in sottosegmenti
    embedding   --> non utilizzato
    embedding_shape --> non utilizzato
    type --> non utilizzato
    """
    
    # Dizionario per una rapida ricerca dei chunks per document_id
    document_chunks = {}    #crea il dizionario
    for document in data:
        document_id = str(document["document_id"])  # Converti l'ID in stringa se necessario
        
        """
        alla chiave document_id del dizionario document_chunks
        viene associato il valore della chiave "chunks" del dizionario document. 
        Se la chiave "chunks" non esiste nel dizionario document, 
        viene associato il valore predefinito [] (una lista vuota)
        quindi document_chunks conterrà una serie di dizionari in cui la chiave è l'id del documento 
        ed il valore è l'elenco dei chunks in cui è stato suddiviso
        document_chunks= {1:[[chunk1],[chunk2], ... , [chunk n]]}, {2:[[chunk1],[chunk2], ... , [chunk m]]}, {3:[[chunk1],[chunk2], ... , [chunk p]]}

        """
        document_chunks[document_id] = document.get("chunks", []) 
    
    # Costruisci coppie contesto-domanda
    for domanda in domande:
        score=domanda.rating_domanda
        if score:
            document_id = str(domanda.documento.id)  # recupera l'id del documento associato alla domanda Converto l'ID in stringa per fare il match nel JSON
            context_id = domanda.context_id          # recupera l'id del chunk associato alla domanda
                
            # Verifica che il document_id esista nel dizionario e che l'indice context_id sia valido
            if document_id in document_chunks:          # se l'id del documento è tra le chiavi di document_chunks (dovrebbe esserci ma non si sa mai)
                chunks = document_chunks[document_id]   # recupera tutti i chunks associati al documento
                if context_id < len(chunks):            # se l'id del chunk associato alla domanda corrisponde ad una posizone esistente dell'elenco dei chunks (dovrebbe essere ma non si sa mai)
                    contesto = chunks[context_id]
                    # aggiunge un nuovo elemento - un dizionario di 4 coppie chiave valore al dataset
                    dataset.append({
                        "input_text": contesto,              # alla chiave input_text associa il chunk associato alla domanda
                        "target_text": domanda.domanda,      # alla chiave target_text associa il testo della domanda che ha recuperato dal dataset
                        "score": score,                      # alla chiave score associa lo score della domanda che ha recuperato dal dataset
                        "document_id": document_id           # alla chiave document_id associa l'id del documento associato alla domanda che ha recuperato dal dataset
                    })

    """
    a questo punto dataset contiene tanti dizionari con le coppie
    
                    "input_text": contesto,              # alla chiave input_text associa il chunk associato alla domanda
                    "target_text": domanda.domanda,      # alla chiave target_text associa il testo della domanda che ha recuperato dal dataset
                    "score": domanda.rating_domanda,     # alla chiave score associa lo score della domanda che ha recuperato dal dataset
                    "document_id": document_id           # alla chiave document_id associa l'id del documento associato alla domanda che ha recuperato dal dataset
    
    quante sono le domande associate ai documenti selezionati
    """
    return dataset

def costruisci_dataset_per_training_agmodel(documents):
    # Estrai tutte le domande relative ai documenti selezionati
    domande = Domande.objects.filter(documento__in=documents)
    
    dataset = {"version": "1.0", "data": []}
    
    # Carica l'intero file JSON che contiene tutti i documenti e i chunk
    json_file = Path(CONTEXTS_FILENAME)  # CONTEXTS_FILENAME è la variabile che contiene il percorso del file JSON
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Crea un dizionario per associare rapidamente document_id ai suoi chunk
    document_chunks = {}
    for document in data:
        document_id = str(document["document_id"])  # Converti l'ID in stringa se necessario
        document_chunks[document_id] = document.get("chunks", [])
    
    # Inizializza il contatore dei contesti
    total_contexts = 0
    
    # Costruisci il dataset
    for document in documents:
        document_id = str(document.id)
        if document_id not in document_chunks:
            continue  # Se il documento non ha chunk associati nel file JSON, passa oltre

        paragraphs = []
        chunks = document_chunks[document_id]  # Ottieni tutti i chunk per il documento corrente
        
        for i, chunk in enumerate(chunks):
            context = chunk  # Il contesto del chunk
            
            # Filtra le domande relative a questo chunk specifico nel documento corrente
            domande_per_chunk = domande.filter(documento=document, context_id=i)
            
            qas = []
            for domanda in domande_per_chunk:
                pos = context.find(domanda.risposta) # cerca la posizione inziale della risposta nel contesto
                if pos > -1:                         # se la trova ....
                    qas.append({
                        "id": str(domanda.id),
                        "question": domanda.domanda,
                        "answers": [
                            {
                                "text": domanda.risposta,
                                "answer_start": pos,  # Trova la posizione di inizio della risposta nel contesto
                                "rating": domanda.rating_risposta  # Aggiungi il rating della risposta
                            }
                        ],
                        "rating": domanda.rating_domanda,  # Aggiungi il rating della domanda
                        "is_impossible": True  # Se hai domande a cui non è possibile rispondere, cambia a True
                    })

            if qas:  # Aggiungi solo se ci sono domande associate a questo chunk
                paragraphs.append({
                    "context": context,
                    "qas": qas
                })
        
        # Stampa un messaggio per ogni nuovo contesto considerato
        if paragraphs:  # Aggiungi solo se ci sono paragrafi con domande valide
            dataset["data"].append({
                "title": document.descrizione,
                "paragraphs": paragraphs
            })
            # Incrementa il contatore dei contesti
            total_contexts += len(paragraphs)
    # Stampa il numero totale di contesti elaborati
    print(f"Numero totale di contesti elaborati: {total_contexts}")
   
    
    return dataset



def train_qgmodel(request, document_ids):
    documents = Documenti.objects.filter(id__in=document_ids, qg_trained=False)
    if not len(documents):
        return
    dataset_di_training = costruisci_dataset_per_training_qgmodel(documents)   # costruisce il dataset di training
    """
    il dataset contiene tanti dizionari con le coppie
    
                    "input_text": contesto,              # alla chiave input_text associa il chunk associato alla domanda
                    "target_text": domanda.domanda,      # alla chiave target_text associa il testo della domanda che ha recuperato dal dataset
                    "score": domanda.rating_domanda,     # alla chiave score associa lo score della domanda che ha recuperato dal dataset
                    "document_id": document_id           # alla chiave document_id associa l'id del documento associato alla domanda che ha recuperato dal dataset
    
    quante sono le domande associate ai documenti selezionati
    """

    model, tokenizer = load_qgmodel()                                          # carica modello e tokenizer
    optimizer = AdamW(model.parameters(), lr=5e-5)                             # definisce l'optimizer ed il learning rate iniziale
    train_dataset, eval_dataset = train_test_split(dataset_di_training, test_size=0.2, random_state=42) # suddivide il dataset in training ed evaluation
    num_epochs=30
    warmup_steps=30
    optimizer = AdamW(model.parameters(), lr=5e-5)   
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_epochs * len(train_dataset))
    loss_history=[]
    num_iteration=0
    unchanging_ratio = 0.01
    unchanging=0
    delta=0
    mov_avg=1
    end_training=False
    for epoch in range(num_epochs):
        
        model.train()
        num_pair = 1
        for pair in train_dataset:
            pair_time=time.time()
            if pair['score']: # non considera le domande con score nullo
                num_iteration+=1
                # Tokenizza il contesto (input) e la domanda (target)
                inputs = tokenizer(pair['input_text'], return_tensors="pt", padding=True, truncation=True)
                targets = tokenizer(pair['target_text'], return_tensors="pt", padding=True, truncation=True)
                # Applica il modello al contesto (con il target come label) e calcola la loss
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
                loss = outputs.loss/pair['score']       # adegua lo loss con lo score 
                loss.backward()           #calcola i gradienti
                optimizer.step()          #aggiorna i pesi ed i parametri
                lr_scheduler.step()       #modifica il passo di aggiornamento
                optimizer.zero_grad()     #azzera i gradienti

                # Salva la loss
                loss_history.append(loss.item())
                if len(loss_history) > warmup_steps:
                    loss_history.pop(0)
                
                 # calcola la media mobile
                prev_mov_avg=mov_avg

                # se la variazione della media mobile è inferiore a unchanging_ratio per più di warmup_steps steps interrompe il training
                mov_avg = sum(loss_history) / min(warmup_steps, num_iteration)
                
                delta = abs((mov_avg-prev_mov_avg)/prev_mov_avg)
                if delta <unchanging_ratio: # se il delta è minore di unchanging_ratio per più di warmup_steps steps interrompe il training
                    if unchanging == warmup_steps:
                        end_training=True
                        break
                    unchanging +=1
                else:
                    unchanging = 0
            
            print(f"epoch: {epoch} - item: {num_pair:4} - loss: {loss.item():7.4f} - mov_avg: {mov_avg:7.4f} - delta: {delta:7.4f} - unchanging: {unchanging:3} time: {(time.time()-pair_time):7.4f}")
            num_pair+=1
            if end_training: break
                
            
    
    # Salva il modello e il tokenizer 
    model.save_pretrained("TRAINED_QUESTION_GENERATOR")
    tokenizer.save_pretrained("TRAINED_QUESTION_GENERATOR")
    # Aggiorna il campo qg_trained a True per tutti i documenti selezionati
    documents.update(qg_trained=True)    

    print (f"{len(documents)} trained in {(time.time()-start_time):7.4f} seconds")
        
def preprocess_data(dataset, tokenizer):
    questions = []
    contexts = []
    start_positions = []
    end_positions = []
    rating_questions =[]
    rating_answers=[]

    for document in dataset['data']:
        num_context = 0
        for paragraph in document['paragraphs']:
            num_context+=1
            context = paragraph['context']
            # print(f"\n\n--- Contesto:\n{context}\n")  # Debug: Visualizza il contesto intero
            num_question =0
            
            for qa in paragraph['qas']:
                num_question+=1
                question = qa['question']
                rating_question =qa ['rating']
                answer = qa['answers'][0]['text']
                rating_answer=qa['answers'][0]['rating']
                answer_start = qa['answers'][0]['answer_start']
                answer_end = answer_start + len(answer)  # Calcola la fine della risposta

 

                # Tokenizza il contesto e ottieni l'offset_mapping
                tokenized_context = tokenizer(
                    context, 
                    truncation=True, 
                    padding='max_length', 
                    max_length=512, 
                    return_tensors="pt", 
                    return_offsets_mapping=True
                )
                
                # Ottieni offset_mapping
                """
                tokenized_contest ['offset_mapping'][0] è una lista di tuple in cui ciascuna tupla contiene due numeri (start, end) 
                che indicano le posizioni nel testo originale a cui corrisponde un token prodotto dal tokenizer
                """
                offset_mapping = tokenized_context['offset_mapping'][0]  # Mappa token a posizioni nel testo

                # Trova gli indici di inizio e fine della risposta nei token
                start_position = None
                end_position = None

                for i, (start, end) in enumerate(offset_mapping):                       # start ed end sono l'inizio e la fine del corrispondente del  token i-esimo
                    if start <= answer_start < end:                                     # se l'inizio della risposta è tra l'inzio e la fine del token i-esimo
                        start_position = i                                                   # la posizione iniziale è nel token  i-esimo
                    if answer_start == end:                                             # se l'inizio della risposta è alla fine del token i-esimo
                        start_position = min (i+1, tokenizer.model_max_length)               # la posizione iniziale è il minimo tra il token successivo e l'ultimo token
                    if start < answer_end <= end:                                       # se la fine della risposta è tra l'inzio e la fine del token i-esimo
                        end_position = i                                                     # la posizione finale è nel token  i-esimo
                        break
                    if answer_end == start:                                             # se la fine della risposta è all'inizio del blocco
                        end_position = max(start_position, i-1)                              # la posizione finale è il massimo tra l'ultimo token ed il token precedente
                        break

                # Gestisci i casi in cui le posizioni non sono trovate
                if start_position is None:
                    start_position = tokenizer.model_max_length
                if end_position is None:
                    end_position = tokenizer.model_max_length

               
       
                questions.append(question)
                contexts.append(context)
                start_positions.append(start_position)
                end_positions.append(end_position)
                rating_questions.append(rating_question)
                rating_answers.append(rating_answer)
    return questions, contexts, start_positions, end_positions, rating_questions, rating_answers

def train_agmodel(request, document_ids):
    documents = Documenti.objects.filter(id__in=document_ids, ag_trained=False)
    if not documents.exists():
        print("Nessun documento selezionato per l'addestramento.")
        return
    dataset_di_training = costruisci_dataset_per_training_agmodel(documents)
    
    # Controlla che il dataset contenga almeno una domanda
    if not dataset_di_training['data']:
        print("Il dataset di training non contiene domande.")
        return
    
    model, tokenizer = load_agmodel()
    
    # Preprocess the dataset
    questions, contexts, start_positions, end_positions, rating_questions, rating_answers = preprocess_data(dataset_di_training, tokenizer)

        
    # Split the dataset
    train_questions, eval_questions, \
    train_contexts, eval_contexts, \
    train_start_positions, eval_start_positions, \
    train_end_positions, eval_end_positions, \
    train_rating_questions, eval_rating_questions, \
    train_rating_answers, eval_rating_answers = train_test_split(
        questions, contexts, 
        start_positions, end_positions, 
        rating_questions, rating_answers, 
        test_size=0.2, 
        random_state=42
    )
    
    num_epochs=30
    warmup_steps=30
    optimizer = AdamW(model.parameters(), lr=5e-5)   
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_epochs * len(train_questions))
    loss_history=[]
    num_iteration=0
    unchanging_ratio = 0.01
    unchanging=0
    delta=0
    mov_avg=1
    end_training=False
    for epoch in range(num_epochs):
        num_pair=1
        for idx in range(len(train_questions)):
            pair_time=time.time()
            answer_score = train_rating_answers[idx]
            if answer_score:
                question = train_questions[idx]
                context = train_contexts[idx]
                num_iteration+=1
                # Tokenizza la domanda e il contesto
                inputs = tokenizer(
                    question, 
                    context, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512  
                )

                start_position = train_start_positions[idx]
                end_position = train_end_positions[idx]
                
                # Applica il modello e calcola la loss (con le posizioni delle risposte come target)
                outputs = model(
                    input_ids=inputs.input_ids, 
                    attention_mask=inputs.attention_mask, 
                    start_positions=torch.tensor([start_position]),  # Posizione inizio risposta
                    end_positions=torch.tensor([end_position])       # Posizione fine risposta
                )

                # Calcola la loss adattandola con lo score della domanda e della risposta
                loss = outputs.loss / answer_score
                loss.backward()  # Calcola i gradienti

                optimizer.step()  # Aggiorna i pesi del modello
                lr_scheduler.step()  # Aggiorna il learning rate
                optimizer.zero_grad()  # Azzera i gradienti per il prossimo step
                
                # Salva la loss
                loss_history.append(loss.item())
                if len(loss_history) > warmup_steps:
                    loss_history.pop(0)
                
                # calcola la media mobile
                prev_mov_avg=mov_avg

                # se la variazione della media mobile è inferiore a unchanging_ratio per più di warmup_steps steps interrompe il training
                mov_avg = sum(loss_history) / min(warmup_steps, num_iteration)
                
                delta = abs((mov_avg-prev_mov_avg)/prev_mov_avg)
                if delta <unchanging_ratio: # se il delta è minore di unchanging_ratio per più di warmup_steps steps interrompe il training
                    if unchanging == warmup_steps:
                        end_training=True
                        break
                    unchanging +=1
                else:
                    unchanging = 0
            print(f"epoch: {epoch} - item: {num_pair:4} - loss: {loss.item():7.4f} - mov_avg: {mov_avg:7.4f} - delta: {delta:7.4f} - unchanging: {unchanging:3} time: {(time.time()-pair_time):7.4f}")
            num_pair+=1
            if end_training: break
    # Salva il modello e il tokenizer 
    model.save_pretrained("TRAINED_ANSWER_GENERATOR")
    tokenizer.save_pretrained("TRAINED_ANSWER_GENERATOR")
    documents.update(ag_trained=True)


