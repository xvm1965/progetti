import os
import time
import json
from datetime import datetime
from transformers import pipeline
from .utils import clear_screen, load_and_chunk_file, calculate_answer_score
from .utils import load_agmodel, load_qgmodel, load_chunkmodel, load_generative_agmodel
from .config import QA_QUERY_PER_CHUNKS, QA_DO_SAMPLE, QA_TOP_K, QA_TOP_P, QA_FILENAME, CONTEXTS_FILENAME
from .config import QA_TEMPERATURE, QA_MAX_LENGTH, QA_MIN_LENGTH, QA_REPETITION_PENALTY, QA_LENGTH_PENALTY, QA_EARLY_STOPPING
from collections import defaultdict
from .models import Documenti, Domande
from .utils import calcola_punteggio_singola_domanda
import torch
import pdb
import Levenshtein

def save_to_json(filename, document_id, chunks, embeddings):
    # Converti gli embeddings in una lista
    #embeddings_as_list = embeddings.tolist()
    
    # Salva la forma originale del tensore
    #embeddings_shape = embeddings.shape
    
    # Struttura del dato da salvare
    data = {
        "document_id": document_id,
        "chunks": chunks,
        # "embeddings": embeddings_as_list,
        # "embeddings_shape": embeddings_shape,  # Salva la forma del tensore
        "embeddings": None,
        "embeddings_shape": None,  # Salva la forma del tensore

        "type": "tensor"  # Aggiungi un campo per indicare che il tipo è un tensore
    }
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
        existing_data = [item for item in existing_data if item["document_id"] != document_id]
    else:
        existing_data = []
    
    # Aggiungi i nuovi dati
    existing_data.append(data)
    
    # Scrivi nel file JSON
    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)

def get_current_params (ask_to_user=False):

    current_params = {
        'num_return_sequences': QA_QUERY_PER_CHUNKS,
        'do_sample': QA_DO_SAMPLE,
        'top_k': QA_TOP_K,
        'top_p': QA_TOP_P,
        'output_file': QA_FILENAME,
        'temperature': QA_TEMPERATURE, 
        'max_length': QA_MAX_LENGTH, 
        'min_length': QA_MIN_LENGTH, 
        'qa_repetition_penalty': QA_REPETITION_PENALTY, 
        'qa_length_penalty': QA_LENGTH_PENALTY, 
        'qa_early_stopping': QA_EARLY_STOPPING,
        'num_beams':QA_QUERY_PER_CHUNKS
    }

    
    return current_params

def createqas(request, document_ids=None):
    documents = Documenti.objects.filter(id__in=document_ids, qg_trained=False)
   
    if not documents:
        return 
        
    chunk_model, chunk_tokenizer = load_chunkmodel()
    chunk_pipeline=pipeline("feature-extraction", model=chunk_model, tokenizer=chunk_tokenizer)

    qg_model, qg_tokenizer = load_qgmodel() # inizializza checkpoint e tokenizer del generatore di domande
    qg_pipeline=pipeline ("text2text-generation", model=qg_model, tokenizer=qg_tokenizer) # inizializzazione della pipeline per la generazione delle domande
    qg_params=get_current_params() # prende i parametri per la generazione delle domande

    ag_model, ag_tokenizer = load_agmodel() # inizializza checkpoint e tokenizer del generatore di risposte
    ag_pipeline = pipeline("question-answering", model=ag_model, tokenizer=ag_tokenizer)

    ag_generative_model, ag_generative_tokenizer=load_generative_agmodel() ## inizializza checkpoint e tokenizer del generatore di risposte con il modello generativo
        
    def get_max_num_tokens (s):
        return max (len(qg_tokenizer.encode(chunk_text, add_special_tokens=False)), 
                    len(ag_tokenizer.encode(chunk_text, add_special_tokens=False)),
                    len(ag_generative_tokenizer.encode(chunk_text, add_special_tokens=False)))

    # definisce le dimensioni del blocco di testo
    chunk_size = 512
    
    for document in documents:  # per ogni documento
        print (f"document: {document.descrizione}")
        doc_time=time.time()
        # suddivide il documento in chunks
        chunks = load_and_chunk_file(document.file_name, chunk_size, chunk_tokenizer, chunk_pipeline)  

        results = []
        
        for chunk_id, chunk_text in enumerate (chunks):  # per ogni chunk
 
            """ 
            elimina una parola dalla fine del chunk sino a che il massimo del numero di token
            del tokenizzatore del generatore di domande e del  tokenizzatore del generatore di risposte
            è superiore a chunk_size
            """
            words = chunk_text.split()   # suddivide il testo in parole
            num_words = len (words)      # numero di parole
            num_tokens = get_max_num_tokens (chunk_text) # calcola il massimo numero di token
            
            while num_tokens > chunk_size: # sino a che il numero di token è maggiore di chunk_size
                num_words -=1 
                chunk_text =  ' '.join(words[:num_words])  # elimina l'ultima parola
                num_tokens = get_max_num_tokens (chunk_text) # calcola il massimo numero di token

            chunk_questions = generate_questions(qg_pipeline, chunk_text, qg_params)   #genera le domande                     
           
            # Itera sulle domande 
            for question in chunk_questions: #per ogni domanda
                answer = ag_pipeline(question=question['question'], context=chunk_text, handle_impossible_answer=True)
                #{'score': 0.93, 'start': 80, 'end': 86, 'answer': 'Blasco'} struttura della risposta della pipeline
                answer_text=answer.get('answer', "No answer generated")
                # Salva e stampa solo se la risposta è valida
                
                if answer_text != "No answer generated" and len(answer_text) > 5:
                    results.append ({
                            "question": question['question'],
                            "answer": answer_text,
                            "question_score": question['score'],
                            "answer_score": calculate_answer_score(question['question'], answer_text, chunk_text, answer.get('score', 0))['final_score'],
                            "context_ids": chunk_id
                    })
                    
                    print (f"\n\nquestion: {results[-1]['question']}[{results[-1]['question_score']:7.2f}]")
                    print (f"extracted answer  : {results[-1]['answer']}[{results[-1]['answer_score']:7.2f}]")

                    input_text = f"domanda: {question['question']} contesto: {chunk_text}"
                    inputs = ag_generative_tokenizer(input_text, return_tensors="pt")
                    outputs = ag_generative_model.generate(inputs['input_ids'], max_length=50)
                    generative_answer=ag_generative_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    
                    results.append ({
                            "question": question['question'],
                            "answer": generative_answer,
                            "question_score": question['score'],
                            "answer_score": calculate_answer_score(question['question'], answer_text, chunk_text, 1)['final_score'],
                            "context_ids": chunk_id
                    })
                    
                    print (f"generated answer  : {results[-1]['answer']}[{results[-1]['answer_score']:7.2f}]")




                
        
        if len(results):
            save_to_json(CONTEXTS_FILENAME, document.id, chunks, None)  # salva chunks sul file json
            Domande.objects.filter(documento=document, auto_generated=True).delete()  # cancella tutte le domande auto generate precedenti
            for result in results:
                domanda = Domande(
                    domanda=result['question'],
                    risposta=result['answer'],
                    rating_domanda=result['question_score'],
                    rating_risposta=result['answer_score'],
                    context_id=result['context_ids'],
                    documento=document,
                    auto_generated=True,
                    data_creazione=datetime.now()
                )
                domanda.save()  # registra le nuove domande
                # Aggiorna i campi ag_trained e qg_trained a False 
                document.qg_trained = False
                document.ag_trained = False
                document.save()
          
        print_summary_query_generation (results)
  
def print_summary_query_generation(document_questions):
    # Dati aggregati per ogni context_id
    context_data = defaultdict(lambda: {
        'total_questions': 0,
        'total_answers': 0,
        'question_total_score': 0.0,
        'question_max_score': float('-inf'),
        'question_min_score': float('inf'),
        'question_scores': [],
        'answer_total_score': 0.0,
        'answer_max_score': float('-inf'),
        'answer_min_score': float('inf'),
        'answer_scores': []
    })

    # Itera attraverso le domande e aggrega i dati per ogni context_id
    for entry in document_questions:
        context_id = entry['context_ids']
        question_score = entry['question_score']
        answer_score = entry['answer_score']
        
        # Aggrega i dati delle domande
        context_data[context_id]['total_questions'] += 1
        context_data[context_id]['question_total_score'] += question_score
        context_data[context_id]['question_scores'].append(question_score)
        context_data[context_id]['question_max_score'] = max(context_data[context_id]['question_max_score'], question_score)
        context_data[context_id]['question_min_score'] = min(context_data[context_id]['question_min_score'], question_score)
        
        # Aggrega i dati delle risposte
        context_data[context_id]['total_answers'] += 1
        context_data[context_id]['answer_total_score'] += answer_score
        context_data[context_id]['answer_scores'].append(answer_score)
        context_data[context_id]['answer_max_score'] = max(context_data[context_id]['answer_max_score'], answer_score)
        context_data[context_id]['answer_min_score'] = min(context_data[context_id]['answer_min_score'], answer_score)

    # Stampa i risultati per ogni context_id
    for context_id, data in context_data.items():
        # Calcolo della media delle domande
        if data['total_questions'] > 0:
            question_avg_score = data['question_total_score'] / data['total_questions']
        else:
            question_avg_score = 0.0

        # Calcolo della media delle risposte
        if data['total_answers'] > 0:
            answer_avg_score = data['answer_total_score'] / data['total_answers']
        else:
            answer_avg_score = 0.0
        
        print(f"Context ID: {context_id}")
        print(f"Numero di domande prodotte .: {data['total_questions']:9}")
        print("Domande")
        print(f"Score totale ...............: {data['question_total_score']:9.2f}")
        print(f"Score medio ................: {question_avg_score:9.2f}")
        print(f"Score massimo ..............: {data['question_max_score']:9.2f}")
        print(f"Score minimo ...............: {data['question_min_score']:9.2f}")
        
        print("\nRisposte")
        print(f"Numero di risposte prodotte .: {data['total_answers']:9}")
        print(f"Score totale ...............: {data['answer_total_score']:9.2f}")
        print(f"Score medio ................: {answer_avg_score:9.2f}")
        print(f"Score massimo ..............: {data['answer_max_score']:9.2f}")
        print(f"Score minimo ...............: {data['answer_min_score']:9.2f}")
        print()

def generate_questions(qg_pipeline, chunk_text, qg_params):
    start_time=time.time()
    generated_questions = qg_pipeline(                     # genera le domande
            f"genera domande: {chunk_text}",
            num_return_sequences=qg_params['num_return_sequences'],
            do_sample=qg_params['do_sample'],
            top_k=qg_params['top_k'],
            top_p=qg_params['top_p'],
            temperature=qg_params['temperature'], 
            max_length=qg_params['max_length'], 
            min_length=qg_params['min_length'], 
            repetition_penalty=qg_params['qa_repetition_penalty'], 
            length_penalty=qg_params['qa_length_penalty'], 
            early_stopping=qg_params['qa_early_stopping'],
            num_beams=qg_params['num_beams']
    )
    
    questions=[]
    
    
    for j,question in enumerate (generated_questions, start=1):
        # Calcola il punteggio per la domanda usando il contesto
        score = calcola_punteggio_singola_domanda(chunk_text, question['generated_text'])
        if score > 0:
            # Aggiungi la domanda e lo score al dizionario esistente
            questions.append ({'question':question['generated_text'],
                            'score': score})
            
        
    #questions contiene l'elenco delle domande generate con il proprio score
    
    #ordina in base allo score
    chunk_questions = sorted(questions, key=lambda x: x['score'], reverse=True)
    for j in range(len(chunk_questions)):
        #misura la distanza dalla domanda precedente per favorire le risposte maggiormente distanti
        if j > 0:
                delta_pp = min(1, Levenshtein.distance(chunk_questions[j]['question'], chunk_questions[j - 1]['question'])/len (chunk_questions[j]['question']))
        else:
                delta_pp = 1

        # Stampa il risultato per ogni domanda 
            
        # Aggiorna ogni entry con il nuovo score basato su score * delta_pp
        chunk_questions[j]['new_score'] = chunk_questions[j]['score'] * delta_pp
    
    #ordina in base a new_score
    chunk_questions = sorted(chunk_questions, key=lambda x: x['new_score'], reverse=True)
    tot_score=0
    for j in range (len(chunk_questions)):
        tot_score+=chunk_questions[j]['new_score']
    
    level = tot_score *.6
    num_questions =0
    cum_score=0.0
        
    while cum_score < level:
        cum_score+=chunk_questions[num_questions]['new_score']
        num_questions+=1

       
    #prende le prime num_questions domande in ordine decrescente di score
    chunk_questions = chunk_questions [:num_questions]
        
    
    #chunk_questions contiene l'elenco delle domande generate con il proprio score
    
    return chunk_questions
