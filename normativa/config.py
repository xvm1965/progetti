
from pathlib import Path


# Configurazione delle variabili globali

import os
from pathlib import Path
from flowe.settings import DATA_DIR, IS_WINDOWS


# Definisci la funzione per sostituire il secondo segmento con 'temp' o 'TEMP'
def build_chat_temp_dir(data_dir):
    parts = list(data_dir.parts)  # Ottieni le parti del percorso
    if len(parts) > 1:  # Controlla che ci siano almeno due parti
        parts[len(parts)-1] = "temp" if IS_WINDOWS else "TEMP"  # Sostituisci il secondo segmento
    return Path(*parts)  # Ricostruisci il percorso

# Imposta la variabile CHAT_TEMP_DIR
CHAT_TEMP_DIR = build_chat_temp_dir(DATA_DIR)


MAX_DIM_BLOCK = 0.8  # Percentuale della dimensione massima del blocco
OVERLAP_PERCENTAGE = 0.1  # Percentuale di overlap
TOKENIZED_DIR = 'tokenized_documents'  # Directory di output

#parametri per il preprocessing
SEPARATORS = [' ', '.', ',', ';', '\n', '\t', '?', '!', ':']
REMOVE_PUNCTUATION = True
REMOVE_STOPWORDS = False
LEMMATIZER = False
EXPAND_CONTRACTIONS = True
REMOVE_NUMBERS=False
REMOVE_SPECIAL_CHARACTERS=False
SPELL_CORRECTION=True
LANGUAGE='it'

#parametri per l'addestramento
NUM_TRAIN_EPOCH=3
PER_DEVICE_TRAIN_BATCH_SIZE=4 #era 19
SAVE_STEPS=10000
SAVE_TOTAL_LIMIT=2
OUTPUT_DIR='./results'

CLASSE_DI_GESTIONE = 'AutoModelForCausalLM'

CONTEXTS_FILENAME = 'chunksandembeddings.json' #file dei chunks e degli embeddings
STANDARD_QUESTION_GENERATOR="gsarti/it5-base-question-generation" #modello preaddestrato per la generazione delle domande
TRAINED_QUESTION_GENERATOR = Path('.') / 'trained_qgm'
TRAINED_ANSWER_GENERATOR = Path('.') / 'trained_agm'
TRAINED_GENERATIVE_ANSWER_GENERATOR = Path('.') / 'trained_generative_agm'
QUESTION_GENERATOR_MODELS = ["gsarti/it5-base-question-generation", 
                             "vocabtrimmer/mbart-large-cc25-trimmed-it-itquad-qg",
                             "lmqg/mt5-small-itquad-qg"]
STANDARD_ANSWER_GENERATOR = "anakin87/electra-italian-xxl-cased-squad-it" #modello preaddestrato per la generazione delle risposte
STANDARD_GENERATIVE_ANSWER_GENERATOR="google/flan-t5-small"
ANSWER_GENERATOR_MODEL=STANDARD_ANSWER_GENERATOR
ANSWER_GENERATOR_MODELS = [ "osiria/deberta-italian-question-answering", 
                            "timpal0l/mdeberta-v3-base-squad2", 
                            "anakin87/electra-italian-xxl-cased-squad-it",
                            "luigisaetta/squad_it_xxl_cased_hub1"]
PARAPHRASE_GENERATOR_MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
QA_TRAINED_MODEL='./trained_model' #modello addestrato personalizzato
QA_MODEL_MANAGEMENT = 'transformers.DebertaV2ForQuestionAnswering'  # Percorso completo per il modello
QA_TOKENIZER = 'transformers.DebertaV2TokenizerFast'  # Percorso completo per il tokenizer
CHUNKS_PER_SAMPLE=5 #numero dei contesti che seleziona per cercare la risposta migliore

QA_QUERY_PER_CHUNKS=10 #domande generate per ogni chunk
QA_DO_SAMPLE = True # Se impostato su True, la generazione delle domande avviene attraverso campionamento casuale, altrimenti avviene attraverso una ricerca deterministica del token più probabile (greedy search)
QA_TOP_K = 500 #Limita il campionamento ai k token più probabili nel passaggio di generazione
QA_TOP_P = 0.98 # Limita il campionamento ai token che insieme rappresentano una probabilità cumulativa p.
QA_TEMPERATURE = 0.37 #Influenza la "creatività" della generazione. Valori più bassi rendono il modello più conservatore, preferendo token più probabili. Valori più alti rendono il modello più creativo
QA_MAX_LENGTH = 100 #lunghezza massima della domanda generata
QA_MIN_LENGTH = 5  #lunghezza minima della domanda generata
QA_REPETITION_PENALTY = 1.4 #Penalizza la ripetizione di frasi o parole nella domanda generata. Un valore più alto può ridurre la ripetizione, migliorando la qualità del testo generato.
QA_LENGTH_PENALTY = 0.75 #Penalizza o premia la lunghezza della generazione rispetto al contesto fornito. Valori maggiori di 1.0 rendono la generazione più breve, mentre valori minori di 1.0 possono portare a domande più lunghe
QA_EARLY_STOPPING = False # Se impostato su True, interrompe la generazione quando viene raggiunta una sequenza completa.può evitare la generazione di domande che continuano senza necessità, ma può anche interrompere prematuramente domande che potrebbero avere una coda importante
#QA_NUM_BEAMS=QA_QUERY_PER_CHUNKS #specifica il numero di sequenze candidate che vengono considerate a ogni passo del processo di generazione. Più alto è il valore, più sequenze vengono esplorate dvee essere al più il numero di domande generate per ogni chunk

QA_FILENAME = 'qa_results.json'

# config.py
DOCUMENT_INDEX = 'document_index.faiss'


STOPWORDS_LIST = [
    'a', 'di', 'da', 'in', 'su', 'per', 'con', 'tra', 'fra', 'del', 'delle', 'degli', 'dal', 'dagli', 'dalle',
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'una', 'uno', "l'", "dell'", "all'", "dei", "dai", "alla", "allo", "agli",
    # Aggiungi altre stopwords se necessario
]

