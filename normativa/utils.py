import fitz  # PyMuPDF per PDF
import json
import nltk
import numpy as np
import os
import pdb
import random
import re
import spacy
import time
import torch
import unicodedata

from bs4 import BeautifulSoup
from .config import CONTEXTS_FILENAME
from .config import SEPARATORS, REMOVE_PUNCTUATION, REMOVE_STOPWORDS, EXPAND_CONTRACTIONS
from .config import REMOVE_NUMBERS, REMOVE_SPECIAL_CHARACTERS, SPELL_CORRECTION, LANGUAGE, LEMMATIZER
from .config import TRAINED_QUESTION_GENERATOR, STANDARD_ANSWER_GENERATOR, STANDARD_QUESTION_GENERATOR, TRAINED_ANSWER_GENERATOR, TRAINED_GENERATIVE_ANSWER_GENERATOR, STANDARD_GENERATIVE_ANSWER_GENERATOR
from datetime import datetime, timedelta
from django.conf import settings
from docx import Document
from spacy.lang.it.stop_words import STOP_WORDS as it_stopwords
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_distances



def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# Carica il modello italiano di spacy
nlp = spacy.load("it_core_news_sm")

def check_for_cycles(obj):
    """Verifica se l'assegnazione del processo padre crea un ciclo."""
    current = obj.padre
    ciclo = False
    while current is not None and not ciclo:
        ciclo = (current == obj)
        current = current.padre
    return ciclo

def hierarchy_error(obj):
    err_code = 0
    if obj.padre == obj:
        err_code = 1  # ("L'azienda non può controllare se stessa.")
    else:
        if obj.padre is None:
            if obj.__class__.objects.exclude(pk=obj.pk).filter(padre__isnull=True).exists():
                err_code = 2  # Non ha indicato il padre e c'è già un record radice
        else:
            if check_for_cycles(obj): err_code = 3  # L'assegnazione crea un ciclo
    return err_code     

def file_path(file_name):
    return Path(settings.DOCS_DIR) / file_name

def preprocess_document (document):
    # print ("entra in preprocess")
    file_name = document.file_path()  # Usa il metodo file_path per ottenere il percorso completo
    # print (f"preprocess - file name {file_name}")
    text = estrai_testo_da_file(file_name)
    # print (f"preprocess richiama normalizza testo")
    text=normalizza_testo(text)
    # print (f"preprocess - document.testo {document.testo [:50]}")
    return text 


def estrai_testo_da_file(file_name):
    file_name = str(file_name)  # Converti il Path in una stringa
    file_ext = file_name.split('.')[-1].lower()
    if file_ext == 'pdf':
        return estrazione_testo_da_pdf(file_path(file_name))
    elif file_ext == 'docx' or file_ext =='doc':
        return estrazione_testo_da_docx(file_path(file_name))
    return ''

# Funzione per estrarre il testo da un file PDF
def extract_text_from_pdf(file):
    text = estrazione_testo_da_pdf(file)
    text = re.sub(r'([.,;:\t\n])\1+', r'\1', text)
    text = re.sub(r'([.,;:\t\n])[.,;:\t\n]+', r' ', text)
    text = text.lower()
    
    return text

def split_text_into_sentences(text):
    sentences = text.split('. ')  # Dividi il testo sulle frasi basate su punto e spazio
    return [s.strip() for s in sentences if s.strip()]  # Rimuovi spazi vuoti e frasi vuote

def load_and_chunk_file(file, chunk_size, tokenizer, nlp_pipeline):
    chunks=[]
    if file.endswith((".pdf", ".doc", ".docx")):
        text = normalizza_testo(estrai_testo_da_file(file))
        sentences = split_text_into_sentences(text)
        current_chunk = []
        current_chunk_token_count = 0
        current_embedding = None
        for sentence in sentences:
            # Tokenizza la frase e ottieni la lunghezza in token
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False) 
            sentence_token_count = len(sentence_tokens)

            # Controllo per suddividere la frase se è più lunga di chunk_size
            if sentence_token_count > chunk_size:
            # Se il chunk corrente ha già delle frasi, lo aggiungiamo a 'chunks' prima di gestire la frase lunga
                if current_chunk:
                    chunks.append(' '.join(current_chunk)) #aggiunge il chubnk corrente all'elenco dei chunk
                    current_chunk = []                     #inizializza un nuovo chunk corrente
                    current_chunk_token_count = 0          #azzera il contatore dei token del chunk corrente
                
                # Suddividi la frase in chunk più piccoli
                for i in range(0, sentence_token_count, chunk_size):   #calcola la dimensione del blocco della frase corrente da inseire nel nuovo chunk
                    sub_tokens = sentence_tokens[i:i + chunk_size]     #copia i token del blocco in sub tokens
                    sub_sentence = tokenizer.decode(sub_tokens, skip_special_tokens=True) #trasforma il blocco di token in stringa
                    chunks.append(sub_sentence)  # Aggiungi ogni sottosegmento come un chunk separato

                # Vai alla prossima frase, senza fare ulteriori controlli su questa
                continue

            # Ottieni l'embedding per la frase
            sentence_embedding = torch.tensor(nlp_pipeline(sentence)[0])

            # Controlla se la frase si adatta nel chunk corrente o se deve iniziarne uno nuovo
            if current_chunk_token_count + sentence_token_count > chunk_size or (                       #se la dimensione del chunk corrente + quella della frase è maggiore del massimo
                current_embedding is not None and                                                       #oppure se l'emebdding del chunk corrente non è vuoto e
                torch.cosine_similarity(current_embedding.mean(dim=0).unsqueeze(0), sentence_embedding.mean(dim=0).unsqueeze(0)).item() < 0.75 #la distanza coseno tra chunk corrente e nuova frase è < 0.75
            ):
                # Salva il chunk corrente
                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                current_chunk = [sentence] #inizializza un nuovo chunk corrente con la frase
                current_chunk_token_count = sentence_token_count #inizializza il contatore dei token del chunk corrente
                current_embedding = sentence_embedding  # Imposta l'embedding della nuova frase
            else:
                # il chunk corrente e la frase corrente hanno features simili: aggiunge la frase al chunk corrente
                current_chunk.append(sentence) #aggiunge la frase al chunk corrente
                current_chunk_token_count += sentence_token_count #incrementa il contatore di numero di token
                
                # Se current_embedding è None (quindi è la prima iterazione), assegniamo direttamente sentence_embedding
                if current_embedding is None:
                    current_embedding = sentence_embedding
                else:
                    # Altrimenti concateniamo gli embedding
                    current_embedding = torch.cat((current_embedding, sentence_embedding), dim=0)

        # Aggiungi l'ultimo chunk se esiste
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


def convert_seconds(seconds):
    # Calcola le ore
    seconds=int(seconds)
    hours = int (seconds /3600)
    rem_seconds = seconds - hours*3600
    # Calcola i minuti rimanenti dopo aver tolto le ore
    minutes = int(rem_seconds / 60)
    rem_seconds -= minutes*60
    
    
    # Restituisce i valori in una struttura
    time_structure = {
        'hours': hours,
        'minutes': minutes,
        'seconds': rem_seconds
    }
    
    return time_structure

def format_time_structure(time_structure):
    hours =   time_structure.get('hours', 0)
    minutes = time_structure.get('minutes', 0)
    seconds = time_structure.get('seconds', 0)
    
    # Formatta i valori come hh:mm:ss
    time_string = f"{hours:02}:{minutes:02}:{seconds:02}"
    
    return time_string




def get_current_time_plus_seconds(seconds):
    # Ottieni l'ora corrente
    current_time = datetime.now()
    # Calcola il nuovo tempo aggiungendo i secondi
    new_time = current_time + timedelta(seconds=seconds)
    # Restituisci il nuovo tempo formattato come hh:mm:ss
    time_string = new_time.strftime("%H:%M:%S")
    
    return time_string



def estrazione_testo_da_pdf(file):
    doc = fitz.open(file)
    text = ''
    for page in doc:
        text += page.get_text()
    return text

def estrazione_testo_da_docx(file):
    doc = Document(file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def update_text_from_file(self, request, queryset):
    total_documents = queryset.count()
    count = 0
    for documento in queryset:
        count += 1
        try:
            file_path = documento.file_path()  # Ottieni il percorso completo del file
            with fitz.open(file_path) as doc:
                text = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
                documento.testo = text
                documento.save()

            # Mostra un messaggio di progresso
            self.message_user(request, f"Lettura completata per il documento '{documento.descrizione}' ({count}/{total_documents})")

        except Exception as e:
            messages.error(request, f"Errore durante la lettura del documento '{documento.descrizione}': {str(e)}")

def extract_sentence_from_block(text=None, block_size=0, random_start=False):
    sentence = ""
    if text and block_size:
        context_length = len(text)
        if random_start and block_size <= context_length:
            start_index = random.randint(0, context_length - block_size)
        else:
            start_index = 0
        last_index_available = context_length - 1
        end_index = min(last_index_available, start_index + block_size)
        while text[start_index] not in SEPARATORS and start_index > 0:
            start_index -= 1
        while text[start_index] in SEPARATORS and start_index < last_index_available:
            start_index += 1
        while text[end_index] in SEPARATORS and end_index > start_index:
            end_index -= 1
        while text[end_index] not in SEPARATORS and end_index < last_index_available:
            end_index += 1
        sentence = text[start_index:end_index]
    return sentence


def split_text(text, block_size, overlap_size, tokenizer):
    words = text.split()  # Suddividi il testo in parole
    blocks = []
    attention_masks = []
    labels = []
    start = 0
    end = 0
    #print (f"tokenizzazione del documento di {len (words)} parole")
    num_blk = 1
    while end < len(words):
        end = min(start + block_size, len(words))
        block_text = ' '.join(words[start:end])
        start_time = time.time()
        #print (f"blocco {num_blk:4}", end=" ")
        # Tokenizza il blocco
        tokens = tokenizer.encode(block_text, add_special_tokens=False)
         # Se il numero di token è superiore a block_size, riduci il blocco
        while len(tokens) > block_size:
            end -= 1
            block_text = ' '.join(words[start:end])
            tokens = tokenizer.encode(block_text, add_special_tokens=False)
        
        attention_mask = [1] * len(tokens) + [0] * (block_size - len(tokens))
        #print (f" di {(end-start):5} parole ", end=" ")
        # Se il numero di token è inferiore a block_size, aggiungi padding
        
        if len(tokens) < block_size:
            tokens += [tokenizer.pad_token_id] * (block_size - len(tokens))
        
            
        #print (f"finito in {  (time.time()-start_time):5.2f} secondi - tokens: {len(tokens)} - attention: {len(attention_mask)}")
        blocks.append(tokens)
        attention_masks.append(attention_mask)
        labels.append(tokens)
        start = end - overlap_size  # Sovrapposizione tra i blocchi
        num_blk+=1
    # Se l'ultimo blocco è inferiore a block_size, aggiungi padding
    if len(blocks[-1]) < block_size:
        padding_length = block_size - len(blocks[-1])
        blocks[-1] += [tokenizer.pad_token_id] * padding_length
        labels[-1] += [tokenizer.pad_token_id] * padding_length
        
    
    return blocks, attention_masks, labels



def lemmatizza(testo):
    #print(f"sono in lemmatizza testo, lunghezza del testo {len(testo)}", end=" ")
    doc = nlp(testo)
    parole = [token.text for token in doc]
    num_parole = len(parole)
    termini_distinti_iniziali = len(set(parole))
    #print(f"{num_parole} parole individuate, {termini_distinti_iniziali} termini distinti all'inizio")
    parole_lematizzate = []
    diversi = []
    blocco_dimensione = 100
    for i in range(0, num_parole, blocco_dimensione):
        lemmatizzati_blocco = [token.lemma_ for token in doc[i:i + blocco_dimensione]]
        parole_lematizzate.extend(lemmatizzati_blocco)
    testo_lemmatizzato = ' '.join(parole_lematizzate)
    termini_distinti_finali = len(set(parole_lematizzate))
    #print(f"Lunghezza finale del testo {len(testo_lemmatizzato)}. Termini distinti finali: {termini_distinti_finali}")
    return testo_lemmatizzato

def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def expand_contractions(text):
    return text

def advanced_tokenization(text, lang=LANGUAGE):
    doc = nlp(text)
    return [token.text for token in doc]

def correct_spelling(text, lang=LANGUAGE):
    return text

def remove_urls_emails(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    return text

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def togli_sommario(testo):
    pattern_sommario = re.compile(r'\b(sommario|indice)\b', re.IGNORECASE)
    match_sommario = pattern_sommario.search(testo)
    testo_originale = testo
    #print("\nEliminazione del sommario cerca il termine sommario o indice....")

    if match_sommario:
        start_pos = match_sommario.end()
        resto_testo = testo[start_pos:]
        prima_parola_match = re.search(r'\b(?!sommario|indice)\w+\b', resto_testo, re.IGNORECASE)
        # print(f"ha trovato il termine {testo[match_sommario.start():match_sommario.end()]} alla posizione {start_pos}")
        # print("cerca la prima parola che segue")

        if prima_parola_match:
            prima_parola = prima_parola_match.group(0)
            # print(f"(il termine che segue è {prima_parola} ... cerca una seconda occorrenza del termine ")
            second_occurrence_match = re.search(r'\b' + re.escape(prima_parola) + r'\b', resto_testo[prima_parola_match.end():])

            if second_occurrence_match:
                second_occurrence_pos = start_pos + prima_parola_match.end() + second_occurrence_match.start()
                # print(f"ha trovato la seconda occorrenza di {prima_parola} alla posizione {second_occurrence_pos}")
                testo_senza_sommario = testo[:match_sommario.start()] + testo[second_occurrence_pos:]
                testo = testo_senza_sommario
            # else:
                # print(f" ... non trovata una seconda occorrenza di {prima_parola}")
        # else:
            # print("Nessun termine trovato dopo")
        
        # # Rimozione delle righe che iniziano o finiscono con un numero
        # lines = testo.split('\n')
        # testo = '\n'.join(line for line in lines if not re.match(r'^\d+.*|\d+$', line))
    # else:
        # print("Non ha trovato il termine sommario o indice")

    # print("Cerca il primo termine preceduto da 1 ")
    pattern_numero = re.compile(r'\b1\s+(\w+\b.*?)\b(?=\s|\d|[^\w\s])', re.IGNORECASE)
    pattern_numero = re.compile(r'\b1\s+(\w+(?:\s+\w+)*[a-zA-Z])(?=\s|\d|[^\w\s])', re.IGNORECASE)
    match_numero = pattern_numero.search(testo)
    
    if match_numero:
        prima_stringa = match_numero.group(0)
        start_pos = match_numero.end()
        # print(f"ha trovato il termine {testo[match_numero.start():match_numero.end()]} alla posizione {start_pos}")
        # print("cerca una seconda occorrenza ")
        second_occurrence_match = re.search(re.escape(prima_stringa), testo[start_pos:])
        
        if second_occurrence_match:
            second_occurrence_pos = start_pos + second_occurrence_match.start()
            # print(f" ha trovato una seconda occorrenza alla posizione {second_occurrence_pos}")
            testo_senza_sommario = testo[:match_numero.start()] + testo[second_occurrence_pos:]
            testo = testo_senza_sommario
        # else:
            # print(f"non trova una seconda occorrenza del termine {testo[match_numero.start():match_numero.end()]}")
    # else:
        # print(" .... non trovato!")

    return testo

# Definisci la funzione per sostituire le sequenze di caratteri uguali
pattern_uguali = '|'.join(re.escape(char) + '{2,}' for char in SEPARATORS)
def replace_equal_sequences(text):
    def replace_match(match):
        return match.group(0)[0]
    return re.sub(pattern_uguali, replace_match, text)

# Definisci la funzione per sostituire le sequenze di caratteri diversi con uno spazio
pattern_diversi = '[' + re.escape(''.join(SEPARATORS)) + ']+'
def replace_different_sequences_with_space(text):
    return re.sub(pattern_diversi, ' ', text)

def replace_separators(text):
    text = replace_equal_sequences(text)
    text = replace_different_sequences_with_space(text)
    return text


def normalizza_testo(testo=None):
    # print ("normalizza testo ....")
    if testo is None or not isinstance(testo, str):
        return ""
    try:

        # print (f"elimina gli spazi inutili , da {len(testo)} caratteri a ", end=" ")
        testo = re.sub(r'\s+', ' ', testo).strip()
        # print (f"{len(testo) } caratteri")

        # print (f"elimina i separatori , da {len(testo)} caratteri a ", end=" ")
        testo=replace_separators (testo)
        # print (f"{len(testo) } caratteri")
        
        #tutto minuscolo
        testo = testo.lower()
        
        # print (f"elimina il sommario, da {len(testo)} caratteri a ", end=" ")
        testo = togli_sommario(testo)
        # print (f"{len(testo) } caratteri")

       
        
        # print (f"elimina le stopwords da {len(testo)} caratteri a ", end=" ")
        if REMOVE_STOPWORDS:
            testo = ' '.join([word for word in testo.split() if word not in it_stopwords])
        # print (f"{len(testo) } caratteri")
 
        # print (f"elimina la punteggiatura , da {len(testo)} caratteri a ", end=" ")
        if REMOVE_PUNCTUATION:
            #testo = re.sub(r'[^\w\s]', '', testo)
            # Crea una stringa di pattern dai separatori per sequenze uguali
            # Funzione principale che combina le due sostituzioni
            

            # pattern = '[' + re.escape(''.join(SEPARATORS)) + ']+'
            # testo = re.sub(pattern, ' ', testo)
            pass
        # print (f"{len(testo) } caratteri")
        
       

        #testo = ''.join(c for c in unicodedata.normalize('NFD', testo) if unicodedata.category(c) != 'Mn')
        if EXPAND_CONTRACTIONS:
            testo = expand_contractions(testo)
        # print (f"elimina i numeri , da {len(testo)} caratteri a ", end=" ")
        if REMOVE_NUMBERS:
            testo = remove_numbers(testo)
        # print (f"{len(testo) } caratteri")
        # print (f"elimina i caratteri speciali, da {len(testo)} caratteri a ", end=" ")
        if REMOVE_SPECIAL_CHARACTERS:
            testo = remove_special_characters(testo)
        # print (f"{len(testo) } caratteri")
        # print (f"elimina urls, da {len(testo)} caratteri a ", end=" ")
        testo = remove_urls_emails(testo)
        # print (f"{len(testo) } caratteri")
        if SPELL_CORRECTION:
            testo = correct_spelling(testo)
        # print (f"elimina html, da {len(testo)} caratteri a ", end=" ")
        testo = remove_html(testo)
        # print (f"{len(testo) } caratteri")
        if LEMMATIZER: testo = lemmatizza(testo)
    except Exception as e:
        print(f"Errore nella normalizzazione del testo: {e}")
    return testo



def calculate_block_parameters(model_config, max_dim_block, overlap_percentage):
    # Dimensione massima del blocco in token
    model_max_dim_block = model_config.max_position_embeddings
    
    block_size = int(model_max_dim_block * max_dim_block)
    
    # Dimensione dell'overlap
    max_overlap = min(0.5 * model_max_dim_block, 0.5 * block_size)
    overlap_size = int(overlap_percentage * block_size)
    overlap_size = min(overlap_size, max_overlap)
    
    return block_size, overlap_size

def create_directory_if_not_exists(directory):
    """
    Crea la directory se non esiste.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        # print(f"Directory creata: {directory}")
    # else:
    #     print(f"Directory già esistente: {directory}")

def check_model_files(directory):
    required_files = ['config.json', 
                      'model.safetensors', 
                      'tokenizer.json', 
                      'sentencepiece.bpe.model',
                      'special_tokens_map.json',
                      'tokenizer_config.json']
            
    return all(os.path.exists(os.path.join(directory, file)) for file in required_files)






def get_synonyms(word):
    """Restituisce un set di sinonimi per una data parola usando WordNet."""
    synonyms = set()
    for synset in wn.synsets(word, lang='ita'):
        for lemma in synset.lemmas('ita'):
            synonyms.add(lemma.name())
    return synonyms

def is_keyword(word):
    """Determina se una parola è una keyword (non è una stop word)."""
    stop_words = set(stopwords.words('italian'))
    return word.lower() not in stop_words

def calculate_similarity(question, answer):
    """Calcola la similarità coseno tra la domanda e la risposta usando TF-IDF."""
    vectorizer = TfidfVectorizer().fit([question, answer])
    vectors = vectorizer.transform([question, answer])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity

def calculate_answer_score(question, context, answer, model_score=1):
    """
    Valuta la qualità di una risposta generata in base ai criteri di rilevanza, completezza, accuratezza, chiarezza,
    concisione e originalità, e restituisce un punteggio complessivo.

    Args:
    - question (str): La domanda a cui la risposta si riferisce.
    - context (str): Il contesto da cui la risposta dovrebbe derivare.
    - answer (str): La risposta generata dal modello.
    - model_score (float): Il punteggio assegnato dal modello alla risposta.

    Returns:
    - dict: Un dizionario contenente il punteggio per ciascun criterio e il punteggio totale.
    """

    # Definizione delle categorie di valutazione
    def evaluate_relevance(context, answer):
        # Simuliamo una valutazione della rilevanza, ad esempio con una semplice verifica di overlap o keyword matching
        return 5 if answer in context else 3 if any(word in context for word in answer.split()) else 1

    def evaluate_completeness(question, answer):
        # Verifica se la risposta copre la domanda. Questo richiede inferenza sul contenuto (può essere esteso)
        return 5 if len(answer) > len(question) else 3 if len(answer) > 0.5 * len(question) else 1

    def evaluate_accuracy(context, answer):
        # La valutazione dell'accuratezza può essere simulata verificando se le affermazioni nella risposta sono nel contesto
        return 5 if answer in context else 3 if any(word in context for word in answer.split()) else 1

    def evaluate_clarity(answer):
        # Valutazione soggettiva della chiarezza (ad esempio in base alla lunghezza delle frasi o alla grammatica)
        return 5 if len(answer.split()) < 30 and answer[0].isupper() and answer[-1] in '.!?' else 3

    def evaluate_concision(answer):
        # Valuta se la risposta è concisa o prolissa rispetto alla domanda
        return 5 if len(answer.split()) <= len(question.split()) * 2 else 3

    def evaluate_originality(answer, context):
        # Verifica se la risposta non è una semplice ripetizione del contesto
        return 5 if answer != context else 1

    # Applica i criteri di valutazione
    relevance_score = evaluate_relevance(context, answer)
    completeness_score = evaluate_completeness(question, answer)
    accuracy_score = evaluate_accuracy(context, answer)
    clarity_score = evaluate_clarity(answer)
    concision_score = evaluate_concision(answer)
    originality_score = evaluate_originality(answer, context)

    # Calcolo del punteggio finale come media ponderata (puoi cambiare i pesi se preferisci)
    final_score = (
        0.3 * relevance_score +
        0.2 * completeness_score +
        0.2 * accuracy_score +
        0.1 * clarity_score +
        0.1 * concision_score +
        0.1 * originality_score
    )

    # Considera lo scoring assegnato dal modello come fattore moltiplicativo (es. 0.5-1 per moderare il punteggio finale)
    final_score = final_score * model_score

    # Ritorna il punteggio di ciascun criterio e il punteggio totale
    return {
        'relevance_score': relevance_score,
        'completeness_score': completeness_score,
        'accuracy_score': accuracy_score,
        'clarity_score': clarity_score,
        'concision_score': concision_score,
        'originality_score': originality_score,
        'final_score': round(final_score, 2)
    }


# Carica il modello di lingua di spaCy 
nlp = spacy.load('it_core_news_sm')  

# Carica le stopwords
stop_words = set(stopwords.words('italian'))  # Puoi cambiare la lingua se necessario

# Carica il modello NLP
nlp = spacy.load("it_core_news_sm")

def calcola_punteggio_singola_domanda(context, question):
    """
    Assegna un punteggio a una singola domanda rispetto al contesto.
    """
    
    # 1. Penalizzazione per ripetizioni
    words = question.lower().split()
    word_frequencies = {}
    penalty = 1.0

    # Calcolo della penalizzazione per ripetizioni consecutive e frequenze
    previous_word = None
    for word in words:
        if word == previous_word:
            return 0.0
        if word not in nlp.Defaults.stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
            previous_word = word
    
    # Penalizzazione geometrica per ripetizioni non consecutive
    for word, freq in word_frequencies.items():
        penalty *= 1 / (2 ** (freq - 1))  # Penalizzazione geometrica
 
    # 2. Completezza e chiarezza (basata sulla lunghezza e struttura grammaticale)
    doc = nlp(question)
    total_tokens = len(doc)
   
    # Verifica se il documento contiene solo stop words
    if total_tokens == 0 or all([token.is_stop for token in doc]):
        return 0.0
    
    # 3. Rilevanza al contesto (basata su TF-IDF e similarità coseno)
    documents = [context, question]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    relevance_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
       
    # Calcola la percentuale di token non rilevanti
    words = [token.text.lower() for token in doc]
    context_terms = set(context.lower().split())
    relevant_tokens = sum([1 for word in words if word in context_terms])
    not_relevant_tokens = total_tokens - relevant_tokens

    non_relevant_percentage = not_relevant_tokens / total_tokens if total_tokens > 0 else 1
    clarity_score = 1.0 - non_relevant_percentage
    
    # 4. Coerenza linguistica (basata sulla complessità della struttura della domanda)
    # Calcolo del numero di token unknown non rilevanti
    num_unknown_non_relevant = sum(1 for token in doc if token.dep_ == 'unk' and token.label_ == '')
    grammatical_score = max(0, 1.0 - 0.1 * num_unknown_non_relevant)
    
      
    # Punteggio complessivo (media ponderata dei punteggi con l'incremento)
    overall_score = ((0.5 * relevance_score) + (0.3 * clarity_score) + (0.2 * grammatical_score)) * penalty
     
    # 5. Incremento del punteggio basato sulle entità rilevanti uniche
    unique_entities = set(ent.text for ent in doc.ents if ent.label_ != '')
    entity_bonus = min(1, len(unique_entities) * 0.2)  # Bonus proporzionale al numero di entità uniche (ad esempio, 5% per entità)
    
    overall_score *= (1+entity_bonus)  # Aggiungi il bonus basato sulle entità uniche

    return overall_score


def calcola_punteggi_domande(context, questions):
    """
    Assegna un punteggio a ogni domanda in base a vari criteri dati il contesto.

    Args:
    - context (str): Il contesto di riferimento.
    - questions (list of str): Una lista di domande generate.

    Returns:
    - sorted_scores (list of dict): Lista di dizionari con domanda e score, ordinata in ordine decrescente di score.
    """

    

    # Calcola lo score per ciascuna domanda
    scores = [
        {"question": question, "score": calcola_punteggio_singola_domanda(context, question)}
        for question in questions
    ]

    # Ordina le domande in base allo score in ordine decrescente
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

    return sorted_scores



def load_agmodel():
    if TRAINED_ANSWER_GENERATOR.exists():
        model = AutoModelForQuestionAnswering.from_pretrained(TRAINED_ANSWER_GENERATOR)
        tokenizer = AutoTokenizer.from_pretrained(TRAINED_ANSWER_GENERATOR)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(STANDARD_ANSWER_GENERATOR)
        tokenizer = AutoTokenizer.from_pretrained(STANDARD_ANSWER_GENERATOR)
    return model, tokenizer

def load_generative_agmodel():
    
    if TRAINED_GENERATIVE_ANSWER_GENERATOR.exists():
        
        model = AutoModelForSeq2SeqLM.from_pretrained(TRAINED_GENERATIVE_ANSWER_GENERATOR)
        tokenizer = AutoTokenizer.from_pretrained(TRAINED_GENERATIVE_ANSWER_GENERATOR)
    else:
        
        model = AutoModelForSeq2SeqLM.from_pretrained(STANDARD_GENERATIVE_ANSWER_GENERATOR)
        tokenizer = AutoTokenizer.from_pretrained(STANDARD_GENERATIVE_ANSWER_GENERATOR)
    
    return model, tokenizer

def load_qgmodel():
    if TRAINED_QUESTION_GENERATOR.exists():
        model = AutoModelForSeq2SeqLM.from_pretrained(TRAINED_QUESTION_GENERATOR)
        tokenizer = AutoTokenizer.from_pretrained(TRAINED_QUESTION_GENERATOR)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(STANDARD_QUESTION_GENERATOR)
        tokenizer = AutoTokenizer.from_pretrained(STANDARD_QUESTION_GENERATOR)
    
    return model, tokenizer

def load_chunkmodel():
    chunk_model = AutoModel.from_pretrained("bert-base-uncased") #modello per la generazione dei chunks
    chunk_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return chunk_model, chunk_tokenizer





def costruisci_dataset_per_agmodel(documents):
    # Inizializza il dataset
    dataset = {}
    
    # Carica il file JSON che contiene tutti i documenti e i chunk
    json_file = Path(CONTEXTS_FILENAME)  # CONTEXTS_FILENAME è la variabile che contiene il percorso del file JSON
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Crea un set di document_id dei documenti in 'documents' per un confronto rapido
    document_ids = set(str(doc_id) for doc_id in documents)

    print (f"\ndocument ids")
    for d in document_ids:
        print (d)
    
    # Itera sui documenti nel file JSON
    print (f"\nloop su document")
    for document in data:
        document_id = str(document["document_id"])
        print (f"document id: {document_id}")
        
        # Se il document_id è tra quelli che ci interessano, aggiungi i chunk al dataset
        if document_id in document_ids:
            dataset[document_id] = document.get("chunks", [])
            print (f"aggiunti {len(dataset[document_id])} chunks, associati al documento {document_id}")
        else:
            print (f"nessun chunk associato al documento {document_id}")
    

    return dataset
