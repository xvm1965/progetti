import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from spacy.lang.it.stop_words import STOP_WORDS as it_stopwords
import spacy
from bs4 import BeautifulSoup

SEPARATORS = [' ', '.', ',', ';', '\n', '\t', '?', '!', ':']
REMOVE_PUNCTUATION = True
REMOVE_STOPWORDS = True
LEMMATIZER = True
EXPAND_CONTRACTIONS = True
REMOVE_NUMBERS = True
REMOVE_SPECIAL_CHARACTERS = False
SPELL_CORRECTION = True
LANGUAGE = 'it'
# Carica il modello italiano di spacy
nlp = spacy.load("it_core_news_sm")

# Funzione per leggere il contenuto del file PDF
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Funzione per generare l'input per il modello LLM
def create_input(prompt, context):
    return f"Domanda: {prompt}\nContesto: {context}"

# Funzione per suddividere il testo in blocchi di dimensioni gestibili
def split_text_into_chunks(text, tokenizer, prompt_length, max_length):
    effective_max_length = max_length - prompt_length
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), effective_max_length):
        chunk_tokens = tokens[i:i + effective_max_length]
        if len(chunk_tokens) < effective_max_length:
            chunk_tokens += [tokenizer.pad_token_id] * (effective_max_length - len(chunk_tokens))
        chunks.append(chunk_tokens)
    return chunks

def lemmatizza(testo):
    doc = nlp(testo)
    parole_lematizzate = [token.lemma_ for token in doc]
    return ' '.join(parole_lematizzate)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def expand_contractions(text):
    return text

def correct_spelling(text, lang=LANGUAGE):
    return text

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def togli_sommario(testo):
    pattern_sommario = re.compile(r'\b(sommario|indice)\b', re.IGNORECASE)
    match_sommario = pattern_sommario.search(testo)

    if match_sommario:
        start_pos = match_sommario.end()
        resto_testo = testo[start_pos:]
        prima_parola_match = re.search(r'\b(?!sommario|indice)\w+\b', resto_testo, re.IGNORECASE)
        if prima_parola_match:
            prima_parola = prima_parola_match.group(0)
            second_occurrence_match = re.search(r'\b' + re.escape(prima_parola) + r'\b', resto_testo[prima_parola_match.end():])
            if second_occurrence_match:
                second_occurrence_pos = start_pos + prima_parola_match.end() + second_occurrence_match.start()
                testo_senza_sommario = testo[:match_sommario.start()] + testo[second_occurrence_pos:]
                testo = testo_senza_sommario
    
    pattern_numero = re.compile(r'\b1\s+(\w+\b.*?)\b(?=\s|\d|[^\w\s])', re.IGNORECASE)
    pattern_numero = re.compile(r'\b1\s+(\w+(?:\s+\w+)*[a-zA-Z])(?=\s|\d|[^\w\s])', re.IGNORECASE)
    match_numero = pattern_numero.search(testo)
    
    if match_numero:
        prima_stringa = match_numero.group(0)
        start_pos = match_numero.end()
        second_occurrence_match = re.search(re.escape(prima_stringa), testo[start_pos:])
        if second_occurrence_match:
            second_occurrence_pos = start_pos + second_occurrence_match.start()
            testo_senza_sommario = testo[:match_numero.start()] + testo[second_occurrence_pos:]
            testo = testo_senza_sommario
    
    return testo

def normalizza_testo(testo=None):
    testo = re.sub(r'\s+', ' ', testo).strip()
    testo = testo.lower()
    testo = togli_sommario(testo)
    if REMOVE_STOPWORDS:
        testo = ' '.join([word for word in testo.split() if word not in it_stopwords])
    if REMOVE_PUNCTUATION:
        pass
    if EXPAND_CONTRACTIONS:
        testo = expand_contractions(testo)    
    if REMOVE_NUMBERS:
        testo = remove_numbers(testo)
    if REMOVE_SPECIAL_CHARACTERS:
        testo = remove_special_characters(testo)
    if SPELL_CORRECTION:
        testo = correct_spelling(testo)
    testo = remove_html(testo)
    testo = lemmatizza(testo)
    return testo

def main():
    max_length = 256
    file_path = r'C:\Users\Asus\Desktop\vprj\flowe\data\filedocs\07. Policy per la Gestione dei reclami, dei ricorsi e degli esposti a Banca d’Italia.pdf'
    document_text = normalizza_testo(read_pdf(file_path))
    
    model_name = 'Musixmatch/umberto-commoncrawl-cased-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    while True:
        user_question = input("Inserisci la tua domanda: ")
        
        if not user_question:
            print("Domanda non valida. Termino il programma.")
            break
        
        prompt = f"Domanda: {user_question}\nContesto: "
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_tokens)
        
        chunks = split_text_into_chunks(document_text, tokenizer, prompt_length, max_length)

        responses = []
        for i, chunk_tokens in enumerate(chunks):
            chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            input_text = create_input(user_question, chunk)
            
            inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
            
            input_ids = inputs['input_ids']
            input_length = input_ids.size(1)
            print(f"Lunghezza del blocco {i+1}: {input_length} token")
            
            if input_length > 512:
                print(f"Errore: La lunghezza del blocco {i+1} supera i 512 token. Riducendo la lunghezza del contesto.")
                continue
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(max_length/4),
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
            print(f"Risposta per il blocco {i+1}:")
            print(response)
            print("-" * 80)

        print("Tutte le risposte ottenute:")
        for resp in responses:
            print(resp)
            print("-" * 80)

if __name__ == "__main__":
    main()
