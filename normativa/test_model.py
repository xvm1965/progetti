import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_documents(folder_path):
    # Caricare i file PDF dalla directory
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    
    # Creare un dizionario per memorizzare il numero di pagine per file
    file_page_counts = {}
    
    # Popolare il dizionario con il numero di pagine per file
    for doc in documents:
        source = doc.metadata['source']
        filename = os.path.basename(source) 
        if filename in file_page_counts:
            file_page_counts[filename] += 1
        else:
            file_page_counts[filename] = 1

    # Stampa l'elenco dei nomi dei file e il numero di pagine corrispondenti
    for i, (filename, page_count) in enumerate (file_page_counts.items(), start=1):
        print(f"File [{i:2}]: {filename}, Pagine caricate: {page_count}")
    input ("Return ...")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    # Suddividere i documenti in piccoli pezzi
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(documents)
    return all_splits

def embed_documents(doc_splits, embedding_model_name="sentence-transformers/all-mpnet-base-v2", persist_directory="chroma_db"):
    # Creare embeddings dei documenti
    model_kwargs = {"device": "cpu"}  # Usa "cuda" se hai una GPU
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)
    vectordb = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory=persist_directory)
    return vectordb

def initialize_model(model_name="dbmdz/italian-gpt2"):
    # Caricare il tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Caricare il modello ottimizzato per CPU
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to('cpu')

    # Impostare il modello in modalità di valutazione
    model.eval()
    
    return model, tokenizer

def chat_with_model(prompt, model, tokenizer, vectordb, max_length=100, temperature=0.7):
    # Recuperare informazioni rilevanti
    retriever = vectordb.as_retriever()
    relevant_docs = retriever.retrieve(prompt)
    context = " ".join([doc.page_content for doc in relevant_docs])

    # Creare input per il modello
    inputs = tokenizer(context + " " + prompt, return_tensors='pt')
    inputs = inputs.to('cpu')  # Assicurarsi che gli input siano su CPU

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

import time
def main():
    # Configurazione
    pdf_folder_path = r"C:\Users\Asus\Desktop\vprj\flowe\data\ragtest"
    model_name = "dbmdz/italian-gpt2"
    model_name = "dbmdz/gpt2-italian"  # Modello alternativo
    model_name = 'Musixmatch/umberto-commoncrawl-cased-v1'  # Modello alternativo

    # Caricare e processare i documenti
    start=time.time()
    print ("carica i documenti", end = " ")
    documents = load_documents(pdf_folder_path)
    print (f" {len(documents)} caricati in {(time.time()-start)} secondi")
    start=time.time()
    print ("splitting ...", end=" ")
    doc_splits = split_documents(documents)
    print (f"done in {(time.time()-start)} secondi")
    start=time.time()
    print ("embedding ...", end=" ")
    vectordb = embed_documents(doc_splits)
    print (f"done in {(time.time()-start)} secondi")

    # Caricare il modello
    start=time.time()
    print ("initializing model ...", end=" ")
    model, tokenizer = initialize_model(model_name)
    print (f"done in {(time.time()-start)} secondi")

    print("Chatbot in italiano. Scrivi 'esci' per terminare.")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() in ["esci", "exit"]:
            break
        response = chat_with_model(user_input, model, tokenizer, vectordb)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
