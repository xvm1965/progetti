from .utils import estrai_testo_da_file, normalizza_testo, calculate_block_parameters, split_text
from .utils import create_directory_if_not_exists, check_model_files
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Documenti
from django.db import transaction, IntegrityError
import traceback
import json
import time
from .config import MAX_DIM_BLOCK, OVERLAP_PERCENTAGE, MODEL_NAME, NUM_TRAIN_EPOCH, SAVE_STEPS
from .config import PER_DEVICE_TRAIN_BATCH_SIZE, SAVE_TOTAL_LIMIT, OUTPUT_DIR, TRAINED_MODEL_DIR
from .config import DOMANDE
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
import pdb

import torch
from torch.utils.data import Dataset
import json
import os

# Funzione per preparare il dataset con contesto

# Definisci una classe CustomDataset che estende torch.utils.data.Dataset

# Funzione per il fine-tuning del modello

def preprocess_document (document):
    file_name = document.file_path()  # Usa il metodo file_path per ottenere il percorso completo
    text = estrai_testo_da_file(file_name)
    text=normalizza_testo(text)
    return text 
   


 
def train_model(request, document_ids=None):
    if document_ids:
        #print(f"chiamato dalla list view: document_ids provided: {document_ids}")
        
        # Filtra i documenti esistenti
        documents = Documenti.objects.filter(id__in=document_ids, trained=False)
        documents = Documenti.objects.filter(id__in=document_ids) # solo per il debugging 
        
        if not documents:
            return JsonResponse({'status': 'error', 'message': 'Nessun documento da addestrare'})
        
        total_documents = len(document_ids)
        documents_trained=0
        
        try:
            # Carica il modello esistente o uno nuovo se non esiste
            if os.path.exists(TRAINED_MODEL_DIR) and check_model_files(TRAINED_MODEL_DIR):
                # print ("carica il modello preaddestrato")
                pretrained_model = TRAINED_MODEL_DIR
            else:
                # print ("carica il nuovo modello")
                pretrained_model = MODEL_NAME

            model, tokenizer = load_model (pretrained_model)
            # model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
            # tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

            model_config = AutoConfig.from_pretrained(pretrained_model)
           
            
            #crea le directory se non esistono
            create_directory_if_not_exists(TRAINED_MODEL_DIR)
            create_directory_if_not_exists(OUTPUT_DIR)
            
            model_max_dim_block, overlap_size = calculate_block_parameters(model_config, 
                                                                           MAX_DIM_BLOCK, 
                                                                           OVERLAP_PERCENTAGE)
            #parametri per il training
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=NUM_TRAIN_EPOCH,
                per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                save_steps=SAVE_STEPS,
                save_total_limit=SAVE_TOTAL_LIMIT,
            )

            session_blocks = []
            session_labels = []
            session_attention_masks = []
            
            with transaction.atomic():    
                # Esegui il preprocessing e salva i documenti
                for document in documents:
                    print (f"documento {document.id}")
                    text=preprocess_document(document)
                    blocks, attention_masks, labels = split_text(text, model_max_dim_block, overlap_size, tokenizer)
                    
                    session_blocks +=blocks
                    session_labels +=labels
                    session_attention_masks +=attention_masks

                    documents_trained +=1
                    document.trained=True
                    document.save()
                
                # Verifica delle dimensioni
                # for i in range(len(session_blocks)):
                #     print(f"Blocco {i}: {len(session_blocks[i])} tokens, {len(session_attention_masks[i])} attention, {len(session_labels[i])} labels")

                # Crea il dataset
                dataset = CustomDataset(session_blocks, session_attention_masks, session_labels)

                # Crea il trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                )

               
                # Esegui l'addestramento
                print (f"addestramento ...", end=" ")
                start_time=time.time()
                trainer.train()
                print (f"concluso in {(time.time()-start_time):6.1f} secondi")

                # Salva il modello fine-tuned
                
                model.save_pretrained(TRAINED_MODEL_DIR)
                tokenizer.save_pretrained(TRAINED_MODEL_DIR)

            # prepara il messaggio di ritorno
            success_message = get_return_message (total_documents, documents_trained)
            return JsonResponse({'status': 'success', 'message': success_message})    


        except IntegrityError as ie:
            # Django gestirà automaticamente il rollback
            error_message = str(ie)
            return JsonResponse({'status': 'error', 'message': error_message})
        except Exception as e:
            # Django gestirà automaticamente il rollback
            error_message = str(e)
            traceback_message = traceback.format_exc()
            return JsonResponse({'status': 'error', 'message': error_message, 'traceback': traceback_message})
    else:
        return JsonResponse({'status': 'error', 'message': 'Nessun documento da addestrare'})




def get_return_message (total_documents, documents_trained):
    final_str = 'o'
    if total_documents >1 :
        final_str = 'i'
    success_message = f"Selezionat{final_str} {total_documents} document{final_str}, "
    if not documents_trained:
        success_message += "nessun documento addestrato"
    elif documents_trained == total_documents:
        success_message += f"addestrat{final_str}"
    else:
        if documents_trained > 1:
            final_str='i'
        else:
            final_str='o'
        success_message += f"{documents_trained} document{final_str} addestrat{final_str}"
    success_message += "!"
    return success_message
        
def load_tokenized_texts(json_files):
    tokenized_texts = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            tokenized_texts.extend(data)  # Assumendo che `blocks` sia una lista di liste di token
    return tokenized_texts

class CustomDataset(Dataset):
    def __init__(self, tokenized_texts, attention_masks, labels):
        self.tokenized_texts = tokenized_texts
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tokenized_texts[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }




from .config import CLASSE_DI_GESTIONE
import transformers

def load_model(model_dir):
    # Importa dinamicamente la classe del modello
    model_class = getattr(transformers, CLASSE_DI_GESTIONE)
    
    # Carica il modello e il tokenizer
    model = model_class.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer

def generate_answer(question, model, tokenizer):
    # Append the mask token to the question
    input_text = f"{question} {tokenizer.mask_token}"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Predict the masked token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    
    return predicted_token


