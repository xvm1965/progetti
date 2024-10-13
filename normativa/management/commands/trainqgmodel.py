import json
from pathlib import Path
from normativa.models import Domande
from normativa.config import TRAINED_QUESTION_GENERATOR, STANDARD_QUESTION_GENERATOR, CONTEXTS_FILENAME
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from django.core.management.base import BaseCommand
from sklearn.model_selection import train_test_split
import torch
import math

class Command(BaseCommand):
    help = "Addestra il modello di generazione delle domande"

    def handle(self, *args, **kwargs):
        train_qgmodel()
        self.stdout.write(self.style.SUCCESS('Modello addestrato e salvato con successo!'))


def costruisci_dataset_per_training_qgmodel():
    # Estrai tutte le domande dal database
    domande = Domande.objects.all()
    dataset = []
    
    # Carica l'intero file JSON che contiene tutti i documenti e i chunk
    json_file = Path(CONTEXTS_FILENAME)  # CONTEXTS_FILENAME è la variabile che contiene il percorso del file JSON
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Dizionario per una rapida ricerca degli chunks per document_id
    document_chunks = {}
    for document in data:
        document_id = str(document["document_id"])  # Converti l'ID in stringa se necessario
        document_chunks[document_id] = document.get("chunks", [])
    
    # Costruisci coppie contesto-domanda
    for domanda in domande:
        document_id = str(domanda.documento.id)  # Converto l'ID in stringa per fare il match nel JSON
        context_id = domanda.context_id
               
        # Verifica che il document_id esista nel dizionario e che l'indice context_id sia valido
        if document_id in document_chunks:
            chunks = document_chunks[document_id]
            if context_id < len(chunks):
                contesto = chunks[context_id]
                dataset.append({
                    "input_text": contesto,
                    "target_text": domanda.domanda,
                    "score": domanda.rating_domanda,
                    "document_id": document_id
                })

    return dataset


class QGTrainingDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        inputs = self.tokenizer(data["input_text"], truncation=True, padding="max_length", max_length=512)
        targets = self.tokenizer(data["target_text"], truncation=True, padding="max_length", max_length=512)

        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'labels': torch.tensor(targets['input_ids']),
            'scores': torch.tensor(data["score"], dtype=torch.float32)  # Include il rating_domanda
        }


def load_qgmodel():
    if TRAINED_QUESTION_GENERATOR.exists():
        model = AutoModelForSeq2SeqLM.from_pretrained(TRAINED_QUESTION_GENERATOR)
        tokenizer = AutoTokenizer.from_pretrained(TRAINED_QUESTION_GENERATOR)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(STANDARD_QUESTION_GENERATOR)
        tokenizer = AutoTokenizer.from_pretrained(STANDARD_QUESTION_GENERATOR)
    
    return model, tokenizer


def custom_loss_function(labels, logits, scores):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    # Se scores è None, gestiamo il caso della fase di valutazione
    if scores is None:
        print("\nFase di evaluation: scores non presente, calcolo della loss senza pesi.")
        return loss.mean()

    # Verifica le dimensioni
    print("\nFase di training: calcolo della loss con i pesi.")
    print(f"Dimensioni di loss: {loss.size()}")
    print(f"Dimensioni di scores: {scores.size()}")

    # Assicurati che 'scores' e 'loss' abbiano la stessa dimensione
    if scores.size(0) != loss.size(0):
        scores = scores.unsqueeze(1).expand(-1, loss.size(0) // scores.size(0))

    weights = 1 / (scores + 1e-8)  # Calcola i pesi

    # Verifica le dimensioni di weights
    print(f"Dimensioni di weights: {weights.size()}")

    weighted_loss = loss * weights.view_as(loss)
    
    return weighted_loss.mean()





class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = custom_loss_function
        self.custom_collator = custom_collator

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.custom_collator,
            shuffle=True
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        # Estrai labels e scores dai dati
        labels = inputs.get("labels")
        scores = inputs.get("scores")

        # Passa solo gli argomenti previsti al modello
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=labels
        )
        logits = outputs.get("logits")

        # Calcola la perdita personalizzata
        loss = custom_loss_function(labels, logits, scores)
        return (loss, outputs) if return_outputs else loss
      
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.is_training = False
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    def train(self, *args, **kwargs):
        self.is_training = True
        return super().train(*args, **kwargs)


def custom_collator(features):
    # Verifica se tutte le features contengono una chiave 'scores' con un valore valido
    all_have_scores = all('scores' in f and f['scores'] is not None and f['scores'].numel() > 0 for f in features)
    
    batch = {
        'input_ids': torch.stack([f['input_ids'].clone().detach() for f in features]),
        'attention_mask': torch.stack([f['attention_mask'].clone().detach() for f in features]),
        'labels': torch.stack([f['labels'].clone().detach() for f in features])
    }

    if all_have_scores:
        batch['scores'] = torch.stack([f['scores'].clone().detach() for f in features])
    else:
        print("Non tutte le features contengono un valore valido per 'scores'.")
    
    return batch



def train_qgmodel():
    dataset_di_training = costruisci_dataset_per_training_qgmodel()
    model, tokenizer = load_qgmodel()
    
    train_dataset, eval_dataset = train_test_split(dataset_di_training, test_size=0.2, random_state=42)

    # Contiamo quante istanze non hanno uno score definito (ad esempio, None o NaN)
    def conta_istanze_senza_score(dataset):
        return sum(1 for item in dataset if item.get("score") is None or math.isnan(item.get("score")))
    
    # # Contiamo per il dataset di training e quello di test
    # istanze_senza_score_train = conta_istanze_senza_score(train_dataset)
    # istanze_senza_score_eval = conta_istanze_senza_score(eval_dataset)
    
    # print(f"Istanze senza score nel dataset di training: {istanze_senza_score_train}")
    # print(f"Istanze senza score nel dataset di test: {istanze_senza_score_eval}")

    # input ("... premi un tasto")

    qg_train_dataset = QGTrainingDataset(train_dataset, tokenizer)
    qg_eval_dataset = QGTrainingDataset(eval_dataset, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        logging_dir='./logs',
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=qg_train_dataset,
        eval_dataset=qg_eval_dataset,
        tokenizer=tokenizer,
        data_collator=custom_collator
    )

    trainer.train()
    trainer.save_model(TRAINED_QUESTION_GENERATOR)


if __name__ == "__main__":
    train_qgmodel()
