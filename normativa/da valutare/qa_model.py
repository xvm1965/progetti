# normativa/qa_model.py

from transformers import BertForQuestionAnswering, BertTokenizer
import torch

class QAModel:
    def __init__(self, model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def answer_question(self, question, text):
        max_length = 512  # Lunghezza massima per BERT
        inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = self.model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return answer
