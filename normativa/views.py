from django.shortcuts import render

# Create your views here.
# In yourapp/views.py
from django.shortcuts import get_object_or_404
from django.http import FileResponse, Http404, HttpResponse, HttpResponseBadRequest
from .models import Documenti
from .trainmodel import costruisci_dataset_per_training_qgmodel
from .utils import clear_screen, costruisci_dataset_per_agmodel


def download_document(request, document_id):
    document = get_object_or_404(Documenti, id=document_id)
    file_path = document.file_path()
    
    if file_path.exists():
        response = FileResponse(open(file_path, 'rb'), as_attachment=True)
        return response
    else:
        raise Http404("File not found")
    
    print("View 'download_document' called successfully.")

# In normativa/views.py
from django.views.generic import ListView


class DocumentListView(ListView):
    model = Documenti  # Specifica il modello su cui basare la ListView
    template_name = 'normativa/document_list.html'  # Specifica il nome del template da utilizzare
    context_object_name = 'documents'  # Specifica il nome del contesto per gli oggetti nella ListView



from django.http import JsonResponse
from .models import UnitaOrganizzative


def get_unita_organizzative(request):
    legal_entity_id = request.GET.get('legal_entity_id')
    units = UnitaOrganizzative.objects.filter(LegalEntity_id=legal_entity_id).values('id', 'descrizione')
    return JsonResponse(list(units), safe=False)



# normativa/views.py


from .models import Documenti
from django.views.generic import TemplateView



# views.py

from django.views.decorators.csrf import csrf_exempt
import json

training_status = {
    'current_document': '',
    'current_phase': '',
    'progress': 0,
    'total_documents': 0,
    'current_index': 0,
}

@csrf_exempt
def get_training_status(request):
    return JsonResponse(training_status)




class ChatAdminView(TemplateView):
    pass



class AddestraAdminView(TemplateView):
    # Esegui la funzione per ottenere il dataset
    pass
   
    

class SetupAdminView():
    pass

# views.py

from .config import QA_TRAINED_MODEL
from .utils import check_model_files

import json

def generate_answer_view(request):
    if request.method == 'POST':
        try:
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)
            print("Body data:", body_data)  # Log per debug
        except json.JSONDecodeError as e:
            print("Errore nella decodifica del JSON:", e)
            return JsonResponse({'status': 'error', 'message': 'Errore nella decodifica del JSON.'})

        question = body_data.get('question', '').strip()
        
        if not question:
            return JsonResponse({'status': 'error', 'message': 'Il campo domanda è vuoto.'})

        if not check_model_files(QA_TRAINED_MODEL):
            return JsonResponse({'status': 'error', 'message': 'Il modello preaddestrato non è disponibile.'})

        model, tokenizer = load_model(QA_TRAINED_MODEL)
        answer = generate_answer(question, model, tokenizer)
        
        return JsonResponse({'status': 'success', 'answer': answer})
    
    return JsonResponse({'status': 'error', 'message': 'Richiesta non valida.'})




import logging

# Configura il logger
logger = logging.getLogger(__name__)

def chat_view(request):
    #return render(request, 'admin/chat_window.html')
    return render(request, 'admin/chat_template.html')

# views.py




from .evmodels import get_answer  # Importa la funzione get_answer
from .config import CONTEXTS_FILENAME
from .models import Domande
from pathlib import Path

def chat_response(request):
    # pdb.set_trace()
    if request.method == 'POST':
        data = json.loads(request.body)  # Carica i dati JSON dal corpo della richiesta
        user_input = data.get('user_input')
        dataset = data.get('dataset')
        if user_input: 
            result = get_answer(user_input, request, dataset)  # Chiamata alla funzione get_answer
            return JsonResponse({
                'response': result['best_answer'],
                'document_id': result['document_id'],
                'document_name': result['document_name'],
                'chunk_id': result['chunk_id']
            })
        else:
            return JsonResponse({'error': 'user_input is missing'}, status=400)
        
    return JsonResponse({'error': 'Invalid request'}, status=400)




from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Documenti
from .utils import costruisci_dataset_per_agmodel  # Importa la funzione utilitaria
from django.core.cache import cache


@csrf_exempt
def request_dataset(request):
    print ("Sono in request dataset")
    if request.method == 'GET':
        print (f"request method {request.method}")
        try:
            # Ottieni tutti i documenti
            documents = Documenti.objects.all()
            # Ottieni una lista di tutti gli ID dei documenti
            document_ids = [doc.id for doc in documents]
            
            # Costruisci il dataset utilizzando la funzione utilitaria
            dataset = costruisci_dataset_per_agmodel(document_ids)
            json_dataset = json.dumps(dataset)
            
            dataset_key = f"chat_dataset_{request.session.session_key}"
            
            # Salva il dataset nella cache
            cache.set(dataset_key, json_dataset, timeout=3600)  # Timeout di 1 ora

             # Salva la chiave del dataset nella sessione
            request.session['chat_dataset_key'] = dataset_key
            request.session['show_modal'] = True  # Aggiungi un flag per mostrare la modale
            
            
            return JsonResponse({'dataset_id': dataset_key})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

        
@csrf_exempt  # Permetti richieste POST senza CSRF token solo se necessario
def reset_modal_session(request):
    if request.method == 'POST':
        request.session['show_modal'] = False
        return JsonResponse({'success': True})
    return JsonResponse({'success': False}, status=400)


import json
import statistics
from django.http import JsonResponse
from .models import Domande, Documenti

def save_response(request):
    print ("E' tornato sul server .... sono in save response")
    if request.method == 'POST':
        try:
            # Estrazione dei dati dalla richiesta POST
            data = json.loads(request.body)
            question = data.get('question')
            answer = data.get('answer')
            document_id = data.get('document_id')
            chunk_id = data.get('chunk_id')

            print (f"domanda: {question}")
            print (f"risposta: {answer}")
            

            # Verifica che tutti i parametri necessari siano presenti
            if not all([question, answer, document_id, chunk_id]):
                print (f" mancano alcuni parametri ...")
                return JsonResponse({'error': 'Parametri mancanti nella richiesta'}, status=400)
            

            # Recupera l'istanza del documento utilizzando l'ID
            try:
                documento_instance = Documenti.objects.get(id=document_id)
            except Documenti.DoesNotExist:
                return JsonResponse({'error': f"Documento con id {document_id} non trovato."}, status=404)

            print (f"documento: {documento_instance.descrizione}")
            print (f"chunk: {chunk_id}")
            # Recupera tutte le domande con lo stesso documento e context_id (chunk_id)
            domande_simili = Domande.objects.filter(documento=documento_instance, context_id=chunk_id)

            # Calcola il rating_domanda come il massimo del rating_domanda tra le domande simili
            if domande_simili.exists():
                print (f"ci sono {len(domande_simili)}  coppie domanda risposta per lo stesso contesto")
                rating_domanda = max(domande_simili.values_list('rating_domanda', flat=True))
                print (f"massimo punteggio della domanda: {rating_domanda}")
                ratings_risposta = domande_simili.values_list('rating_risposta', flat=True)
                media_rating_risposta = statistics.mean(ratings_risposta)
                dev_standard_rating_risposta = statistics.stdev(ratings_risposta) if len(ratings_risposta) > 1 else 0
                rating_risposta = media_rating_risposta + dev_standard_rating_risposta
                print (f"media punteggio risposte {media_rating_risposta:7.4f} dev std risposte {dev_standard_rating_risposta:7.4f} rating risposta {rating_risposta:7.4f}")
            else:
                rating_domanda = 0  # Se non ci sono domande precedenti, si assegna un valore di default
                rating_risposta = 0  # Se non ci sono domande precedenti, si assegna un valore di default

            # Crea una nuova istanza di Domande
            nuova_domanda = Domande(
                domanda=question,
                risposta=answer,
                documento=documento_instance,  # Salva l'istanza del documento
                auto_generated=False,  # Imposta auto_generated a False
                rating_domanda=rating_domanda,  # Imposta il rating_domanda calcolato
                rating_risposta=rating_risposta,  # Imposta il rating_risposta calcolato
                context_id=chunk_id  # Imposta il context_id con il chunk_id
            )

            # Salva la nuova domanda nel database
            nuova_domanda.save()

            # Restituisci una risposta di successo
            return JsonResponse({'success': 'Risposta salvata correttamente', 'domanda_id': nuova_domanda.id})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Errore nella decodifica del JSON'}, status=400)

    # Se il metodo della richiesta non è POST, restituisce un errore
    return JsonResponse({'error': 'Metodo non supportato'}, status=405)

