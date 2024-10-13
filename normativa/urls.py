# In yourapp/urls.py
from django.urls import path
from . import views
from . import trainmodel

app_name = 'normativa'

urlpatterns = [
    # altre URL...
    path('documents/', views.DocumentListView.as_view(), name='document_list'),
    path('download/<int:document_id>/', views.download_document, name='download_document'),
    path('addestramento/', views.AddestraAdminView.as_view(), name='ai_addestramento_admin'),
    #path('setup/', views.SetupAdminView(), name='ai_setup_admin'),
    path('get_unita_organizzative/', views.get_unita_organizzative, name='get_unita_organizzative'),
    path('train-model/', trainmodel.train_qgmodel, name='train_model'),
    path('generate-answer/', views.generate_answer_view, name='generate_answer_view'),
    path('chat-response/', views.chat_response, name='chat_response'),
    # path('chat/', views.doc_chat, name='doc_chat'),
    path('reset-modal-session/', views.reset_modal_session, name='reset_modal_session'),
    path('request-dataset/', views.request_dataset, name='request_dataset'),
    path('save-response/', views.save_response, name='save_response'),

   
]

     
