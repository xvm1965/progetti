from django.contrib.admin import AdminSite
from django.urls import reverse, path
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from .models import Domande
from django.contrib import admin
from django.http import JsonResponse


class MyAdminSite(AdminSite):
    site_header = 'Flowe Normativa (header)'
    site_title = 'Flowe Normativa (site title)'
    index_title = 'Flowe Normativa (index title)'
    

       
    def get_app_list(self, request, dummy=None):
        app_list = super().get_app_list(request)
        
            
        #chat_url = reverse('normativa:chat_view')
        chat_url = "#"
        addestra_url = reverse('normativa:ai_addestramento_admin')
        #setup_url = reverse('normativa:ai_setup_admin')
    
        # Verifica se l'utente è superuser o è nel gruppo 'administrators'
        is_superuser = request.user.is_superuser
        is_administrator = request.user.groups.filter(name='administrators').exists()

        ai_apps = [
            {
                'name': _('AI'),
                'app_label': 'normativa',
                'models': [
                   
                    {
                        'name': _('Chat'),
                        'admin_url': chat_url,
                        # 'onclick': "openChatModal(); return false;", #openchatmodeal  è definita in base_site.html nella directiory flowe\flowe\template\admin
                    },
                    
                ],
            },
        ]

        if is_superuser or is_administrator:
            ai_apps[0]['models'].extend([
                {
                    'name': _('Addestra'),
                    'admin_url': addestra_url,
                },
                # {
                #     'name': _('Imposta'),
                #     'admin_url': setup_url,
                # },
            ])
    
        app_list += ai_apps
        return app_list
      
       

from django.urls import path
from urllib.parse import quote
from django.contrib import messages
from django.http import HttpResponseRedirect



my_admin_site = MyAdminSite(name='myadmin')

