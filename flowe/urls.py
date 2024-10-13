"""
URL configuration for flowe project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from .views import redirect_to_admin
from normativa.admin import my_admin_site
from django.conf import settings
from django.conf.urls.static import static

from normativa import views

urlpatterns = [
    path('admin/', my_admin_site.urls),  # Admin site personalizzato
    path('normativa/', include('normativa.urls')),  # Includi le URL dell'applicazione normativa
    path('', redirect_to_admin, name='redirect_to_admin'),  # Reindirizza l'utente autenticato
    

    
   ] + static(settings.MEDIA_URL, document_root=settings.DOCS_DIR)
