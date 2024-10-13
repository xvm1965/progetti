from django.conf import settings
from django.shortcuts import redirect
from django.urls import reverse
from django.contrib import admin




class LoginRequiredMiddleware:
    """
    Middleware that requires a user to be authenticated to view any page other than the login page.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        url=self.get_response(request)
        if not request.user.is_authenticated and not request.path.startswith('/admin/login/'):
            # Se l'utente non è autenticato e non sta cercando di accedere alla pagina di login, reindirizzalo alla home
            url=redirect(settings.LOGIN_URL)
        return url


