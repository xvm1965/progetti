from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required

def redirect_to_admin(request):
    return redirect('/admin/')
