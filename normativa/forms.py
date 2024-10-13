from django import forms

class ChatForm(forms.Form):
    question = forms.CharField(
        label='Domanda',
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',  # Classe CSS per Bootstrap
            'placeholder': 'Inserisci la tua domanda qui',  # Testo di esempio
            'style': 'width: 100%;',  # Stile inline per la larghezza
        })
    )
# Importa le librerie necessarie

