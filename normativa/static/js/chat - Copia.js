function openChatModal() {
    document.getElementById('chat-modal').style.display = 'flex';
}

function closeChatModal() {
    document.getElementById('chat-modal').style.display = 'none';
}

// Funzione per il ridimensionamento automatico della finestra di input
function autoResizeInput() {
    const input = document.getElementById('chat-input');
    input.style.height = 'auto'; // Reset height
    input.style.height = input.scrollHeight + 'px'; // Imposta l'altezza al livello del contenuto
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function sendMessage() {
    const chatArea = document.getElementById('chat-area');
    const userInput = document.getElementById('chat-input').value;

    if (userInput.trim()) {
        const userMessage = document.createElement('div');
        userMessage.textContent = `Tu: ${userInput}`;
        chatArea.appendChild(userMessage);

        // Invia la richiesta al server
        fetch('/normativa/chat-response/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({ user_input: userInput }),  // Assicurati che i dati siano in formato JSON
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.response) {
                const responseMessage = document.createElement('div');
                responseMessage.textContent = `Irene: ${data.response}`;
                responseMessage.style.fontStyle = 'italic';
                chatArea.appendChild(responseMessage);
            } else if (data.error) {
                console.error('Errore dal server:', data.error);
            }
        })
        .catch(error => {
            console.error('Errore durante l\'invio del messaggio:', error);
        });

        document.getElementById('chat-input').value = '';
        autoResizeInput(); // Reset dell'altezza dopo l'invio del messaggio

        // Scrolla fino in fondo
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Aggiunge l'evento di input per il ridimensionamento automatico
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('input', autoResizeInput);
    }

    // Aggiunge l'evento di invio messaggio al tasto Invio
    if (chatInput) {
        chatInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault(); // Impedisce il comportamento predefinito (aggiungere una nuova riga)
                sendMessage(); // Invia il messaggio
            }
        });
    }

    // Aggiunge l'evento di click al pulsante Esci
    const exitButton = document.getElementById('exit-button');
    if (exitButton) {
        exitButton.addEventListener('click', closeChatModal);
    }

    // Aggiunge l'evento di click al link "Avvia Chat"
    const chatLink = document.querySelector('a[href="#"]'); // Assicurati che il selettore sia corretto
    if (chatLink) {
        chatLink.addEventListener('click', function(event) {
            event.preventDefault(); // Previene il comportamento di default del link
            openChatModal(); // Apre la finestra modale
        });
    }
});
