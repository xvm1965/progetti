let chatDataset_id = null;

function openChatModal(dataset_id = null) {
    chatDataset_id = dataset_id; // Salva il dataset nella variabile globale

    const chatModal = document.getElementById('chat-modal');
    if (chatModal) {
        chatModal.style.display = 'flex'; // Mostra il modale
    }

    // Utilizza il dataset per inizializzare il modale
    const warningMessage = document.createElement('div');
    if (chatDataset_id) {
        warningMessage.textContent = 'Dataset: ' + chatDataset_id;
        warningMessage.style.color = 'blue';
    } else {
        warningMessage.textContent = 'Il dataset non è stato passato.';
        warningMessage.style.color = 'red';
    }
    const chatArea = document.getElementById('chat-area');
    if (chatArea) {
        chatArea.innerHTML = '';  // Elimina il contenuto precedente
        chatArea.appendChild(warningMessage);
    }
}

function closeChatModal() {
    // Nascondi la modale
    document.getElementById('chat-modal').style.display = 'none';
    console.log("Entro in closeChatModal, chatDataset_id: " + chatDataset_id);

    // Reset della variabile chatDataset_id
    chatDataset_id = null;
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

// Nuova funzione per richiedere un nuovo dataset_id
function requestNewDataset() {
    console.log ("siamo in requestNewDataset ... lancia request-dataset")
    return fetch('/normativa/request-dataset/', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.dataset_id) {
            chatDataset_id = data.dataset_id;
            return data.dataset_id;
        } else {
            throw new Error('Failed to get dataset_id from server');
        }
    });
}

function sendMessage() {
    const chatArea = document.getElementById('chat-area');
    const userInput = document.getElementById('chat-input').value;
    console.log("è atterrato in sendmessage");

    if (userInput.trim()) {
        console.log("ha rilevato un messaggio: " + userInput.trim());
        const userMessage = document.createElement('div');
        userMessage.textContent = `Tu: ${userInput}`;
        chatArea.appendChild(userMessage);
        console.log("lancia la fetch per restituire il controllo al server ed elaborare la risposta");
        console.log('user input: ' + userInput);
        console.log('dataset: ' + chatDataset_id);

        // Se chatDataset_id è null, richiedi un nuovo dataset_id
        if (!chatDataset_id) {
            console.log ("in sendMessage, ha rilevato che non c'è un dataset e cerca di crearlo")
            requestNewDataset()
                .then(() => {
                    sendRequestToServer(userInput); // Invia la richiesta dopo aver ottenuto il nuovo dataset_id
                })
                .catch(error => {
                    console.error('Errore durante la richiesta di un nuovo dataset_id:', error);
                });
        } else {
            sendRequestToServer(userInput); // Invia la richiesta se chatDataset_id è già presente
        }

        document.getElementById('chat-input').value = '';
        autoResizeInput(); // Reset dell'altezza dopo l'invio del messaggio
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

function sendRequestToServer(userInput) {
    // Disabilita l'input dell'utente e mostra l'icona di caricamento
    const chatInput = document.getElementById('chat-input');
    const saveButton = document.getElementById('save-button');
    
    chatInput.disabled = true;
    saveButton.disabled = true;
    saveButton.style.cursor = 'not-allowed';

    // Visualizza un'icona di caricamento
    const loadingMessage = document.createElement('div');
    loadingMessage.textContent = "Elaborazione in corso..."; // Puoi anche usare un'icona animata
    loadingMessage.id = "loading-message"; // Aggiungi un ID per poterlo rimuovere facilmente
    loadingMessage.style.fontStyle = 'italic';
    document.getElementById('chat-area').appendChild(loadingMessage);

    fetch('/normativa/chat-response/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({ user_input: userInput, dataset: chatDataset_id }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Rimuovi il messaggio di caricamento
        const loadingMsgElement = document.getElementById('loading-message');
        if (loadingMsgElement) {
            loadingMsgElement.remove();
        }

        if (data.response) {
            const responseMessage = document.createElement('div');
            responseMessage.textContent = `Irene: ${data.response}`;
            responseMessage.style.fontStyle = 'italic';
            document.getElementById('chat-area').appendChild(responseMessage);

            // Abilita il pulsante "Salva" solo quando arriva una risposta
            saveButton.disabled = false;
            saveButton.removeAttribute('disabled'); // Rimuovi l'attributo 'disabled'
            saveButton.style.backgroundColor = '#28a745'; // Verde per abilitato
            saveButton.style.cursor = 'pointer';
            saveButton.style.color = 'white';

            saveButton.onclick = function() {
                // Disabilita il pulsante dopo il clic
                saveButton.disabled = true;
                saveButton.setAttribute('disabled', 'true'); // Aggiungi l'attributo 'disabled'
                saveButton.style.backgroundColor = '#ccc'; // Grigio per disabilitato
                saveButton.style.color = '#6c757d'; // Colore del testo grigio per disabilitato
                saveButton.style.cursor = 'not-allowed';
                
                // Salva la domanda, risposta, document_id e chunk_id sul server
                saveResponseToServer(userInput, data.response, data.document_id, data.chunk_id);
            };
        } else if (data.error) {
            console.error('Errore dal server:', data.error);
        }
    })
    .catch(error => {
        console.error('Errore durante l\'invio del messaggio:', error);
        // Rimuovi il messaggio di caricamento in caso di errore
        const loadingMsgElement = document.getElementById('loading-message');
        if (loadingMsgElement) {
            loadingMsgElement.remove();
        }
    })
    .finally(() => {
        // Riabilita l'input utente alla fine del processo
        chatInput.disabled = false;
    });
}



// Funzione per inviare la risposta salvata al server
function saveResponseToServer(userQuestion, serverResponse, documentId, chunkId) {
    fetch('/normativa/save-response/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({
            question: userQuestion,
            answer: serverResponse,
            document_id: documentId, // ID del documento dalla risposta
            chunk_id: chunkId // ID del chunk dalla risposta
        }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            console.log('Risposta salvata con successo');
        } else {
            console.error('Errore durante il salvataggio:', data.error);
        }
    })
    .catch(error => {
        console.error('Errore durante il salvataggio:', error);
    });
}


document.addEventListener('DOMContentLoaded', function() {
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('input', autoResizeInput);
        chatInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });
    }

    const exitButton = document.getElementById('exit-button');
    if (exitButton) {
        exitButton.addEventListener('click', closeChatModal);
    }

    const saveButton = document.getElementById('save-button');
    if (saveButton) {
        // Assicurati che il pulsante sia visibile ma disabilitato all'inizio
        saveButton.style.backgroundColor = '#ccc';
        saveButton.style.color = '#6c757d';
        saveButton.disabled = true;
    }

    const chatLink = document.querySelector('a[href="#"]');
    if (chatLink) {
        chatLink.addEventListener('click', function(event) {
            event.preventDefault();
            openChatModal();
        });
    }
});
