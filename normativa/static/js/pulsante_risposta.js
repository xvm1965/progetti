document.addEventListener('DOMContentLoaded', function() {
    const answerButton = document.getElementById('answer-btn');
    
    if (answerButton) {
        console.log("Bottone trovato nel DOM");
        answerButton.addEventListener('click', function() {
            console.log("Bottone cliccato!");
            
            // Recupera la domanda dal campo input
            const questionInput = document.getElementById('id_domanda');
            
            if (questionInput && questionInput.value.trim()) {
                const question = questionInput.value.trim();
                
                // Invio della richiesta AJAX
                fetch('/generate_answer/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')  // Assicurati di inviare il token CSRF
                    },
                    body: JSON.stringify({ question: question })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // Visualizza la risposta
                        alert(`Risposta: ${data.answer}`);
                    } else {
                        // Gestisci l'errore
                        alert(`Errore: ${data.message}`);
                    }
                })
                .catch(error => {
                    console.error('Errore:', error);
                });
            } else {
                alert('Il campo domanda è vuoto.');
            }
        });
    } else {
        console.log("Bottone non trovato nel DOM");
    }
});

// Funzione per ottenere il cookie CSRF
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
