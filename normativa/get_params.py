import os
import json
from datetime import datetime


#parametri per la creazione dei dataset qa
from config import qa_num_return_sequences, qa_do_sample, qa_top_k, qa_top_p, qa_output_file
def get_user_parameters():
    # Imposta i valori predefiniti per i parametri
    default_params = {
        'num_return_sequences': qa_num_return_sequences,
        'do_sample': qa_do_sample,
        'top_k': qa_top_k,
        'top_p': qa_top_p,
        'output_file': qa_output_file
    }


    # Richiesta dei parametri all'utente
    print("Please enter the values for the following parameters (press Enter to use the default value):")
    num_return_sequences = input(f"Number of return sequences (default {default_params['num_return_sequences']}): ")
    do_sample = input(f"Do sampling (default {default_params['do_sample']}): ")
    top_k = input(f"Top-k value (default {default_params['top_k']}): ")
    top_p = input(f"Top-p value (default {default_params['top_p']}): ")

    params = {
        'num_return_sequences': int(num_return_sequences) if num_return_sequences else default_params['num_return_sequences'],
        'do_sample': do_sample.lower() in ['true', '1'] if do_sample else default_params['do_sample'],
        'top_k': int(top_k) if top_k else default_params['top_k'],
        'top_p': float(top_p) if top_p else default_params['top_p']
    }

    defined = False
    while not defined:
        defined=True
        output_file = input(f"Output file name (default {default_params['output_file']}): ")
        if not output_file: 
            output_file = default_params['output_file']
        else:
            output_file += '.json'
        # output_file = output_file.strip() or default_params['output_file']
        output_file_path = os.path.join(os.getcwd(), output_file)
        if os.path.exists(output_file_path):
            defined = (input(f"Il file {output_file} esiste già. Vuoi sovrascriverlo? (s/n): ").strip().lower()=='s')
    
    params['output_file'] = output_file
    
        
    
    return params




#parametri per l'addestraemento dei modelli
def prompt_user_for_parameters():
    # Valori predefiniti
    defaults = {
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'num_train_epochs': 3,
        'weight_decay': 0.01,
        'eval_steps': 100,
        'save_steps': 500,
        'logging_steps': 1,
        'evaluation_strategy': 'steps'
    }

    # Directory predefinita per salvare i modelli addestrati
    output_training = "trained_model"

    print("Please enter the values for the following parameters (press Enter to use the default value):")

    # Richiesta di input per i parametri che non dipendono dagli altri
    learning_rate = input(f"Learning rate (default {defaults['learning_rate']}): ")
    per_device_train_batch_size = input(f"Per device train batch size (default {defaults['per_device_train_batch_size']}): ")
    per_device_eval_batch_size = input(f"Per device eval batch size (default {defaults['per_device_eval_batch_size']}): ")
    num_train_epochs = input(f"Number of training epochs (default {defaults['num_train_epochs']}): ")
    weight_decay = input(f"Weight decay (default {defaults['weight_decay']}): ")

    # Imposta i valori predefiniti se l'utente non inserisce nulla
    learning_rate = float(learning_rate) if learning_rate else defaults['learning_rate']
    per_device_train_batch_size = int(per_device_train_batch_size) if per_device_train_batch_size else defaults['per_device_train_batch_size']
    per_device_eval_batch_size = int(per_device_eval_batch_size) if per_device_eval_batch_size else defaults['per_device_eval_batch_size']
    num_train_epochs = int(num_train_epochs) if num_train_epochs else defaults['num_train_epochs']
    weight_decay = float(weight_decay) if weight_decay else defaults['weight_decay']

    # Richiesta di evaluation_strategy
    print("\nSelect the evaluation strategy:")
    print("1. No")
    print("2. Steps")
    print("3. Epoch")
    evaluation_strategy_index = input("Enter the number corresponding to your choice (default 2): ")
    
    # Mappa la scelta dell'utente alla strategia di valutazione
    evaluation_strategy_map = {
        '1': 'no',
        '2': 'steps',
        '3': 'epoch'
    }
    evaluation_strategy = evaluation_strategy_map.get(evaluation_strategy_index, defaults['evaluation_strategy'])

    # Richiedi eval_steps solo se la strategia è 'steps'
    if evaluation_strategy == 'steps':
        eval_steps = input(f"Evaluation steps (default {defaults['eval_steps']}): ")
        eval_steps = int(eval_steps) if eval_steps else defaults['eval_steps']
    else:
        eval_steps = defaults['eval_steps']  # Default se non necessario

    # Richiesta di save_steps e logging_steps, che sono sempre rilevanti
    save_steps = input(f"Save steps (default {defaults['save_steps']}): ")
    logging_steps = input(f"Logging steps (default {defaults['logging_steps']}): ")

    # Imposta i valori predefiniti se l'utente non inserisce nulla
    save_steps = int(save_steps) if save_steps else defaults['save_steps']
    logging_steps = int(logging_steps) if logging_steps else defaults['logging_steps']

    params = {
        'learning_rate': learning_rate,
        'per_device_train_batch_size': per_device_train_batch_size,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'num_train_epochs': num_train_epochs,
        'weight_decay': weight_decay,
        'eval_steps': eval_steps,
        'save_steps': save_steps,
        'logging_steps': logging_steps,
        'evaluation_strategy': evaluation_strategy
    }

    # Cerca il file JSON 'qa_library.json'
    params_file_path = os.path.join(os.getcwd(), 'qa_library.json')
    if os.path.exists(params_file_path):
        with open(params_file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

        # Filtra solo i file effettivamente esistenti nella directory corrente
        existing_data = [
            entry for entry in existing_data 
            if os.path.exists(os.path.join(os.getcwd(), entry['output_file']))
        ]

        if existing_data:
            while True:
                print("\nAvailable output files for training:")
                for idx, entry in enumerate(existing_data, start=1):
                    print(f"{idx}. {entry['output_file']} (num_return_sequences: {entry['num_return_sequences']}, num_results: {entry['num_results']})")
                print("0. Cancel operation")

                file_choice = input("Select the file to be used for training (enter the number): ")
                try:
                    file_choice_index = int(file_choice) - 1
                    if file_choice_index == -1:
                        print("Operation cancelled.")
                        params['qa_dataset'] = None
                        model_output_dir = output_training
                        break
                    elif 0 <= file_choice_index < len(existing_data):
                        params['qa_dataset'] = existing_data[file_choice_index]['output_file']
                        model_output_dir = os.path.join(os.getcwd(), existing_data[file_choice_index]['output_file'])
                        break
                    else:
                        print("Invalid choice. Please select a valid file number or 0 to cancel.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print("No valid output files found. Setting 'qa_dataset' to None.")
            params['qa_dataset'] = None
            model_output_dir = output_training
    else:
        print("qa_library.json not found. Setting 'qa_dataset' to None.")
        params['qa_dataset'] = None
        model_output_dir = output_training

    # Chiedi all'utente dove salvare il modello addestrato
    while True:
        model_output_dir = input(f"Enter the directory where the trained model should be saved (default {output_training}): ").strip()
        if not model_output_dir:
            model_output_dir = output_training
        if os.path.exists(model_output_dir):
            overwrite = input(f"The directory {model_output_dir} already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
            if overwrite == 'yes':
                params['model_output_dir'] = model_output_dir
                break
            
        else:
            os.makedirs(model_output_dir, exist_ok=True)
            params['model_output_dir'] = model_output_dir
            break

    return params

if __name__ == "__main__":
    params = prompt_user_for_parameters()
    print(params)
