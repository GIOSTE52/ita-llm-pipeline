# from datatrove.pipeline.readers import ParquetReader

# # limit determines how many documents will be streamed (remove for all)
# # this will fetch the Italian filtered data
# data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-2/data/ita_Latn/train", limit=400) 
# for document in data_reader():
#     # do something with document
#     print(document)


import os
import json
from datatrove.pipeline.readers import ParquetReader

# 1. Configurazione percorsi
output_dir = "../../data"
output_file = os.path.join(output_dir, "hand_label.jsonl")

# Assicurati che la cartella esista
os.makedirs(output_dir, exist_ok=True)

# 2. Inizializzazione Reader
# Fetch dei dati italiani da FineWeb-2
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-2/data/ita_Latn/train", limit=500)

print(f"Inizio estrazione e salvataggio in: {output_file}")

# 3. Ciclo di lettura e scrittura
with open(output_file, "w", encoding="utf-8") as f:
    count = 0
    for document in data_reader():
        # Creiamo un dizionario con i dati del documento
        # Includiamo 'text' e 'id' (e opzionalmente i metadata originali)
        data_to_save = {
            "id": document.id,
            "text": document.text,
            "metadata": document.metadata
        }
        
        # Scrittura riga per riga in formato JSON
        f.write(json.dumps(data_to_save, ensure_ascii=False) + "\n")
        
        count += 1
        if count % 100 == 0:
            print(f"Salvati {count} documenti...")

print(f"Completato! Salvati {count} documenti in {output_file}")