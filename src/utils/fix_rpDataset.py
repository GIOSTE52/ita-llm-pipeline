import json
import os

"""
Questo script normalizza il dataset RedPajama:
1. Rinomina 'raw_content' in 'text' (se necessario).
2. Trasforma 'bucket' in 'label'.
3. Converte i valori: head -> good, tail -> bad.
"""

# Da controllare i path per farlo funzionare

def fix_rp_dataset():
    # Assicurati che il nome del file sia quello corretto nella tua cartella
    input_file = "it_head_0001.jsonl" 
    temp_file = "rpDataset_temp.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Errore: Il file {input_file} non è in questa cartella!")
        return

    count = 0
    print(f"Inizio elaborazione di {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(temp_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 1. Gestione TESTO: da raw_content a text
                if "raw_content" in data:
                    data["text"] = data.pop("raw_content")
                
                # 2. Gestione LABEL: da bucket a label con mapping valori
                if "bucket" in data:
                    old_value = data.pop("bucket")
                    
                    # Mapping dei valori
                    if old_value == "head":
                        data["label"] = "good"
                    elif old_value == "tail":
                        data["label"] = "bad"
                    else:
                        # Se ci sono altri valori (es. middle), li lasciamo o mappiamo a bad
                        data["label"] = old_value 
                
                # Scriviamo la riga aggiornata
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
                
                # Feedback ogni 1000 righe
                if count % 1000 == 0:
                    print(f"--- Processate {count} righe...", end="\r")

            except Exception as e:
                print(f"\nErrore alla riga {count+1}: {e}")

    # Sostituiamo l'originale con il file temporaneo
    os.replace(temp_file, input_file)
    print(f"\nCompletato! Modificate {count} righe.")
    print(f"Il file {input_file} ora ha le chiavi 'text' e 'label' (good/bad).")

if __name__ == "__main__":
    fix_rp_dataset()