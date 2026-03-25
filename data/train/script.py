import json
import os


def fix_rp_dataset():
    input_file = "rpDataset.jsonl"
    temp_file = "rpDataset_temp.jsonl"
    
    if not os.path.exists(input_file):
        print(f"❌ Errore: Il file {input_file} non è in questa cartella!")
        return

    count = 0
    print(f"🔄 Inizio conversione di {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(temp_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Se esiste raw_content e NON esiste text, rinominiamo la chiave
                if "raw_content" in data:
                    # pop rimuove 'raw_content' e restituisce il suo valore
                    data["text"] = data.pop("raw_content")
                    count += 1
                
                # Scriviamo la riga aggiornata
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️ Errore alla riga {count+1}: {e}")

    # Sostituiamo l'originale con il file temporaneo
    os.replace(temp_file, input_file)
    print(f"✅ Fatto! Modificate {count} righe.")
    print(f"📂 Il file {input_file} è ora pronto con la chiave 'text'.")

if __name__ == "__main__":
    fix_rp_dataset()