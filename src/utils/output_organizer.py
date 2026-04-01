import json
import os
import glob

def output_classification(rejected_dir, output_base_dir):
    insp_dir = os.path.join(output_base_dir, "inspection")
    os.makedirs(insp_dir, exist_ok=True)
    
    # Apriamo in modalità scrittura
    f_good = open(os.path.join(insp_dir, "rejected_was_good.jsonl"), "w", encoding="utf-8")
    f_bad = open(os.path.join(insp_dir, "rejected_was_bad.jsonl"), "w", encoding="utf-8")

    all_jsonls = glob.glob(rejected_dir + "/**/*.jsonl", recursive=True)
    
    for filepath in all_jsonls:
        if "inspection" in filepath: continue
        
        filtro = os.path.basename(os.path.dirname(filepath))
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                
                obj = json.loads(line)
                
                # ESTRAZIONE LABEL DA METADATA
                metadata = obj.get("metadata", {})
                label = metadata.get("label") or obj.get("label")
                
                # Se label è ancora None, passiamo alla riga dopo
                if not label: continue

                nuova_riga = {
                    "text": obj.get("text"),
                    "label_rilevata": label,
                    "scartato_da": filtro
                }
                
                # Smistamento (stringa pulita)
                l_str = str(label).lower().strip()
                if l_str == "good":
                    f_good.write(json.dumps(nuova_riga, ensure_ascii=False) + "\n")
                else:
                    f_bad.write(json.dumps(nuova_riga, ensure_ascii=False) + "\n")

    f_good.close()
    f_bad.close()
    print("✅ Smistamento eseguito con successo guardando i metadati.")

# Definire nuova funzione per la gestione dei grafici in output (csv, plot, ecc...)