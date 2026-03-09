import json
from datasets import load_dataset
from tqdm import tqdm
import time

def get_label(text):
    text_len = len(text)
    punctuation_chars = ".,!?;:-()\""
    punctuation_density = sum(1 for c in text if c in punctuation_chars) / (text_len + 1)
    
    if text_len >= 800 and punctuation_density < 0.20:
        return "good"
    return "bad"

def collect_dataset(output_file, total_target=1000):
    # Aumenta il timeout e imposta retry
    import os
    os.environ['HF_HUB_READ_TIMEOUT'] = '60'  # 60 secondi invece di 10
    os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'
    
    print("--- Inizio download (con timeout aumentato) ---")
    
    # Retry loop
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Tentativo {attempt + 1}/{max_retries}...")
            ds = load_dataset(
                "HuggingFaceFW/fineweb-2", 
                name="ita_Latn",
                split="train", 
                streaming=True,
                trust_remote_code=True
            )
            print("✓ Dataset caricato con successo!")
            break
        except Exception as e:
            print(f"✗ Errore al tentativo {attempt + 1}: {type(e).__name__}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)  # Aspetta 10s, poi 20s, poi 30s
                print(f"  Riprovo tra {wait_time} secondi...")
                time.sleep(wait_time)
            else:
                print("✗ Falliti tutti i tentativi!")
                raise
    
    counts = {"good": 0, "bad": 0}
    target_per_class = total_target // 2
    
    print(f"Target per classe: {target_per_class}\n")
    
    f = open(output_file, 'w', encoding='utf-8')
    processed = 0
    
    try:
        for example in tqdm(ds, desc="Elaborazione"):
            text = example.get("text", "")
            if not text:
                continue
            
            processed += 1
            label = get_label(text)
            
            if processed % 100 == 0:
                pct_good = (counts["good"] / target_per_class * 100) if target_per_class > 0 else 0
                pct_bad = (counts["bad"] / target_per_class * 100) if target_per_class > 0 else 0
                print(f"  Processati: {processed} | Good: {counts['good']}/{target_per_class} ({pct_good:.1f}%) | Bad: {counts['bad']}/{target_per_class} ({pct_bad:.1f}%)")
            
            if counts[label] < target_per_class:
                entry = {
                    "id": f"{label}_{counts[label]}",
                    "text": text,
                    "metadata": {"label": label, "length": len(text)}
                }
                
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                f.flush()
                counts[label] += 1
            
            if counts["good"] >= target_per_class and counts["bad"] >= target_per_class:
                print(f"\n✓ Target raggiunto!")
                break
    finally:
        f.close()
        print(f"\nChiusura file completata.")
        print(f"Processati totali: {processed}")
        print(f"Salvati: Good={counts['good']}, Bad={counts['bad']}")
        print(f"File salvato in: {output_file}")

if __name__ == "__main__":
    collect_dataset("../../data/train/training_data.jsonl")