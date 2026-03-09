import json
from datasets import load_dataset
from tqdm import tqdm
import time

def get_label(text):
    """
    Criteri più realistici per buona/cattiva qualità.
    Mescola più fattori per evitare separabilità perfetta.
    """
    text_len = len(text)
    
    # Se troppo corto → bad
    if text_len < 200:
        return "bad"
    
    # Se troppo lungo e ripetitivo → bad
    if text_len > 5000:
        # Controlla ripetizioni
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:  # Troppa ripetizione
                return "bad"
    
    # Controlla caratteri strani/corrotti
    special_ratio = sum(1 for c in text if ord(c) > 127 and c not in 'àèéìòùÀÈÉÌÒÙ') / (text_len + 1)
    if special_ratio > 0.1:
        return "bad"
    
    # Controlla se sembra spam (molte maiuscole consecutive)
    consecutive_caps = sum(1 for i in range(len(text)-1) if text[i].isupper() and text[i+1].isupper())
    caps_ratio = consecutive_caps / (text_len + 1)
    if caps_ratio > 0.15:
        return "bad"
    
    # Controlla whitespace sporco
    whitespace_ratio = sum(1 for c in text if c.isspace()) / (text_len + 1)
    if whitespace_ratio > 0.4:
        return "bad"
    
    # Controlla URL/email eccessive
    url_count = text.count("http") + text.count("www")
    email_count = text.count("@")
    if url_count + email_count > 5:
        return "bad"
    
    # Se passa tutti i controlli
    return "good"

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