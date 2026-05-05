import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfiguration

# 1. Configurazione per il risparmio di memoria (4-bit quantization)
bnb_config = BitsAndBytesConfiguration(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Usa bfloat16 se hai una RTX 3000/4000
    bnb_4bit_use_double_quant=True,
)

# 2. Nome del modello su Hugging Face
model_id = "sapienzaLP/Minerva-3B-base-v1.0"

print(f"Sto scaricando/caricando il modello {model_id}...")

# 3. Caricamento del Tokenizer (fondamentale per l'italiano)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Necessario per l'addestramento

# 4. Caricamento del Modello
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # Distribuisce automaticamente il modello sulla GPU
    trust_remote_code=True
)

print("Modello caricato correttamente!")

# 5. Test rapido di generazione
prompt = "L'importanza della lingua italiana nel mondo moderno è"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50)
print("\n--- Test di Generazione ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True)