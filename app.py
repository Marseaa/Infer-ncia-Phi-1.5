import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def gerar_texto(prompt):
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=200)
    texto_gerado = tokenizer.batch_decode(outputs)[0]

    return texto_gerado

if __name__ == "__main__":
    print("----GERADOR DE TEXTO PHI-1.5----")
    prompt = input("Insira o texto inicial: ")
    texto_gerado = gerar_texto(prompt)
    print("Texto Gerado:")
    print(texto_gerado)
