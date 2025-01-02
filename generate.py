from transformers import LlamaForCausalLM, LlamaTokenizer

def generate_response(prompt, model_path, tokenizer_path):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Sua pergunta aqui"
    response = generate_response(prompt, 'llama-3.1-model', 'tokenizer/')
    print(response)
