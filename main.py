from retrieve import retrieve
from generate import generate_response
from document_loader import load_documents

def main(query):
    indices = retrieve(query, 'index.faiss')
    documents = load_documents(indices)
    context = " ".join(documents)
    prompt = f"Contexto: {context}\nPergunta: {query}\nResposta:"
    response = generate_response(prompt, 'llama-3.1-model', 'tokenizer/')
    print(response)

if __name__ == "__main__":
    user_query = input("Digite sua pergunta: ")
    main(user_query)
