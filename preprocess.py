import pandas as pd
from nltk.corpus import stopwords

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    # Aplicar limpeza e processamento
    data['cleaned_text'] = data['text'].apply(lambda x: limpar_texto(x))
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data('raw_data.csv', 'cleaned_data.csv')
