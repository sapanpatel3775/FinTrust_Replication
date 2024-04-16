import numpy as np
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import torch
import pandas as pd

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")
 
def measure_consistency(data, prompt, model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=device)

    predictions = pd.DataFrame(columns=['Date','Company', 'Sentence', 'Original', 'Negative', 'Symmetric', 'Transitive'])
    for row_idx in tqdm(range(data.shape[0]), desc=f"Predicting with {model_name}", total=data.shape[0]):
        date = data[row_idx, 0]
        company = data[row_idx, 1]
        transcripts = data[row_idx, 2:]
        splits = []
        for i in range(transcripts.shape[0]):
            splits.append(transcripts[i].split('\n'))
        splits = np.array(splits)
        for sentence_idx in tqdm(range(splits.shape[1]), desc=f"Comparing sentences for {company}", total=splits.shape[1]):
            pred = [date, company, sentence_idx]
            for i in range(splits.shape[0]):
                pred.append(fill_mask(prompt.replace('[PHRASE]', splits[i, sentence_idx])))
            predictions.loc[len(predictions.index)] = pred
    return predictions

def analyze_predictions(folder_path):
    pass

def load_joined():
    ec_path = os.path.join('..', 'Used_Data', 'earnings_call.npy')
    neg_ec_path = os.path.join('..', 'Used_Data', 'neg_earnings_call.npy')
    sym_ec_path = os.path.join('..', 'Used_Data', 'sym_earnings_call.npy')
    tra_path = os.path.join('..', 'Used_Data', 'tra_earnings_call.npy')

    ec = np.load(ec_path)
    neg_ec = np.load(neg_ec_path)
    sym_ec = np.load(sym_ec_path)
    tra_ec = np.load(tra_path)

    return np.column_stack((ec, neg_ec[:, 2], sym_ec[:, 2], tra_ec[:, 2]))
        
def main():
    device = get_device()

    data = load_joined()
    print(data.shape)

    prompt = """
    Given the following text from an earnings call:
    [PHRASE]

    Based solely on the information provided in this text, do you predict the stock price for the associated company will go up or down in the near future? 
    Please answer with either "up" or "down".

    The stock price for the associated company will go [MASK].
    """

    models = [
        'bert-base-uncased', 
        'bert-large-uncased', 
        'roberta-base',                     #Uses a different mask token (<mask>), so would need to remake data
        'roberta-large',                    #Uses a different mask token (<mask>), so would need to remake data
#        'ProsusAI/finbert',                #Strictly sentiment analysis model, gives garbage when mask infilling
        'yiyanghkust/finbert-pretrain',
        'distilbert-base-uncased',
        ]
    
    for model in tqdm(models, desc="Processing models"):
        model_predictions_path = f"./{model.replace('/', '-')}-predictions.csv"
        if not os.path.exists(model_predictions_path):
            measure_consistency(
                data,
                prompt, 
                model, 
                device
            ).to_csv(model_predictions_path, index=False)

    analyze_predictions('.')

if __name__ == '__main__':
    main()