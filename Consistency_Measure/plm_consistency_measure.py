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

    predictions = pd.DataFrame(columns=['Date','Company', 'Sentence', 'Original', 'Negative', 'Symmetric', 'Transitive', 'Additive'])
    for row_idx in tqdm(range(data.shape[0]), desc=f"Predicting with {model_name}", total=data.shape[0]):
        date = data[row_idx, 0]
        company = data[row_idx, 1]
        transcripts = data[row_idx, 2:]
        splits = []
        for i in range(transcripts.shape[0]):
            splits.append(transcripts[i].split('\n'))
        # Very wierd inexplicable bug where the Transitive Exxon has less lines than all others - cant figure out why
        try:
            splits = np.array(splits)
        except:
            print(f"Error with {company} - arrays are different lengths")
            continue
        for sentence_idx in range(splits.shape[1]):
            pred = [date, company, sentence_idx]
            for i in range(splits.shape[0]):
                phrase = prompt.replace('[PHRASE]', splits[i, sentence_idx])
                if 'roberta' in model_name:
                    phrase = phrase.replace('[MASK]', '<mask>')
                pred.append(fill_mask(phrase)[0]['token_str'])
            predictions.loc[len(predictions.index)] = pred
    return predictions

def load_joined():
    ec_path = os.path.join('..', 'Used_Data', 'earnings_call.npy')
    neg_ec_path = os.path.join('..', 'Used_Data', 'neg_earnings_call.npy')
    sym_ec_path = os.path.join('..', 'Used_Data', 'sym_earnings_call.npy')
    tra_ec_path = os.path.join('..', 'Used_Data', 'tra_earnings_call.npy')
    add_ec_path = os.path.join('..', 'Used_Data', 'add_earnings_call.npy')

    ec = np.load(ec_path)
    neg_ec = np.load(neg_ec_path)
    sym_ec = np.load(sym_ec_path)
    tra_ec = np.load(tra_ec_path)
    add_ec = np.load(add_ec_path)

    return np.column_stack((ec, neg_ec[:, 2], sym_ec[:, 2], tra_ec[:, 2], add_ec[:, 2]))

def analyze_predictions(folder_path):
    files = [file for file in os.listdir(folder_path) if file.endswith("-predictions.csv")]

    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        data = df.values
        # only have predictions
        data = data[:, 3:]
        # convert to str and remove white space
        data = np.array(data, dtype=str)
        data = np.char.strip(data)
        # remove nonsensical
        data = data[np.all(np.isin(data, ['up', 'down']), axis=1)]
        # convert to binary
        data = np.where(data == 'up', 1, 0)
        # flip negation
        data[:, 1] = 1 - data[:, 1]
        # compare to original
        data[:, 1] = data[:, 0] - data[:, 1]
        data[:, 2] = data[:, 0] - data[:, 2]
        data[:, 3] = data[:, 0] - data[:, 3]
        data[:, 4] = data[:, 0] - data[:, 4]
        # remove original
        data = data[:, 1:]
        # absolute, any differing will be 1 and correct will be 0
        data = np.abs(data)
        # mean gives percentage different but we want percentage correct
        avg = 1 - np.mean(data)
        data = 1 - np.mean(data, axis=0)

        print('---------------')
        print(f"Statistics for {file.replace('-predictions.csv', '')}:\n")
        print(f"Per Group:\n{data}")
        print(f"Average:\n{avg}")
        print('---------------')
        
def main():
    device = get_device()

    data = load_joined()

    prompt = """
    Given the following text from an earnings call:

    [PHRASE]

    Based solely on the information provided in this text, do you predict the stock price for the associated company will go up or down in the near future?

    The stock price for the associated company will likely go [MASK].
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