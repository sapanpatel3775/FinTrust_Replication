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

def mask_words(sentences, positive_words, negative_words):
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            lower_word = word.lower().rstrip('.').rstrip(',')
            if lower_word in positive_words or lower_word in negative_words:
                masked_sentence = words[:]
                masked_sentence[i] = '[MASK]'
                masked_sentence = ' '.join(masked_sentence)
                sentiment = 'positive' if lower_word in positive_words else 'negative'
                yield masked_sentence, lower_word, sentiment

def generate_masked_data(ec, positive_words, negative_words):
    masked_phrases = []
    for date, company, transcript in tqdm(ec, desc="Generating masked transcript data"):
        sentences = transcript.split('\n')
        for masked_sentence, word, sentiment in mask_words(sentences, positive_words, negative_words):
            masked_phrases.append([date, company, masked_sentence, word, sentiment])
    masked_phrases = pd.DataFrame(masked_phrases, columns=['Date', 'Company', 'MaskedSentence', 'Word', 'Sentiment'])
    return masked_phrases

def predict_masked_tokens(masked_phrases, model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=device)

    predicted_tokens = []
    for _, row in tqdm(masked_phrases.iterrows(), desc=f"Predicting with {model_name}", total=masked_phrases.shape[0]):
        if 'roberta' in model_name:
            prediction = fill_mask(row['MaskedSentence'].replace('[MASK]', '<mask>'))
        else:
            prediction = fill_mask(row['MaskedSentence'])
        predicted_tokens.append(prediction[0]['token_str'])
    masked_phrases['Prediction'] = predicted_tokens
    return masked_phrases

def analyze_predictions(folder_path, positive_words, negative_words):
    files = [file for file in os.listdir(folder_path) if file.endswith("-predictions.csv")]
    sentiment_mapping = {word: 'positive' for word in positive_words}
    sentiment_mapping.update({word: 'negative' for word in negative_words})
    
    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        analysis_results = []
        for _, row in df.iterrows():
            original_word = row['Word'].lower().strip()
            predicted_word = row['Prediction'].lower().strip()
            original_sentiment = row['Sentiment']

            if original_word == predicted_word:
                result = 'exact'
            elif predicted_word in sentiment_mapping:
                if sentiment_mapping[predicted_word] == original_sentiment:
                    result = 'same'
                else:
                    result = 'opposite'
            else:
                result = 'none'

            analysis_results.append(result)

        df['AnalysisResult'] = analysis_results
        df.to_csv(os.path.join(folder_path, file), index=False)

        print('---------------')
        print(f"Statistics for {file.replace('-predictions.csv', '')}:\n")
        print("Including None:")
        result_counts = pd.Series(analysis_results).value_counts(normalize=True) * 100
        print(result_counts.round(2).to_string())
        total_results = len(analysis_results)
        consistency_count = analysis_results.count('exact') + analysis_results.count('same')
        consistency_percentage = (consistency_count / total_results) * 100
        print(f"Consistency - {consistency_percentage:.2f}%")


        print("\nExcluding None:")
        filtered_results = [result for result in analysis_results if result != 'none']
        result_counts_filtered = pd.Series(filtered_results).value_counts(normalize=True) * 100
        print(result_counts_filtered.round(2).to_string())
        total_results = len(filtered_results)
        consistency_count = filtered_results.count('exact') + filtered_results.count('same')
        consistency_percentage = (consistency_count / total_results) * 100
        print(f"Consistency - {consistency_percentage:.2f}%")
        print('---------------')
        
def main():
    device = get_device()

    ec_path = os.path.join('..', 'Data', 'earnings_call.npy')
    ec = np.load(ec_path)

    positive_words = [
    "more", "positive", "yes", "able", "increase", "sales", "sale", "best", "larger", "large", 
    "good", "high", "higher", "up", "like", "right", "a lot of", "many", "much", "believe", "better", 
    "revenue", "remain", "continuing", "continue", "approve", "grew", "growth", "grow", "short",
    "improvement", "improve", "focus", "major", "strong", "full", "start", "progress", "greater",
    "earnings", "well", "expect", "over", "forward", "margin", "profit", "benefits", "income",
    "benefit", "completely", "most", "add", "unchange", "unchanged", "opportunities", "opportunity", "within"
    ]

    negative_words = [
        "less", "negative", "no", "unable", "decrease", "buy", "worst", "smaller", "small", 
        "bad", "low", "lower", "down", "dislike", "wrong", "few", "little", "disbelieve", "worse", 
        "expense", "abandon", "stopping", "stop", "refuse", "decayed", "decay", "degenerate", 
        "degeneration", "ignore", "minor", "weak", "empty", "end", "decline", "cost", "badly", 
        "dismiss", "below", "back", "loss", "harm", "slightly", "least", "decrease", "change",
        "changes", "without"
    ]
    
    masked_ec_path = "./masked-ec.csv"
    if os.path.exists(masked_ec_path):
        masked_phrases = pd.read_csv(masked_ec_path)
    else:
        masked_phrases = generate_masked_data(ec, positive_words, negative_words)
        masked_phrases.to_csv(masked_ec_path, index=False)

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
            predict_masked_tokens(masked_phrases.copy(), model, device).to_csv(model_predictions_path, index=False)


    analyze_predictions('.', positive_words, negative_words)

if __name__ == '__main__':
    main()