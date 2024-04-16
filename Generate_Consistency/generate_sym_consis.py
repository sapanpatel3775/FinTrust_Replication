import numpy as np
import re

def split_into_sub_sentences(input_text):
    # Split the input text into sentences using regular expressions
    sentences = re.split(r'(?<=[,.] )', input_text)
    return sentences

def make_sym(sentence):
    sub_sentences = split_into_sub_sentences(sentence)
    if len(sub_sentences) == 1:
        return sub_sentences[0]
    if len(sub_sentences) == 2:
        return sub_sentences[1] + ', ' + sub_sentences[0]
    else:
        return ', '.join(np.random.permutation(sub_sentences))


def main():
    in_file = '../Used_Data/earnings_call.npy'
    out_file = '../Used_Data/sym_earnings_call.npy'

    data = np.load(in_file)

    for row in range(data.shape[0]):
        text = data[row][2]
        sentences = text.split('\n')
        sym_sentences = []
        for sentence in sentences:
            sym_sentences.append(make_sym(sentence))

        sym_text = '\n'.join(sym_sentences)
        data[row][2] = sym_text
        
    np.save(out_file, data)
    return

if __name__ == "__main__":
    main()