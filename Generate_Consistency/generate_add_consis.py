import numpy as np
import copy
import random

def make_add(data, increasing, decreasing):
    for row in range(data.shape[0]):
        text = data[row][2]
        sentences = text.split('\n')
        added_sentences = []
        for sentence in sentences:
            if row in increasing:
                idx = random.choice(increasing)
            else:
                idx  = random.choice(decreasing)
            lines = data[idx, 2].split('\n')
            line = random.randrange(len(lines))
            added_sentences.append(sentence + ', ' + lines[line])

        added_text = '\n'.join(added_sentences)
        data[row][2] = added_text

    return data

def split(data):
    increasing = []
    decreasing = []
    for i in range(data.shape[0]):
        if float(data[i, 1]) > 0:
            increasing.append(i)
        else:
            decreasing.append(i)
    
    return increasing, decreasing

def main():
    in_file = '../Data/earnings_call_3days.npy'
    out_file = '../Used_Data/add_earnings_call.npy'

    data = np.load(in_file)

    increasing, decreasing = split(data)

    data = make_add(data[:, [0,2,3]], increasing, decreasing)

    np.save(out_file, data)
    return

if __name__ == "__main__":
    main()
