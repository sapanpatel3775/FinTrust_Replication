import numpy as np
import pandas as pd

def make_tra(data, constituents_data, top_data):
    for i in data:
        for j in constituents_data:
            if j[1] == i[1]:
                sector = j[2]
                break
        for j in top_data:
            if j[0] == sector:
                top_c = j[1]
                top_c_ = j[2]
                if j[1] == i[1]:
                    top_c = j[2]
                    top_c_ = j[3]
        text = i[2].replace(top_c,top_c_).lower().replace(' we ',' '+top_c+' ').replace(' our ',' '+top_c+"'s ").lower()
        i[2] = text
    return data

def main():
    in_file = '../Used_Data/earnings_call.npy'
    out_file = '../Used_Data/tra_earnings_call.npy'
    top_file = 'top_company.xlsx'
    constituents_file = 'constituents-financials.xlsx'

    data = np.load(in_file)
    constituents_df = pd.read_excel(constituents_file)
    constituents_data = constituents_df.values
    top_df = pd.read_excel(top_file)
    top_data = top_df.values

    data = make_tra(data, constituents_data, top_data)
    
    np.save(out_file, data)
    return

if __name__ == "__main__":
    main()