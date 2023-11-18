import pandas as pd

sentence_data = pd.read_json("./sentences/original.jsonl", lines=True)
pos_data = pd.DataFrame()
neg_data = pd.DataFrame()


for index, row in sentence_data.iterrows():
    sentence_label = int(row['label'])
    try:
        if sentence_label in [1, 2]:    #  it is a positive sentence
            pos_data = pos_data.append(row)
        elif sentence_label in [0, 3, 4]: #  it is a negative sentence
            neg_data = neg_data.append(row)
    except Exception as e:
        print(e)

pos_data.to_json(r'./sentences/positive.jsonl', orient='records', lines=True)
neg_data.to_json(r'./sentences/negative.jsonl', orient='records', lines=True)
