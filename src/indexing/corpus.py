import pandas as pd

def create_corpus(dataset: pd.DataFrame) -> list[str]:
    corpus = []
    for sample in dataset['passages']:
        passage_text = sample['passage_text']

        corpus.extend(passage_text)
    
    return corpus