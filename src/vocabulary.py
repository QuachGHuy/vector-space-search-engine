from .tokenizer import tokenize

def create_vocabulary(corpus: list[str]) -> list[str]:
    vocabulary = set()

    for doc in corpus:
        vocabulary.update(tokenize(doc))
    
    return sorted(vocabulary)

if __name__ == "__main__":
    corpus = [
        "Machine learning is fun",
        "Deep learning is part of machine learning"
    ]

    vocab = create_vocabulary(corpus)
    print(vocab)