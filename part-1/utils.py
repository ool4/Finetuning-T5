import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def get_related_words(word):
    synsets = wordnet.synsets(word)
    related = set()
    for syn in synsets[:5]:
        for h in syn.hypernyms() + syn.hyponyms():
            for lemma in h.lemmas():
                related.add(lemma.name().replace("_", " "))
        for lemma in syn.lemmas():
            related.add(lemma.name().replace("_", " "))
    return list(related)


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    sentence = example["text"]
    lst = sentence.split()
    
    for i in range(len(lst)):
        word = lst[i]
        lemmas =[]

        
        if random.random() > 0.35:
            continue
        else:

            #print(f"word: {word}, pos: {pos}")
            lemmas1 = get_related_words(word)
            if lemmas1:
                choice = random.choice(lemmas1)
                lemmas2 = get_related_words(choice)
                if lemmas2 and random.random() < 0.7:
                    choice2 = random.choice(lemmas2)
                    if choice2 != word:
                        choice = choice2
                else: 
                    if choice != word:
                        choice = choice
                lst[i] = choice
        
    example["text"] = " ".join(lst)
    

    ##### YOUR CODE ENDS HERE ######

    return example
