
"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailnments are poisitive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset
Usage:
python training_nli_v2.py
OR
python training_nli_v2.py pretrained_transformer_model_name

adpated from https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli_v2.py
"""
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random
import json

from utils import parse_args
args = parse_args()
SEED = 1

def get_description_data(base_path):
    with open(base_path[:base_path[:-1].rfind("/")] + "/segmentation/data/" + 'loc_description_dict.json', 'r') as infile:
        desc_dict = json.load(infile)
        desc_dict["START"] = "This is the beginning of the testimony."
        desc_dict["END"] = "This is the end of the testimony."

    train_data = {}
    for k, v in desc_dict.items():
        others = list(desc_dict.values())
        others.remove(v)
        k1 = f"The event location is {k}"
        if k1 not in train_data:
            train_data[k1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[k1]['entailment'].add(v)
        train_data[k1]['contradiction'].add(random.choice(others))
    return train_data

def get_location_data(base_path, c_dict=None, nba=False):
    data_path = base_path[:base_path[:-1].rfind("/")] + "/segmentation/data/"
    from loc_transformer import get_data, get_nba_data
    if not nba:
        data, _, _ = get_data(data_path=data_path, val_size=0.1, use_test=args.use_test)
    else:
        data, _, _ = get_nba_data(val_size=0.1, test_size=0.1)

    train_data = {}
    if c_dict is not None:
        locs_cats = set([(f"The event location is {s[1]}", f"The event location category is {c_dict[s[2][0]]}")
                         for l in data.values() for s in l if s[1] != ""])
        all_cats = set([f"The event location category is {c_dict[s[2][0]]}" for l in data.values() for s in l if s[2][0] != ""])
    elif not nba:
        locs_cats = set([(f"The event location is {s[1]}", f"The event location category is {s[2][0]}")
                         for l in data.values() for s in l if s[1] != ""])
        all_cats = set([f"The event location category is {s[2][0]}" for l in data.values() for s in l if s[2][0] != ""])
    else:
        locs_cats = set([f"The professional location is {s[1]}" for l in data.values() for s in l if s[1] != ""])

    if not nba:
        all_locs = set([f"The event location is {s[1]}" for l in data.values() for s in l if s[1] != ""])
        for t, l in data.items():
            for s in l:
                if s[1] != "":
                    s1 = f"The event location is {s[1]}"
                    others = list(all_locs)
                    others.remove(s1)
                    if s1 not in train_data:
                        train_data[s1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
                    train_data[s1]['entailment'].add(s[0])

                    # add a contradiction (for a different location)
                    other = random.choice(others)
                    if other not in train_data:
                        train_data[other] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
                    train_data[other]['contradiction'].add(s[0])
    else:
        for t, l in data.items():
            for s in l:
                s1 = f"The professional location is {s[1]}"
                others = list(locs_cats)
                others.remove(s1)
                if s1 not in train_data:
                    train_data[s1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
                train_data[s1]['entailment'].add(s[0])

                # add a contradiction (for a different location)
                other = random.choice(others)
                if other not in train_data:
                    train_data[other] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
                train_data[other]['contradiction'].add(s[0])


        return train_data

    for s1, s2 in locs_cats:
        other_cats = list(all_cats)
        other_cats.remove(s2)

        if s1 not in train_data:
            train_data[s1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[s1]['entailment'].add(s2)

        # add a contradiction (for a different category)
        other_cat = random.choice(other_cats)
        train_data[s1]['contradiction'].add(other_cat)
    return train_data

def get_nli_data(base_path):
    # Check if dataset exsist. If not, download and extract  it
    nli_dataset_path = base_path + 'data/AllNLI.tsv.gz'

    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    # Read the AllNLI.tsv.gz file and create the training dataset
    logging.info("Read AllNLI train dataset")

    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent1][label].add(sent2)

    train_data = {}
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'train':
                sent1 = row['sentence1'].strip()
                sent2 = row['sentence2'].strip()

                add_to_samples(sent1, sent2, row['label'])
                add_to_samples(sent2, sent1, row['label'])  # Also add the opposite
    return train_data


def get_train_data(base_path, types=('nli', ), conv_dict=True, nba=False):
    train_data = {}
    c_dict = None
    if conv_dict and not nba:
        print("Using conv_dict")
        from loc_clusters import make_loc_conversion
        c_dict = make_loc_conversion(data_path=base_path[:base_path[:-1].rfind("/")] + "/segmentation/data/")
    if 'nli' in types:
        train_data = {**train_data, **get_nli_data(base_path)}
    if 'locs' in types:
        train_data = {**train_data, **get_location_data(base_path, c_dict, nba=nba)}
    if 'desc' in types:
        train_data = {**train_data, **get_description_data(base_path)}

    train_samples = []
    num_per_loc = 1 if not nba else 500
    for sent1, others in train_data.items():
        for _ in range(num_per_loc):
            if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
                train_samples.append(InputExample(
                    texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
                train_samples.append(InputExample(
                    texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

    logging.info("Train samples: {}".format(len(train_samples)))
    return train_samples

def get_dev_data(base_path):
    # Check if dataset exsist. If not, download and extract  it
    sts_dataset_path = base_path + 'data/stsbenchmark.tsv.gz'

    # Read STSbenchmark dataset and use it as development set
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    return dev_samples


def train(base_path, load_model=False, lr=2e-5, types=('nli', ), nba=False):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    model_name = args.model
    train_batch_size = 4          #The larger you select this, the better the results (usually). But it requires more GPU memory
    max_seq_length = 75
    num_epochs = 1

    # Save path of the model
    model_save_path = base_path + 'output/training_nli_v2_'+model_name.replace("/", "-") + ("_nba" if nba else "")
    if "locs" in types:
        model_save_path = model_save_path + "_locs"

    if not load_model:
        # Here we define our SentenceTransformer model
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length, cache_dir=args.cache_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model_load_path = base_path + 'output/training_nli_v2_'+model_name.replace("/", "-") + ("_nba" if nba else "")
        model = SentenceTransformer(model_load_path)

    train_samples = get_train_data(base_path, types=types, nba=nba)
    dev_samples = get_dev_data(base_path)

    # Special data loader that avoid duplicates within a batch
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

    # Our training loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              # evaluator=dev_evaluator,
              epochs=num_epochs,
              evaluation_steps=int(len(train_dataloader)*0.1),
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              optimizer_params={'lr': lr},
              use_amp=False          #Set to True, if your GPU supports FP16 operations
              )

    # model.save(model_save_path)


def eval(base_path):
    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################
    model_name = args.model
    sts_dataset_path = base_path + 'data/stsbenchmark.tsv.gz'
    model_save_path = base_path + 'output/training_nli_v2_'+model_name.replace("/", "-")
    train_batch_size = 16

    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
    test_evaluator(model, output_path=model_save_path)


if __name__ == "__main__":

    base_path = args.base_path
    # data_path = base_path + 'data/'
    nba_data = args.bio_data
    print("SEED: ", args.seed)
    lr1 = args.lr1
    lr2 = args.lr1
    print("lr1: ")
    print(lr1)
    train(base_path, types=('nli', ), lr=lr1)
    print("lr2: ")
    print(lr2)
    train(base_path, lr=lr2, load_model=True, types=('locs'), nba=nba_data)
