
import logging
import spacy
import json
import numpy as np
import re
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
# from evaluation import edit_distance, gestalt_diff
from loc_evaluation import edit_distance, gestalt_diff
from sklearn.metrics import precision_recall_fscore_support
import os
import sys
import wandb

import torch
from torch import nn
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Tokenizer
from transformers import Trainer, TrainingArguments
# from transformer_override import MultiTrainer, DividedTrainer

from sentence_transformers import SentenceTransformer, util
from loc_clusters import find_closest
from loc_evaluation import get_gold_xlsx

import random
SEED = 1

import joblib
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dev2 = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu")
dev3 = torch.device("cuda:2") if torch.cuda.device_count() > 2 else torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dev4 = torch.device("cuda:3") if torch.cuda.device_count() > 3 else torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dev5 = torch.device("cuda:4") if torch.cuda.device_count() > 4 else torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dev6 = torch.device("cuda:5") if torch.cuda.device_count() > 5 else torch.device("cuda:2") if torch.cuda.device_count() > 2 else torch.device("cpu")
dev7 = torch.device("cuda:6") if torch.cuda.device_count() > 6 else torch.device("cuda:3") if torch.cuda.device_count() > 3 else torch.device("cpu")
dev8 = torch.device("cuda:7") if torch.cuda.device_count() > 6 else torch.device("cuda:4") if torch.cuda.device_count() > 4 else torch.device("cpu")

NUM_BINS = 10
# SOFTMAX = True
SOFTMAX = False
print(f"num bins: {NUM_BINS}")
print(f"SOFTMAX: {SOFTMAX}")

# import enum
# class Times(enum.Enum):
BEFORE = 0
AFTER_RISE = 1
BEFORE_INVASION = 2  # but after the war broke out
BEFORE_EXTERMINATION = 3  # but after the invasion to this country
DURING_WAR = 4
AFTER_WAR = 5

from utils import parse_args
args = parse_args()


def is_visit(terms, terms_df):
    """
    Checks whether this segment describes a visit
    """
    for t in terms:
        row = terms_df[terms_df['Label'] == t].squeeze().to_dict()
        if len(row['Sub-Type']) > 0 and row['Sub-Type'].find("returns and visits") >= 0:
            return True
    return False


def extract_loc(terms, terms_df, return_cat=False, return_country=False):
    """
    Convert the list of terms for a segment into a list of relevant locations
    """
    _terms = []
    loc_cats = ["cities in", "kibbutzim", "moshavim in", "German concentration camps in", "German death camps in", "displaced persons camps or",
                "refugee camps", "ghettos in", "administrative units in", "Croatian concentration camps in",
                "German prisoner of war camps in", "Slovakian concentration camps in", "Soviet concentration camps in",
                "Hungarian concentration camps in", "Romanian concentration camps in", "Polish concentration camps in", "Cambodian camps",
                "internment camps in", "concentration camps in"]

    cat = ""

    country_cats = ["periodizations by country", "countries"]
    _country = ""
    c_cat = ""

    for t in terms:
        # for bad characters
        end = t.find("?")
        end2 = end + t[end+1:].find("?") + 1
        end3 = end2 + t[end2+1:].rfind("?") + 1

        # if end2 == 0:  # starts with ??
        #     print("bad term!!!!!")
        if end == -1:  # no '?'
            row = terms_df[terms_df['Label'] == t]
        elif end2 == end:  # only one '?'
            row = terms_df[pd.Series([_t[:end] + _t[end+1:] for _t in terms_df['Label']])
                           == t[:end] + t[end+1:]]
        elif end3 == end2:  # two '?'
            row = terms_df[pd.Series([_t[:end] + _t[end+1:] + _t[end2+1:] for _t in terms_df['Label']])
                           == t[:end] + t[end+1:] + t[end2+1:]]
        else:  # more than two '?'
            row = terms_df[pd.Series([_t[:end] + _t[end+1:end2] + _t[end3+1:] + str(_t.count("?")) for _t in terms_df['Label']])
                           == t[:end] + t[end+1:end2] + t[end3+1:] + str(t.count("?"))]
        if row.size == 0:
            row = row.squeeze().to_dict()
        else:
            row = row.iloc[0].squeeze().to_dict()

        if len(row['Label']) > 0:
            for _c in loc_cats:
                # if row['Sub-Type'][:len(_c)] == _c:
                if row['Sub-Type'].find(_c) > -1:
                    _terms.append(row['Label'])
                    if cat == "":  # to take the first
                        # cat = _c
                        cat = row['Sub-Type']
            for _c in country_cats:
                if row['Sub-Type'][:len(_c)] == _c:
                    _country = row['Label']
                    if c_cat == "":  # to take the first
                        c_cat = _c
                    # print("!!!!!" + row['Label'])
        # print(row['Sub-Type'])
        # print(row['Label'])

    # returns only first
    if len(_terms) > 0 or len(c_cat) > 0:
        if len(_terms) == 0:
            _terms.append("")
        if return_cat:
            if not return_country:
                return _terms[0], cat
            return _terms[0], [cat, _country, c_cat]
        return _terms[0]
    if return_cat:
        if not return_country:
            return None, ""
        return None, ["","",""]
    return None

def make_loc_data(data_path, use_segments=True, with_cat=False, with_country=False):
    """
    Make location data from raw data (in path)
    """
    print("Starting")
    terms_df = pd.read_csv(data_path + "All indexing terms 4 April 2022.xlsx - vhitm_data (14).csv", encoding='utf-8')
    # terms_df = pd.read_csv(data_path + "All indexing terms 4 April 2022.xlsx - vhitm_data (14).csv", header=1)

    with open(data_path + 'sf_all.json', 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    with open(data_path + 'sf_unused5.json', 'r') as infile:
        unused = json.load(infile) + [45064]
        # unused = []

    loc_data = {}
    for t, d in data.items():
        visit = False
        print(t)
        last_loc = ""
        last_seg = ''
        cat = ""
        last_cat = ""
        if with_country:
            last_cat = [""]
        cs = []
        last_cs = []
        if int(t) not in unused:
            loc_data[t] = []
            if not use_segments:
                for s in d:
                    if with_cat:
                        current_loc, cat = extract_loc(s['terms'], terms_df=terms_df, return_cat=True, return_country=with_country)
                        if with_country:
                            c = cat[1]
                            # last_cs.append(c)
                    else:
                        current_loc = extract_loc(s['terms'], terms_df=terms_df, return_cat=False)
                    if current_loc is not None and current_loc != "" and current_loc != last_loc:
                        if with_cat:
                            if with_country:
                                loc_data[t].append([last_seg, last_loc, last_cat[0], list(set(last_cs) - {""})])
                                last_cs = []
                                # cs = []
                            else:
                                loc_data[t].append([last_seg, last_loc, last_cat])
                            last_cat = cat
                        else:
                            loc_data[t].append([last_seg, last_loc])
                        last_loc = current_loc
                        last_seg = ""
                    elif visit:
                        if with_cat:
                            if len(loc_data[t]) > 0:
                                if not with_country:
                                    current_loc = loc_data[t][-1][-2]  # last location added before the visit
                                else:
                                    current_loc = loc_data[t][-1][-3]
                            else:
                                if with_country:
                                    current_loc = ["","",""]
                                else:
                                    current_loc = ""
                            if with_country:
                                loc_data[t].append([last_seg, last_loc, last_cat[0], list(set(last_cs) - {""})])
                                last_cs = []
                                # cs = []
                            else:
                                loc_data[t].append([last_seg, last_loc, last_cat])
                            # last_cat = cat
                            last_cat = [loc_data[t][-1][-2]]
                        else:
                            if len(loc_data[t]) > 0:
                                current_loc = loc_data[t][-1][-1]  # last location added before the visit
                            else:
                                current_loc = ""
                            loc_data[t].append([last_seg, last_loc])

                        last_seg = ""
                        last_loc = current_loc
                    visit = is_visit(s['terms'], terms_df=terms_df)
                    last_seg = last_seg + s['text']
                    last_cs.append(c)
                if with_cat:
                    if with_country:
                        loc_data[t].append([last_seg, last_loc, last_cat[0], list(set(last_cs) - {""})])
                        # last_cs = []
                        # cs = []
                    else:
                        loc_data[t].append([last_seg, last_loc, last_cat])
                    # last_cat = cat
                else:
                    loc_data[t].append([last_seg, last_loc])
            else:
                for s in d:
                    if with_cat:
                        current_loc, cat = extract_loc(s['terms'], terms_df=terms_df, return_cat=True, return_country=with_country)
                    else:
                        current_loc = extract_loc(s['terms'], terms_df=terms_df, return_cat=False)
                    if current_loc is None:
                        current_loc = ""
                    if with_cat:
                        loc_data[t].append([s['text'], current_loc, cat, "visit" if is_visit(s['terms'], terms_df=terms_df) else ""])
                    else:
                        loc_data[t].append([s['text'], current_loc, "visit" if is_visit(s['terms'], terms_df=terms_df) else ""])

    if use_segments:
        with open(data_path + 'locs_segments_w_cat2.json', 'w') as outfile:
            json.dump(loc_data, outfile)
    else:
        with open(data_path + 'locs_w_cat2.json', 'w') as outfile:
            json.dump(loc_data, outfile)


def make_description_category_dict(data_path, use_test=False):
    """
    Makes dictionaries with the descriptions and categories for each location mentioned
    :param data_path:
    :return:
    """
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
    all_locs = set([_v[1] for v in data.values() for _v in v])
    if use_test:
        d = get_gold_xlsx(cat_dict=None, converstion_dict=None)
        all_locs |= set([l.rstrip() for v in d.values() for l in v[1]])
    terms_df = pd.read_csv(data_path + "All indexing terms 4 April 2022.xlsx - vhitm_data (14).csv", encoding='utf-8')
    desc_dict = {}
    cat_dict = {}
    for t in all_locs:
        if t not in ["", "START", "END", "New York (USA)"]:
            row = terms_df[terms_df['Label'] == t]
            desc_dict[t] = row['Definition'].to_list()[0]
            if row.isnull().values.any():
                desc_dict[t] = ""
            cat_dict[t] = row['Sub-Type'].to_list()[0]

    with open(data_path + 'loc_description_dict.json', 'w') as outfile:
        json.dump(desc_dict, outfile)
    with open(data_path + 'loc_category_dict.json', 'w') as outfile:
        json.dump(cat_dict, outfile)


def make_spreadsheets(data, out_path, conversion_dict=None, ner=False, use_bins=False):
    new_data = {}
    for t, t_data in data.items():
        texts, labels = _make_texts({t: t_data}, [], out_path, conversion_dict=conversion_dict, ners=ner, use_bins=use_bins, desc_dict={})
        _, labels2 = _make_texts({t: t_data}, [], out_path, conversion_dict=conversion_dict, ners=ner, use_bins=use_bins)
        texts = [t_data[0][0]] + texts
        labels = [t_data[0][1]] + labels
        labels2 = [t_data[0][2][0]] + labels2
        new_data[t] = [list(texts), list(labels), list(labels2)]

    for j in range(6):
        with pd.ExcelWriter(args.base_path + "data/location_lists/" + f"testimonies{j}_full.xlsx") as writer:
            for i, (t, t_data) in enumerate(new_data.items()):
                if 100 * j < i < 100 * (j+1):
                    df = pd.DataFrame({"text": t_data[0], "location": t_data[1], "location category": t_data[2]})
                    df.to_excel(writer, sheet_name=str(t))

# ************** train model
import torch

class Dataset(torch.utils.data.Dataset):
    """
    Dataset object
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class Dataset1(torch.utils.data.Dataset):
    """
    Dataset object
    """
    def __init__(self, data_path, labels):
        # make sure the order is correct!!
        self.data_files = os.listdir(data_path)
        self.data_files = sorted(self.data_files)
        self.labels = labels

    def __getitem__(self, idx):
        # encoded_data = []
        # for i in idx:
        with open(self.data_files[idx], 'r') as infile:
            # encoded_data.append(json.load(infile))
            item = json.load(infile)  # assumes one per batch
        # return load_file(self.data_files[idx])
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def _add_loc_description(labels, desc_dict):
    return [l + ". " + desc_dict[l] for l in labels], labels


def _add_locs(data, sep_token='[SEP]'):
    import spacy
    spacy.require_gpu()
    nlp = spacy.load(args.base_path + "ner/model-best")

    for t, d in tqdm.tqdm(data.items()):
        for _d in d:
            doc = nlp(_d[0])
            _d[0] = f" {sep_token} ".join([doc.text] + [e.text + " " + e.label_ for e in doc.ents])
    return data

    # doc = nlp(text)
    # return f" {sep_token} ".join([text] + [e.text + " " + e.label_ for e in doc.ents])

def _make_texts(data, unused=None, out_path="", desc_dict=None, conversion_dict=None, vectors=None, ners=False,
                sep_token=" ", use_bins=False, matrix=False, nba_data=False, reverse=False):
    """
    Make a list of texts with labels
    :param data:
    :param unused:
    :param out_path:
    :param desc_dict: whether uses location descriptions. In this case the labels are the original detailed locations
    :return:
    """
    if nba_data:
        # from itertools import chain
        # TODO: matrix with nba data!
        texts = [t[0] for v in data.values() for t in v]
        labels = [t[1] for v in data.values() for t in v]
        if matrix:
            _labels = [[t[1] for t in v] for v in data.values()]
            label_vectors = [[l_v, _l[i+1]]
                             for _l in _labels for i, l_v in enumerate(_l[:-1])]
            return texts, label_vectors
        return texts, labels

    c_dict = conversion_dict if conversion_dict is not None else {}

    if out_path.split("/")[-1].find("deberta") >= 0:
        sep_token = '[SEP]'
    else:
        sep_token = '</s>'

    if out_path.find("_all") >= 0 or out_path.find("gpt") >= 0:
        desc_dict = {}
    texts = []
    labels = []
    for t, t_data in data.items():
        prev_text = ""
        if reverse:
            t_data = t_data[::-1]
            prev_loc = ["END", "END"]
            last = ["START", "START", ["START"], ""]
        else:
            prev_loc = ["START", "START"]
            last = ["END", "END", ["END"], ""]
        if t in unused:
            continue
        for i, d in enumerate(t_data + [last]):
            if d[1] == ["", "", ""] or d[1] == "":
                if i > 0 and t_data[i-1][3] == "visit":  # if visit then the prevs stay
                    d[2][0] = labels[-2]
                d[1] = prev_loc[-1]
                if i <= 1:
                    d[2][0] = "START"
                else:
                    d[2][0] = labels[-1]
            if i == 0:
                prev_loc = ["START"] + [d[1]]
                prev_text = d[0]
            if i > 0:
                if desc_dict is None:
                    labels.append(c_dict.get(d[2][0], d[2][0]))
                else:
                    labels.append(d[1])
                text = d[0]
                if use_bins:
                    text = str((NUM_BINS * i) // len(t_data)) + ": " + text
                if out_path[-1] == "1":  # deberta1
                    texts.append(f" {sep_token} ".join([text]))
                elif out_path[-1] == "2":  # deberta2
                    texts.append(f" {sep_token} ".join([prev_loc[0], prev_text, prev_loc[1]]))
                elif out_path[-1] == "3":  # deberta3
                    texts.append(f" {sep_token} ".join([prev_loc[0], prev_loc[1]]))
                elif out_path[-1] == "4" or out_path.split("/")[-1][:6] == "distil":  # deberta4 or distilroberta
                    texts.append(f" {sep_token} ".join([prev_text, text]))
                elif out_path[-1] == "5" and desc_dict is not None:  # from loc to loc with description
                    texts.append(f" {sep_token} ".join([prev_loc[0] + ": " + desc_dict[prev_loc[0]],
                                                 prev_loc[1] + ": " + desc_dict[prev_loc[1]]]))
                elif out_path[-1] == "6":
                    if vectors is not None or matrix:
                        texts.append(f" {sep_token} ".join([prev_text]))
                    elif desc_dict is not None:  # from prev to loc, with label
                        if len(labels) < 2:
                            texts.append(f" {sep_token} ".join([prev_text, "START"]))
                        else:
                            texts.append(f" {sep_token} ".join([prev_text, labels[-2]]))
                    else:
                        texts.append(prev_text + " location: " + prev_loc[1])

                else:
                    texts.append(f" {sep_token} ".join([prev_loc[0], prev_text, prev_loc[1], text]))

                prev_text = text
                prev_loc = [prev_loc[-1], d[1]]
    if vectors is not None or matrix:
        if vectors is not None:
            v_dict = {l:v for l, v in zip(*vectors)}
            label_vectors = [v_dict[l] for l in labels]
        if out_path[-1] == "6":
            e = np.where(np.array(labels) == "END")[0] + 1
            s = np.insert(e, 0, 0)[:-1]
            if vectors is not None:
                _label_vectors = [[v_dict[l] for l in labels[_s:_e]] for _s, _e in zip(s, e)]
                label_vectors = [np.concatenate([l_v, _l[i]])
                                 for _l in _label_vectors for i, l_v in enumerate([v_dict["START"]] + _l[:-1])]
            else:
                _labels = [labels[_s:_e] for _s, _e in zip(s, e)]
                label_vectors = [[l_v, _l[i]]
                                 for _l in _labels for i, l_v in enumerate(["START"] + _l[:-1])]

            return texts, label_vectors
        return texts, label_vectors
    return texts, labels


class MatrixTrainer(Trainer):
    """
    Trainer class for matrix prediction
    """
    def __init__(self, vectors=None, v_scales=False, w_scales=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_scales = v_scales
        self.w_scales = w_scales
        self.vectors = torch.from_numpy(vectors)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')  # of shape (batch_size, dim)?
        if self.w_scales:
            weights = logits[:, -len(self.vectors[0]):]
            logits = logits[:, :-len(self.vectors[0])]

        # out is a tensor of shape (batch_size, vec_len)
        out = torch.bmm(logits.reshape(-1, int(logits.shape[-1] ** .5), int(logits.shape[-1] ** .5)),
                        labels[:, :int(logits.shape[-1] ** .5)].unsqueeze(-1)).squeeze(-1)
        # sims should be a tensor of shape (batch_size, num_classes)
        if self.v_scales:
            out_sims = util.dot_score(out, self.vectors.to(logits.device))
        elif self.w_scales:
            # bi, bi -> bi; bi, ij -> bj
             out_sims = out * weights @ self.vectors.to(logits.device).T
        else:
            out_sims = util.cos_sim(out, self.vectors.to(logits.device))

        labels = torch.stack([find_closest(vectors=self.vectors.to(logits.device),
                      c_vector=l[int(logits.shape[-1] ** .5):], tensor=True, v_scales=self.v_scales) for l in labels])
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(out_sims.softmax(dim=-1), labels)
        return (loss, outputs) if return_outputs else loss

class VectorTrainer(Trainer):
    def __init__(self, vectors=None, v_scales=False, w_scales=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_scales = v_scales
        self.w_scales = w_scales
        self.vectors = torch.from_numpy(vectors)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')  # of shape (batch_size, vec_dim)?
        if self.w_scales:
            weights = logits[:, -len(self.vectors[0]):]
            logits = logits[:, :-len(self.vectors[0])]

        # should be of shape (batch_size, num_labels)
        if self.v_scales:
            # out_sims = util.cos_sim(logits, self.vectors.to(logits.device)) * ((logits**2).sum(dim=-1, keepdims=True)**.5)
            out_sims = util.dot_score(logits, self.vectors.to(logits.device))
        elif self.w_scales:
            # bi, bi -> bi; bi, ij -> bj
            out_sims = logits * weights @ self.vectors.to(logits.device).T
        else:
            out_sims = util.cos_sim(logits, self.vectors.to(logits.device))

        # label_sims = torch.zeros_like(out_sims)
        # label_sims[torch.arange(logits.shape[0]),
        #            [find_closest(vectors=self.vectors.to(logits.device), c_vector=l) for l in labels]] = 1.

        labels = torch.stack([find_closest(vectors=self.vectors.to(logits.device), c_vector=l, tensor=True, v_scales=self.v_scales) for l in labels])

        loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(out_sims.softmax(dim=-1), label_sims)

        # loss = loss_fct(out_sims.softmax(dim=-1), labels)
        loss = loss_fct(out_sims, labels)
        return (loss, outputs) if return_outputs else loss


from sklearn.preprocessing import LabelEncoder
class LabelEncoder2(LabelEncoder):
    """
    A LabelEncoder with matrix capabilities
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def transform2(self, y, from_number=False):
        """
        transform transition labels. This transforms pairs of labels to numbers
        :param y:
        :return:
        """
        if from_number:
            return [np.ravel_multi_index(_y, (len(self.classes_), len(self.classes_))) for _y in y]
        return [np.ravel_multi_index(self.transform(_y), (len(self.classes_), len(self.classes_))) for _y in y]

    def inverse_transform2(self, y):
        """
        from number to pair of label indices
        :param y:
        :return:
        """
        return [np.unravel_index(_y, (len(self.classes_), len(self.classes_))) for _y in y]

def train_classifier(data_path, return_data=False, out_path=None, first_train_size=0.4, val_size=0.1, test_size=0.1,
                     conversion_dict=None, vectors=None, matrix=False, v_scales=False, w_scales=False, wd=0.01, ner=False,
                     use_bins=False, multihead=False, conditional=False, divided=False,
                     use_test=False, nba_data=False, reverse=False):
    """
    Function for training the classifiers
    """
    from sklearn.model_selection import train_test_split
    from transformers import Trainer, TrainingArguments
    from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
    from transformers import LukeTokenizer, LukeForSequenceClassification
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from sklearn.metrics import accuracy_score
    from datasets import load_metric
    import joblib

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("\n********* Training model on the GPU ***************")
    else:
        dev = torch.device("cpu")
        print("\nTraining model on the CPU")

    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        print("Predictions: ")
        print(predictions)
        print(logits[200:400])
        with open(args.base_path + f'logits1b.npy', 'wb') as f:
            np.save(f, logits)
        with open(args.base_path + f'predsb.npy', 'wb') as f:
            np.save(f, predictions)
        return metric.compute(predictions=predictions, references=labels)

    def compute_metrics2(eval_pred):
        vs, labels = eval_pred
        if w_scales:
            vs = [v[:-len(v)//2] * v[len(v)//2:] for v in vs]
        predictions = [find_closest(vectors[0], vectors[1], v, v_scales=v_scales) for v in vs]
        eval_labels = [find_closest(vectors[0], vectors[1], v, v_scales=v_scales) for v in labels]
        return metric.compute(predictions=predictions, references=eval_labels)

    def compute_metrics2_2(eval_pred):
        logits, labels = eval_pred
        _labels = encoder.inverse_transform2(labels)
        predictions = [np.argmax(_logits.reshape(len(encoder.classes_), -1)[_l[0]]) for _logits, _l in zip(logits, _labels)]
        return metric.compute(predictions=predictions, references=list(zip(*_labels))[1])

    def compute_metrics3(eval_pred):
        with torch.no_grad():
            vs, labels = eval_pred
            if w_scales:
                weights = [v[-len(vectors[1][0]):] for v in vs]
                vs = [v[:-len(vectors[1][0])] for v in vs]
                # vs = [v[:-len(v)//2] * v[len(v)//2:] for v in vs]
                # TODO
                _vs = [v.reshape(int(len(v)**.5), -1) @ l[:len(l)//2] * w for v, w, l in zip(vs, weights, labels)]
            else:
                _vs = [v.reshape(int(len(v)**.5), -1) @ l[:len(l)//2] for v, l in zip(vs, labels)]
            predictions = [find_closest(vectors[0], vectors[1], v, v_scales=v_scales) for v in _vs]
            eval_labels = [find_closest(vectors[0], vectors[1], l[len(l)//2:], v_scales=v_scales) for l in labels]
            return metric.compute(predictions=predictions, references=eval_labels)

    # TODO: nba
    if not nba_data:
        with open(data_path + "sf_unused5.json", 'r') as infile:
            unused = json.load(infile) + ['45064']
        with open(data_path + "sf_five_locs.json", 'r') as infile:
            unused = unused + json.load(infile)
        if use_test:
            unused = unused + list(get_gold_xlsx().keys())

        with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
            data = json.load(infile)
            for u in unused:
                data.pop(u, None)
            _l_data = list(data.items())
            random.seed(SEED)
            random.shuffle(_l_data)
            random.seed()
            data = dict(_l_data)

            if ner:
                print("adding NERs")
                _add_locs(data, "[SEP]" if out_path.split('/')[-1].find("deberta") >= 0 else "</s>")

            if vectors is None:
                _, all_labels = _make_texts(data, [], out_path, conversion_dict=conversion_dict, ners=ner, use_bins=use_bins)
                if use_test:  # TODO: what about all_locs?
                    with open(data_path + 'loc_category_dict.json', 'r') as infile:
                        cat_dict = json.load(infile)
                        cat_dict["START"] = "START"
                        cat_dict["END"] = "END"
                    test_data = get_gold_xlsx(data_path + "gold_loc_xlsx/", converstion_dict=conversion_dict,
                                              cat_dict=cat_dict)
                    all_labels = all_labels + [l for t_data in test_data.values() for l in t_data[1]]
            if test_size > 0 and not use_test:
                train_data = {t: text for t, text in list(data.items())[:int(first_train_size * len(data))][:]}
                val_data = {t: text for t, text in list(data.items())[-int(test_size * len(data))-int(val_size * len(data)):
                                                                      -int(test_size * len(data))][:]}
            elif use_test:
                train_data = {t: text for t, text in list(data.items())[:- int(val_size * len(data))]}
                # this is the val_data
                val_data = {t: text for t, text in list(data.items())[-int(val_size * len(data)):]}
            else:
                train_data = data
            print(f"Training on {len(train_data)} documents")

        train_texts, train_labels = _make_texts(train_data, unused, out_path, conversion_dict=conversion_dict,
                                                vectors=vectors, ners=ner, use_bins=use_bins, matrix=matrix, reverse=reverse)

        encoder = LabelEncoder2()
        val_texts, val_labels = _make_texts(val_data, unused, out_path, conversion_dict=conversion_dict,
                                            vectors=vectors, ners=ner, use_bins=use_bins, matrix=matrix, reverse=reverse)

        if vectors is None:
            encoder.fit(all_labels)
            joblib.dump(encoder, out_path + '/label_encoder.pkl')
            if not matrix:
                val_labels = encoder.transform(val_labels)
                train_labels = encoder.transform(train_labels)
            else:
                val_labels = encoder.transform2(val_labels)
                train_labels = encoder.transform2(train_labels)
                # val_labels0 = encoder.transform([l[0] for l in val_labels])
                # train_labels0 = encoder.transform([l[0] for l in train_labels])

            print(f"{len(encoder.classes_)} labels")
            class_weights = 1 - np.array([np.count_nonzero(train_labels == l) for l in range(len(encoder.classes_))]) / len(train_labels)

        if multihead or divided:
            from loc_clusters import make_loc_multi_conversion
            c_dict = make_loc_multi_conversion(data_path)
            label_sets = [list(set(l)) for l in list(zip(*c_dict.values()))]
            encoders = []
            for ls in label_sets:
                encoders.append(LabelEncoder2())
                encoders[-1].fit(ls)
            joblib.dump(encoders, out_path + '/label_encoders.pkl')
            conversion_dict = {encoder.transform([l])[0]: [_e.transform([_ls])[0] for _e, _ls in zip(encoders, ls)] for l, ls in c_dict.items()}

            if not matrix:
                mh_train_labels = list(zip(*[conversion_dict[l] for l in train_labels]))
                mh_class_weights = [1 - np.array([np.count_nonzero(np.array(tl) == l) for l in range(len(e.classes_))]) / len(train_labels) for tl, e in zip(mh_train_labels, encoders)]

                mh_class_weights1 = [1 - np.array([np.count_nonzero(np.array(mh_train_labels[0]) == l) for l in range(len(encoders[0].classes_))]) / len(train_labels)]

                from sklearn.utils.class_weight import compute_class_weight
                mh_class_weights2 = [compute_class_weight('balanced', classes=np.unique(mh_train_labels[0]), y=mh_train_labels[0])]

                i_labels = []
                for i in range(len(encoders[0].classes_)):
                    i_labels.append([int(l1) for l0, l1 in zip(*mh_train_labels) if l0 == i])
                    # i_labels[-1].append(0)
                    mh_class_weights1.append(1 - np.array([np.count_nonzero(np.array(i_labels[-1]) == l) for l in range(len(encoders[1].classes_))]) / len(i_labels[-1]))
                    if len(i_labels[-1]) == 0:
                        mh_class_weights2.append([1.])
                    else:
                        mh_class_weights2.append(compute_class_weight('balanced', classes=np.unique(i_labels[-1]), y=i_labels[-1]))
                i_labels = [list(set(il)) for il in i_labels]
                mh_class_weights1[1:] = [mh_c[mh_c != 1] for mh_c in mh_class_weights1[1:]]

                # uses only the train labels but it's probably the same!!
                with open(data_path + '/i_labels.json', 'w') as outfile:
                    json.dump(i_labels, outfile)


            if matrix:
                conversion_dict2 = {int(encoder.transform2([(l1, l2)])[0]):
                                        [int(encoders[0].transform2([(ls1[0], ls2[0])])[0]), int(encoders[1].transform2([(ls1[1], ls2[1])])[0])]
                                    for l1, ls1 in c_dict.items() for l2, ls2 in c_dict.items()}

                i_labels_m = [[] for _ in range(len(encoders[0].classes_) ** 2)]
                for k, p in conversion_dict2.items():
                    pair = encoders[0].inverse_transform2([p[0]])[0]
                    if not (pair[0] == 0) and not (pair[1] == 4 and pair[0] != 4):
                        i_labels_m[int(p[0])].append(int(p[1]))

                with open(data_path + '/i_labels_m.json', 'w') as outfile:
                    json.dump(i_labels_m, outfile)
                with open(data_path + '/c_dict2.json', 'w') as outfile:
                    json.dump(conversion_dict2, outfile)

        if not matrix:
            print("class weights")
            if vectors is None:
                print(class_weights)
            if multihead:
                print(mh_class_weights)
            print("made data")
    else:
        from loc_transformer import get_nba_data
        train_data, val_data, test_data = get_nba_data()
        _, train_labels = _make_texts(data=train_data, nba_data=True, matrix=False)
        _, val_labels = _make_texts(data=val_data, nba_data=True, matrix=False)
        _, test_labels = _make_texts(data=test_data, nba_data=True, matrix=False)
        all_labels = train_labels + val_labels + test_labels
        train_texts, train_labels = _make_texts(data=train_data, nba_data=True, matrix=matrix)
        val_texts, val_labels = _make_texts(data=val_data, nba_data=True, matrix=matrix)
        encoder = LabelEncoder2()

        if vectors is None:
            encoder.fit(all_labels)
            joblib.dump(encoder, out_path + '/label_encoder.pkl')
            if not matrix:
                val_labels = encoder.transform(val_labels)
                train_labels = encoder.transform(train_labels)
            else:
                val_labels = encoder.transform2(val_labels)
                train_labels = encoder.transform2(train_labels)

    print(f"*************** {out_path.split('/')[-1]} **************")

    distilroberta = out_path.split('/')[-1].find("distil") >= 0
    albert = out_path.split('/')[-1].find("albert") >= 0
    minilm = out_path.split('/')[-1].find("minilm") >= 0
    luke = out_path.split('/')[-1].find("luke") >= 0
    deberta = out_path.split('/')[-1].find("deberta") >= 0
    gpt = out_path.split('/')[-1].find("gpt") >= 0

    if distilroberta:
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", cache_dir=args.cache_dir)
    elif albert:
        tokenizer = AutoTokenizer.from_pretrained('albert-large-v2', cache_dir=args.cache_dir)
    elif minilm:
        tokenizer = AutoTokenizer.from_pretrained('nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large', cache_dir=args.cache_dir)
    elif luke:
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", cache_dir=args.cache_dir)
    elif gpt:
        tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=args.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir=args.cache_dir)

    train_encodings = tokenizer(train_texts[:], truncation=True, padding=True)
    val_encodings = tokenizer(val_texts[:], truncation=True, padding=True)
    print("made encodings")

    train_dataset = Dataset(train_encodings, train_labels[:])
    val_dataset = Dataset(val_encodings, val_labels[:])


    if return_data:
        return train_dataset, val_dataset

    lr = 5e-5
    print("Learning rate: ")
    print(lr)

    training_args = TrainingArguments(
        output_dir='/cs/labs/oabend/eitan.wagner/checkpoints/results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        learning_rate=lr,
        per_device_train_batch_size=4 if not minilm else 16,  # batch size per device during training
        per_device_eval_batch_size=4 if not minilm else 16,   # batch size for evaluation
        gradient_accumulation_steps=1 if not minilm else 1,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=wd,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",

    )

    p_type = "single_label_classification" if vectors is None else None
    n_labels = (len(encoder.classes_) if not matrix else len(encoder.classes_) **2) if vectors is None else \
        len(vectors[1][0]) if not matrix else len(vectors[1][0]) ** 2
    if divided:
        n_labels = sum([len(e.classes_) for e in encoders])
        if matrix:
            n_labels = sum([len(e.classes_) ** 2 for e in encoders])
    if w_scales:
        n_labels = n_labels + len(vectors[1][0])
    if not multihead or divided:
        if out_path.split('/')[-1].find("distil") >= 0:
            model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base",
                                                                       cache_dir=args.cache_dir,
                                                                       num_labels=n_labels, problem_type=p_type)
        elif albert:
            model = AutoModelForSequenceClassification.from_pretrained('albert-large-v2',
                                                                       cache_dir=args.cache_dir,
                                                                       num_labels=n_labels, problem_type=p_type)
        elif minilm:
            model = AutoModelForSequenceClassification.from_pretrained('nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large',
                                                                       cache_dir=args.cache_dir,
                                                                       num_labels=n_labels, problem_type=p_type)
        elif luke:
            model = LukeForSequenceClassification.from_pretrained("studio-ousia/luke-base",
                                                                  cache_dir=args.cache_dir,
                                                                  num_labels=n_labels, problem_type=p_type)
        elif gpt:
            model = AutoModelForSequenceClassification.from_pretrained("gpt2",
                                                                       cache_dir=args.cache_dir,
                                                                       num_labels=n_labels, problem_type=p_type)
            model.config.pad_token_id = model.config.eos_token_id
        else:
            model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base",
                                                                       cache_dir=args.cache_dir,
                                                                       num_labels=n_labels, problem_type=p_type)

    if multihead:
        from transformer_override import MultiTrainer, DividedTrainer, \
            LukeForSequenceMultiClassification, DebertaForSequenceMultiClassification
        from transformers import AutoModel

        # reverse_dict = {tuple(ls): l for l, ls in c_dict.items()}
        if not matrix:
            reverse_dict = {tuple(ls): l for l, ls in conversion_dict.items()}
        else:
            reverse_dict = {tuple(ls): l for l, ls in conversion_dict2.items()}

        if luke or deberta and not divided:
            if conditional:
                sizes = [len(encoders[0].classes_)] + [len(encoders[1].classes_)] * len(encoders[0].classes_) if not matrix \
                    else [len(encoders[0].classes_) ** 2] + [len(encoders[1].classes_) ** 2] * len(encoders[0].classes_)
                sizes1 = [len(encoders[0].classes_)] + [len(i_labels[i]) for i in range(len(encoders[0].classes_))]
            else:
                sizes1 = []
                sizes = [len(e.classes_) for e in encoders] if not matrix else [len(e.classes_) ** 2 for e in encoders]
            if luke:
                _model = AutoModel.from_pretrained("studio-ousia/luke-base", cache_dir=args.cache_dir)
                model = LukeForSequenceMultiClassification(pretrained_model=_model, reverse_dict=reverse_dict,
                                                       sizes=sizes1 if conditional else sizes, i_labels=i_labels,
                                                       size2=len(encoders[1].classes_), conditional=conditional)
            elif deberta:
                _model = AutoModel.from_pretrained("microsoft/deberta-base", cache_dir=args.cache_dir)
                model = DebertaForSequenceMultiClassification(pretrained_model=_model, reverse_dict=reverse_dict,
                                                       sizes=sizes1 if conditional else sizes, i_labels=i_labels,
                                                       size2=len(encoders[1].classes_), conditional=conditional)
        else:
            model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base",
                                                                       cache_dir=args.cache_dir,
                                                                       num_labels=n_labels, problem_type=p_type)

    model.to(dev)
    print("Training")

    if vectors is None:
        if not multihead:
            _Trainer = Trainer
            trainer = _Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                train_dataset=train_dataset,  # training dataset
                eval_dataset=val_dataset,  # evaluation dataset
                compute_metrics=compute_metrics if not matrix else compute_metrics2_2,
            )
        else:
            if divided:
                sizes = [len(e.classes_) for e in encoders] if not matrix else [len(e.classes_) ** 2 for e in encoders]
                _Trainer = DividedTrainer
            else:
                _Trainer = MultiTrainer
            trainer = _Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                train_dataset=train_dataset,  # training dataset
                eval_dataset=val_dataset,  # evaluation dataset
                compute_metrics=compute_metrics,
                lengths=sizes,
                conversion_dict=conversion_dict if not matrix else conversion_dict2,
            )
            if divided:
                trainer.compute_metrics = trainer.compute_double_metrics
                trainer.matrix = matrix
            else:
                trainer.mh_class_weights=mh_class_weights2 if conditional else mh_class_weights

    elif not matrix:
        _Trainer = VectorTrainer
        trainer = _Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics2,
            vectors=vectors[1],
            v_scales=v_scales,
            w_scales=w_scales,
        )
    else:
        _Trainer = MatrixTrainer
        trainer = _Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics3,
            vectors=vectors[1],
            v_scales=v_scales,
            w_scales=w_scales,
        )

    print(_Trainer.__name__)
    trainer.train()
    model.save_pretrained(out_path)

    # if return_data:
    return train_dataset, val_dataset


def preprocess_examples(texts, tokenizer, full_labels, data_path):
    # Check that it can be done in one shot!
    batch_size = 1
    _batch = 0
    encoded_data = {}
    for j in range(0, len(texts), batch_size):
        _batch += 1
        try:
            with open(data_path + f'/batch{_batch}.json', 'r') as infile:
                encoded_data = json.load(infile)
                continue
        except IOError as err:
            continue
            pass

        print("#batch: ", _batch)
        sys.stdout.flush()

        _texts = texts[j: j + batch_size]
        first_sentences = [[text] * len(full_labels) for text in _texts]
        header = "The current location is: "
        second_sentences = [[f"{header}{fl}" for fl in full_labels] for _ in _texts]
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        for k, v in tokenized_examples.items():
            # encoded_data[k] = encoded_data.get(k, []) + [v[i: i + len(full_labels)] for i in range(0, len(v), len(full_labels))]
            encoded_data[k] = [v[i: i + len(full_labels)] for i in range(0, len(v), len(full_labels))]
        with open(data_path + f'/batch{_batch}.json', 'w') as outfile:
            json.dump(encoded_data, outfile)


    # return {k: [v[i: i + len(full_labels)] for i in range(0, len(v), len(full_labels))] for k, v in tokenized_examples.items()}
    # return encoded_data

def print_scores(preds, y):
    print("Accuracy: ", accuracy_score(y, preds))
    print("Balanced Accuracy: ", balanced_accuracy_score(y, preds))
    print("F1 macro: ", f1_score(y, preds, average='macro'))

def evaluate(model_path, val_dataset):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import joblib

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")

    encoder = joblib.load(model_path + "/label_encoder.pkl")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               cache_dir=args.cache_dir,
                                                               num_labels=len(encoder.classes_)).to(dev)

    model.eval()
    val_labels = val_dataset.labels[:]
    val_real = encoder.inverse_transform(val_labels)
    val_encodings = val_dataset.encodings
    val_texts = tokenizer.batch_decode(val_encodings['input_ids'][:], skip_special_tokens=True)
    tensors = torch.split(torch.tensor(val_encodings['input_ids'][:], dtype=torch.long), 1)
    preds = np.array([model(t.to(dev)).logits.detach().cpu().numpy().ravel() for t in tensors])  # should be a 2-d array
    pred_labels = np.argmax(preds, axis=1).astype(int)
    preds = encoder.inverse_transform(pred_labels)

    print_scores(preds=pred_labels, y=val_labels)

    t_p_r = [[text, pred, real] for text, pred, real in zip(val_texts, preds, val_real)]
    with open(model_path + 'preds.json', 'w') as outfile:
        json.dump(t_p_r, outfile)
    print(preds[:50])


# **************** generate and evaluate ******************

def greedy_decode(model, tokenizer, encoder, test_data, only_loc=False, only_text=False, conversion_dict=None,
                  labels=None, nba_data=False):
    """

    :param model:
    :param texts: list of formatted texts for a testimony
    :param labels: real labels (for evaluation)
    :return:
    """
    if labels is None:
        if not nba_data:
            texts, labels = _make_texts(test_data, unused=[], out_path="1", conversion_dict=conversion_dict) # only current text
        else:
            texts, labels = _make_texts(test_data, unused=[], out_path="1", nba_data=True)
    else:
        texts = test_data
    model.eval()
    ll = 0.
    with torch.no_grad():
        output_sequence = []
        prev_text = ""
        _locs = ["START", "START"]
        for text in texts:
            if only_loc:
                t = " [SEP] ".join([_locs[-2], _locs[-1]])
            elif only_text:
                t = text
            else:
                t = " [SEP] ".join([_locs[-2], prev_text, _locs[-1]] + [text])
            encoding = tokenizer(t, truncation=True, padding=True)
            # encoding = torch.split(torch.tensor(encoding['input_ids'], dtype=torch.long), 1)
            encoding = torch.tensor(encoding['input_ids'], dtype=torch.long).to(dev)
            prediction = model(encoding.unsqueeze(0))
            _probs = torch.log_softmax(prediction.logits[0].detach().cpu(), dim=-1).numpy()
            # p = prediction.logits[0].detach().cpu().numpy()
            output_sequence.append(int(np.argmax(_probs)))
            ll += _probs[output_sequence[-1]]
            _locs = [_locs[-1]] + list(encoder.inverse_transform(output_sequence[-1:]))
            prev_text = text
    print(f"Greedy likelihood score: {ll}")
    return output_sequence, encoder.transform(labels)

def beam_decode(model, tokenizer, encoder, test_data, k=3, only_loc=False, only_text=False, conversion_dict=None,
                labels=None, nba_data=False):
    """

    :param model:
    :param texts: list of formatted texts for a testimony
    :param labels: real labels (for evaluation)
    :return:
    """
    #fix!!!
    print(f"Num beams: {k}")
    if labels is None:
        if not nba_data:
            texts, labels = _make_texts(test_data, unused=[], out_path="1",
                                        conversion_dict=conversion_dict)  # only current text
        else:
            texts, labels = _make_texts(test_data, unused=[], out_path="1", nba_data=True)
    else:
        texts = test_data
    model.eval()
    with torch.no_grad():
        #start with an empty sequence with zero score
        # output_sequences = [([], 0)]
        output_sequences = [(["START", "START"], 0)]

        prev_text = ""
        # _locs = ["START", "START"]
        for text in texts:
            new_sequences = []

            for old_seq, old_score in output_sequences:
                if only_loc:
                    t = " [SEP] ".join([old_seq[-2], old_seq[-1]])
                elif only_text:
                    t = text
                else:
                    t = " [SEP] ".join([old_seq[-2], prev_text, old_seq[-1]] + [text])
                encoding = tokenizer(t, truncation=True, padding=True)
                # encoding = torch.split(torch.tensor(encoding['input_ids'], dtype=torch.long), 1)
                encoding = torch.tensor(encoding['input_ids'], dtype=torch.long).to(dev)
                prediction = model(encoding.unsqueeze(0))
                _probs = torch.log_softmax(prediction.logits[0].detach().cpu(), dim=-1).numpy()

                for char_index in range(len(_probs)):
                    new_seq = old_seq + encoder.inverse_transform([char_index]).tolist()
                    #considering log-likelihood for scoring
                    new_score = old_score + _probs[char_index]
                    # new_score = old_score + np.log(_probs[char_index])
                    new_sequences.append((new_seq, new_score))

            #sort all new sequences in the decreasing order of their score
            output_sequences = sorted(new_sequences, key=lambda val: val[1], reverse=True)
            #select top-k based on score
            # *Note- best sequence is with the highest score
            output_sequences = output_sequences[:k]

            # _locs = [_locs[-1]] + list(encoder.inverse_transform(output_sequence[-1:]))
            prev_text = text
    output_sequences = [(os[0][2:], os[1]) for os in output_sequences]
    print(f"Top beam score: {output_sequences[0][1]}")
    return output_sequences, encoder.transform(labels)


def _mergeGrad(modelA, modelB):
    # listGradA = []
    # listGradB = []
    # itr = 0
    for pA, pB in zip(modelA.parameters(), modelB.parameters()):
        # listGradA.append(pA)
        # listGradB.append(pB)
        if not pA.requires_grad or pA. grad is None:
            continue
        # avg = (pA.grad + pB.grad.to(pA.device))/2
        avg = pA.grad + pB.grad.to(pA.device)
        pA.grad = avg
        pB.grad = avg.clone().to(pB.device)
        # itr += 1

class LocCRF:
    """
    Class for CRF (for location tracking)
    """
    def __init__(self, model_path, model_path2=None, use_prior='', conversion_dict=None, vectors=None, theta=None,
                 train_vectors=False, v_scales=False, w_scales=False, divide_hidden2=False, ner=False, use_bins=False,
                 divide_model=False, divide_model2=False, multihead=False, c_dict=None, c_dict2=None, i_labels=None,
                 i_labels_m=None, full_grad=False, nba=False, normalize_crf=False):
        # from TorchCRF import CRF
        from locCRF import CRF
        self.v_scales = v_scales
        self.w_scales = w_scales
        self.model_path = model_path
        self.model_path2 = model_path2
        self.normalize_crf = normalize_crf
        if multihead:
            self.model_path0 = model_path
            self.model_path1 = model_path[:model_path.rfind("/")+1] + "h1" + model_path[model_path.rfind("/")+3:]
            self.model_path2 = model_path2
            self.model_path3 = model_path2[:model_path2.rfind("/")+1] + "h1" + model_path2[model_path2.rfind("/")+3:]
        self.conversion_dict = conversion_dict
        self.vectors = vectors
        self.train_vectors = train_vectors
        self.ner = ner
        self.nba = nba
        self.use_bins = use_bins
        self.divide_model = divide_model
        self.divide_model2 = divide_model2
        self.multihead = multihead
        self.full_grad = full_grad
        if vectors is None:
            self.encoder = joblib.load(model_path + "/label_encoder.pkl")
            self.classes = self.encoder.classes_
            if not nba:
                self.start_id = self.encoder.transform(["START"])[0]
            if multihead:
                self.encoders = joblib.load(model_path + '/label_encoders.pkl')
                self.lengths = [len(e.classes_) for e in self.encoders]
                self.conversion_dict1 = {self.encoder.transform([l])[0]:
                                             [_e.transform([_ls])[0] for _e, _ls in zip(self.encoders, ls)]
                                         for l, ls in c_dict.items()}
                self.conversion_dict2 = {int(k):v for k, v in c_dict2.items()}
        else:
            self.classes = vectors[0]
            if not nba:
                self.start_id = self.classes.index("START")
            from loc_clusters import SBertEncoder
            if train_vectors:
                self.encoder = SBertEncoder(vectors=vectors, train_vectors=True)
                self.vectors = self.encoder.classes_, self.encoder.vectors

                # self.encoder = SBertEncoder(vectors=None, conversion_dict=conversion_dict, cat=True).to(dev3)
                # self.vectors = self.encoder.classes_, self.encoder.vectors.detach()
                # self.vectors[1].requires_grad = True
            else:
                self.encoder = SBertEncoder(vectors=vectors)

        distilroberta = model_path.split('/')[-1].find("distil") >= 0
        albert = model_path.split('/')[-1].find("albert") >= 0
        minilm = model_path.split('/')[-1].find("minilm") >= 0
        gpt = model_path.split('/')[-1].find("gpt") >= 0
        luke = model_path.split('/')[-1].find("luke") >= 0
        if distilroberta:
            self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base",
                                                           cache_dir=args.cache_dir)
        elif albert:
            self.tokenizer = AutoTokenizer.from_pretrained("albert-large-v2",
                                                           cache_dir=args.cache_dir)
        elif minilm:
            self.tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large",
                                                           cache_dir=args.cache_dir)
        elif gpt:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2",
                                                           cache_dir=args.cache_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif luke:
            self.tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base",
                                                           cache_dir=args.cache_dir)
        else:  # deberta
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base",
                                                           cache_dir=args.cache_dir)

        # self.p_model = None
        self.i_labels = i_labels
        self.i_labels_m = i_labels_m

        n_labels = len(self.classes) if model_path.split('/')[-1][0] != "v" else len(vectors[1][0])
        n_labels2 = len(self.classes) ** 2 if vectors is None else len(vectors[1][0]) ** 2
        if multihead:
            n_labels01 = [len(e.classes_) for e in self.encoders]
            n_labels23 = [len(e.classes_) ** 2 for e in self.encoders]
        if gpt:
            n_labels2 = n_labels
        if self.w_scales:
            n_labels += len(vectors[1][0])
            n_labels2 += len(vectors[1][0])
        # p_type = "multi_label_classification" if vectors is None else "regression"
        p_type = "single_label_classification" if vectors is None else None  # TODO!!!
        print(f"******************** {model_path.split('/')[-1]} *******************************")
        if not multihead:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                            cache_dir=args.cache_dir,
                                                                            num_labels=n_labels, problem_type=p_type).to(dev)
            if model_path2 is not None:
                print(f"******************** {model_path2.split('/')[-1]} *******************************")
                self.model2 = AutoModelForSequenceClassification.from_pretrained(model_path2,
                                                                                 cache_dir=args.cache_dir,
                                                                                 num_labels=n_labels2, problem_type=p_type).to(dev2)
            if self.divide_model:
                self.modelb = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                                 cache_dir=args.cache_dir,
                                                                                 num_labels=n_labels, problem_type=p_type).to(dev6)
            if self.divide_model2:
                self.model2b = AutoModelForSequenceClassification.from_pretrained(model_path2,
                                                                                  cache_dir=args.cache_dir,
                                                                                  num_labels=n_labels2, problem_type=p_type).to(dev7)
        else:
            self.model0 = AutoModelForSequenceClassification.from_pretrained(self.model_path0,
                                                                             cache_dir=args.cache_dir,
                                                                             num_labels=n_labels01[0], problem_type=p_type).to(dev)
            self.model1 = AutoModelForSequenceClassification.from_pretrained(self.model_path1,
                                                                             cache_dir=args.cache_dir,
                                                                             num_labels=n_labels01[1], problem_type=p_type).to(dev)
            self.model2 = AutoModelForSequenceClassification.from_pretrained(self.model_path2,
                                                                             cache_dir=args.cache_dir,
                                                                             num_labels=n_labels23[0], problem_type=p_type).to(dev)
            self.model3 = AutoModelForSequenceClassification.from_pretrained(self.model_path3,
                                                                             cache_dir=args.cache_dir,
                                                                             num_labels=n_labels23[1], problem_type=p_type).to(dev)
            if divide_model:
                self.model0b = AutoModelForSequenceClassification.from_pretrained(self.model_path0,
                                                                                 cache_dir=args.cache_dir,
                                                                                 num_labels=n_labels01[0],
                                                                                 problem_type=p_type).to(dev)
                self.model1b = AutoModelForSequenceClassification.from_pretrained(self.model_path1,
                                                                                 cache_dir=args.cache_dir,
                                                                                 num_labels=n_labels01[1],
                                                                                 problem_type=p_type).to(dev)
            if divide_model2:
                self.model2b = AutoModelForSequenceClassification.from_pretrained(self.model_path2,
                                                                                 cache_dir=args.cache_dir,
                                                                                 num_labels=n_labels23[0],
                                                                                 problem_type=p_type).to(dev)
                self.model3b = AutoModelForSequenceClassification.from_pretrained(self.model_path3,
                                                                                 cache_dir=args.cache_dir,
                                                                                 num_labels=n_labels23[1],
                                                                                 problem_type=p_type).to(dev)

        pad_idx = None
        const = None
        pad_idx_val = None
        self.prior = use_prior
        if self.train_vectors:
            self.prior = self.prior + "_tv"
        if model_path2 is not None:
            self.prior = self.prior + "_trans"
        if use_prior == 'I':
            pad_idx = np.arange(len(self.classes))
        if use_prior == 'I1':
            pad_idx = np.arange(len(self.classes))
            pad_idx_val = -1.
        if use_prior == 'const':
            const = 0.1
        self.divide_hidden_2 = divide_hidden2
        self.crf = CRF(len(self.classes), pad_idx=pad_idx, pad_idx_val=pad_idx_val, const=const, theta=theta, divide_h2=divide_hidden2).to(dev)
        if not nba:
            with torch.no_grad():
                self.crf.trans_matrix[:, self.start_id] = -10000.
                self.crf.trans_matrix[self.start_id, self.start_id] = 0.


        # set self.crf.trans_matrix ?
        # self.crf.trans_matrix = nn.Parameter(torch.diag(torch.ones(len(self.encoder.classes_))))
        # set by pairwise distance??


    def  _forward(self, texts):
        """

        :param texts: of shape (batch_size, num_segments)
        :return:
        """
        # print("forward")
        if not self.multihead:
            self.model.to(dev)
            if self.divide_model:
                self.modelb.to(dev3)
        else:
            self.model0.to(dev)
            self.model1.to(dev2)
            self.model2.to(dev3)
            if torch.cuda.device_count() >= 6:
                self.model3.to(dev4)
            else:
                self.model3.to(dev3)
            if self.divide_model:
                self.model0b.to(dev5)
                if torch.cuda.device_count() >= 6:
                    self.model1b.to(dev6)
                else:
                    self.model1b.to(dev4)
                if self.divide_model2:
                    self.model2b.to(dev7)
                    self.model3b.to(dev8)
        # shape (batch_size, seq_len, num_classes)
        # hidden = torch.zeros(len(texts), max([len(t) for t in texts]), len(self.classes)).to(dev)
        hidden = torch.zeros(len(texts), max([len(t) for t in texts]), len(self.classes)).to(dev5)

        divide_hidden_2 = self.divide_hidden_2
        divide_model = self.divide_model
        divide_model2 = self.divide_model2
        use_b = False
        use_2b = False
        if self.model_path2 is not None:
            if not self.multihead:
                self.model2.to(dev2)
            # shape (batch_size, seq_len, num_classes, num_classes)
            seq_len = max([len(t) for t in texts])
            if divide_hidden_2:
                hidden2 = torch.zeros(len(texts), seq_len // 5, len(self.classes),
                                      len(self.classes)).to(dev3)
                hidden2_2 = torch.zeros(len(texts), seq_len//5, len(self.classes),
                                      len(self.classes)).to(dev4)
                hidden2_3 = torch.zeros(len(texts), seq_len//5, len(self.classes),
                                      len(self.classes)).to(dev5)
                hidden2_4 = torch.zeros(len(texts), seq_len // 5, len(self.classes),
                                      len(self.classes)).to(dev6)
                hidden2_5 = torch.zeros(len(texts), seq_len - 4 * (seq_len//5), len(self.classes),
                                      len(self.classes)).to(dev2)
                # hidden2_6 = torch.zeros(len(texts), seq_len - 2 * (seq_len//6), len(self.classes),
                #                       len(self.classes)).to(dev7)
            else:
                hidden2 = torch.zeros(len(texts), seq_len, len(self.classes),
                                      len(self.classes)).to(dev2)
            # hidden2 = torch.zeros(len(texts), max([len(t) for t in texts]), len(self.classes),
            #                       len(self.classes)).to("cpu")

        if len(texts) == 0:
            return torch.from_numpy(hidden).to(dev)
        for k, text in tqdm.tqdm(enumerate(texts), desc="Make encodings", leave=False):
            # encodings = self.tokenizer(text, truncation=True, padding=True)['input_ids'].clone().detach().to(dev)
            encodings = torch.tensor(self.tokenizer(text, truncation=True, padding=True)['input_ids'], dtype=torch.long).to(dev)

            if self.model_path2 is not None and self.vectors is None:
                encodings2 = encodings
                # cats = self.encoder.classes_
                # encodings2 = [torch.tensor(self.tokenizer([_text + " [SEP] " + c for _text in text], truncation=True,
                #                                           padding=True)['input_ids'], dtype=torch.long).to(dev2)
                #               for c in cats]

            for j, e in enumerate(encodings):
                if j == len(encodings) // 2 and divide_model:
                    use_b = True

                if self.vectors is None:
                    if not self.multihead:
                        if use_b:
                            probs = self.modelb(e.unsqueeze(0).to(self.modelb.device)).logits[0]  #????
                        else:
                            probs = self.model(e.unsqueeze(0).to(self.model.device)).logits[0]  #????
                    else:
                        if use_b:
                            logits0 = self.model0b(e.unsqueeze(0).to(self.model0b.device)).logits
                            logits1 = self.model1b(e.unsqueeze(0).to(self.model1b.device)).logits.to(self.model0b.device)
                        else:
                            logits0 = self.model0(e.unsqueeze(0).to(self.model0.device)).logits
                            logits1 = self.model1(e.unsqueeze(0).to(self.model1.device)).logits.to(self.model0.device)
                else:
                    if self.model_path.split('/')[-1][0] != 'v':
                        probs = self.model(e.unsqueeze(0).to(dev)).logits[0]  #????
                    else:
                        outputs = self.model(e.unsqueeze(0).to(dev))
                        logits = outputs.get('logits').squeeze(0)  # of shape (batch_size, vec_dim)?

                        if self.w_scales:
                            weights = logits[-len(self.vectors[1][0]):]
                            logits = logits[:-len(self.vectors[1][0])]
                            if self.train_vectors:
                                probs = (logits * weights @ self.vectors[1].to(logits.device).T).squeeze(0)
                            else:
                                probs = (logits * weights @ torch.from_numpy(self.vectors[1]).to(logits.device).T).squeeze(0)

                        # should be of shape (batch_size, num_labels)
                        # probs = util.cos_sim(logits, torch.from_numpy(self.vectors[1]).to(logits.device)).log_softmax(dim=-1).squeeze(0)
                        else:
                            if self.train_vectors:
                                if self.v_scales:
                                    probs = util.dot_score(logits, self.vectors[1].to(logits.device)).squeeze(0)
                                else:
                                    probs = util.cos_sim(logits, self.vectors[1].to(logits.device)).squeeze(0)
                            else:
                                if self.v_scales:
                                    probs = util.dot_score(logits, torch.from_numpy(self.vectors[1]).to(logits.device)).squeeze(0)
                                else:
                                    probs = util.cos_sim(logits, torch.from_numpy(self.vectors[1]).to(logits.device)).squeeze(0)

                    if self.model_path2 is not None:

                        use_prompt = self.model_path2.find("gpt") >= 0
                        if use_prompt:
                            past_inputs = self.tokenizer(e.unsqueeze(0), return_tensors="pt").to(dev2)
                            past_key_values = self.model2(input_ids=past_inputs['input_ids'], use_cache=True)['past_key_values']

                            for i, c in enumerate(self.classes):
                                inputs = self.tokenizer(f" location: {c}", return_tensors="pt").to(dev2)
                                outputs2 = self.model2(input_ids=inputs['input_ids'], past_key_values=past_key_values)

                                if divide_hidden_2:
                                    if j >= 4*(seq_len // 5):
                                        hidden2_5[k, j - 4*(seq_len // 5), i, :] = outputs2.logits[0]
                                    elif j >= 3*(seq_len // 5):
                                        hidden2_4[k, j - 3*(seq_len // 5), i, :] = outputs2.logits[0]
                                    elif j >= 2*(seq_len // 5):
                                        hidden2_3[k, j - 2*(seq_len // 5), i, :] = outputs2.logits[0]
                                    elif j >= seq_len // 5:
                                        hidden2_2[k, j - seq_len // 5, i, :] = outputs2.logits[0]
                                    else:
                                        hidden2[k, j, i, :] = outputs2.logits[0]
                                else:
                                    hidden2[k, j, i, :] = outputs2.logits[0]

                        else:
                            outputs2 = self.model2(e.unsqueeze(0).to(dev2))
                            # logits2 = outputs2.get('logits').squeeze(0)  # of shape (batch_size, vec_dim**2)?
                            if divide_hidden_2:
                                if j >= 4 * (seq_len // 5):
                                    logits2 = outputs2.get('logits').squeeze(0).to(dev2)  # of shape (batch_size, vec_dim**2)?
                                elif j >= 3 * (seq_len // 5):
                                    logits2 = outputs2.get('logits').squeeze(0).to(dev6)  # of shape (batch_size, vec_dim**2)?
                                elif j >= 2 * (seq_len // 5):
                                    logits2 = outputs2.get('logits').squeeze(0).to(dev5)  # of shape (batch_size, vec_dim**2)?
                                elif j >= seq_len // 5:
                                    logits2 = outputs2.get('logits').squeeze(0).to(dev4)  # of shape (batch_size, vec_dim**2)?
                                else:
                                    logits2 = outputs2.get('logits').squeeze(0).to(dev3)  # of shape (batch_size, vec_dim**2)?
                            else:
                                logits2 = outputs2.get('logits').squeeze(0).to(dev3)
                            # here we work with a single input
                            if self.w_scales:
                                weights2 = logits2[-len(self.vectors[1][0]):]
                                logits2 = logits2[:-len(self.vectors[1][0])]
                                if self.train_vectors:
                                    out2 = logits2.reshape(int(logits2.shape[-1] ** .5), int(logits2.shape[-1] ** .5)) @ self.vectors[1].to(logits2.device).T
                                    probs2 = (out2 * weights2 @ self.vectors[1].to(logits2.device).T)
                                else:
                                    out2 = logits2.reshape(int(logits2.shape[-1] ** .5), int(logits2.shape[-1] ** .5)) @ torch.from_numpy(self.vectors[1]).to(logits2.device).T
                                    probs2 = (out2 * weights2 @ torch.from_numpy(self.vectors[1]).to(logits2.device).T).squeeze(0)
                            else:
                                if self.train_vectors:
                                    # out2 = logits2.reshape(int(logits2.shape[-1] ** .5), int(logits2.shape[-1] ** .5)) @ self.vectors[1].to(logits2.device).T
                                    out2 = logits2.reshape(int(logits2.shape[-1] ** .5), int(logits2.shape[-1] ** .5)) @ self.vectors[1].to(logits2.device).T
                                    if self.v_scales:
                                        probs2 = util.dot_score(out2.T, self.vectors[1].to(logits2.device))
                                    else:
                                        probs2 = util.cos_sim(out2.T, self.vectors[1].to(logits2.device))
                                    if divide_hidden_2:
                                        if j >= 2*(seq_len // 3):
                                            hidden2_3[k, j - 2 * (seq_len // 3), :, :] = probs2
                                        elif j >= seq_len // 3:
                                            hidden2_2[k, j - seq_len // 3, :, :] = probs2
                                        else:
                                            hidden2[k, j, :, :] = probs2
                                    else:
                                        hidden2[k, j, :, :] = probs2
                                    # hidden2[k, j, :, :] = probs2.to(dev3)
                                else:
                                    out2 = logits2.reshape(int(logits2.shape[-1] ** .5), int(logits2.shape[-1] ** .5)) @ torch.from_numpy(self.vectors[1]).to(logits2.device).T
                                    if SOFTMAX:
                                        out2 = out2.log_softmax(dim=-1)

                                    if self.v_scales:
                                        # probs2 = util.dot_score(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device))
                                        # hidden2[k, j, :, :] = util.cos_sim(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device))
                                        if divide_hidden_2:
                                            if j >= 4*(seq_len // 5):
                                                hidden2_5[k, j - 4*(seq_len // 5), :, :] = util.dot_score(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device))
                                            elif j >= 3*(seq_len // 5):
                                                hidden2_4[k, j - 3*(seq_len // 5), :, :] = util.dot_score(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device))
                                            elif j >= 2*(seq_len // 5):
                                                hidden2_3[k, j - 2*(seq_len // 5), :, :] = util.dot_score(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device))
                                            elif j >= seq_len // 5:
                                                hidden2_2[k, j - seq_len // 5, :, :] = util.dot_score(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device))
                                            else:
                                                hidden2[k, j, :, :] = util.dot_score(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device))
                                        else:
                                            hidden2[k, j, :, :] = util.dot_score(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device))
                                    else:
                                        probs2 = util.cos_sim(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device))
                                        hidden2[k, j, :, :] = probs2
                                # probs2 should be a tensor of shape (num_classes, num_classes)
                                # probs2 = util.cos_sim(out2.T, torch.from_numpy(self.vectors[1]).to(logits2.device)).log_softmax(dim=1)
                if not self.multihead:
                    hidden[k, j, :probs.shape[0]] = probs
                    if self.normalize_crf:
                        hidden[k, j, :probs.shape[0]] = hidden[k, j, :probs.shape[0]].log_softmax(dim=-1)
                else:
                    probs = DividedTrainer.make_logits(torch.cat((logits0, logits1), -1),
                                                       conversion_dict=self.conversion_dict1,
                                                       _lengths=self.lengths, i_labels=self.i_labels)
                    hidden[k, j, :probs.shape[-1]] = probs[0]

            if self.model_path2 is not None and self.vectors is None and self.model_path2.find("gpt") == -1:
                if self.multihead:
                    _logits2 = torch.full((len(encodings), self.lengths[0] ** 2), -10000.)
                    _logits3 = torch.full((len(encodings), self.lengths[1] ** 2), -10000.)
                for j, e in enumerate(encodings2):
                    if j == len(encodings) // 2 and divide_model2:
                        use_2b = True
                    if use_2b:
                        if not self.multihead:
                            probs2 = self.model2b(e.unsqueeze(0).to(self.model2b.device)).logits.squeeze(0)
                        else:
                            logits2 = self.model2b(e.unsqueeze(0).to(self.model2b.device)).logits
                            logits3 = self.model3b(e.unsqueeze(0).to(self.model3b.device)).logits.to(self.model2b.device)
                    else:
                        if not self.multihead:
                            probs2 = self.model2(e.unsqueeze(0).to(self.model2.device)).logits.squeeze(0)
                        else:
                            logits2 = self.model2(e.unsqueeze(0).to(self.model2.device)).logits
                            logits3 = self.model3(e.unsqueeze(0).to(self.model3.device)).logits.to(self.model2.device)
                            _logits2[j] = logits2
                            _logits3[j] = logits3

                    # if self.multihead:
                    #     probs2 = DividedTrainer.make_logits(torch.cat((logits2, logits3), -1),
                    #                                         conversion_dict=self.conversion_dict2,
                    #                                         _lengths=[l**2 for l in self.lengths],
                    #                                         i_labels=self.i_labels_m, matrix=True)
                    if not self.multihead:
                        hidden2[k, j, :, :] = probs2.reshape(len(self.classes), -1).log_softmax(dim=-1)
                if self.multihead:
                    # probs2 = DividedTrainer.make_logits(torch.cat((logits2, logits3), -1),
                    #                                     conversion_dict=self.conversion_dict2,
                    #                                     _lengths=[l**2 for l in self.lengths],
                    #                                     i_labels=self.i_labels_m, matrix=True)
                    # hidden2[k, j, :, :] = probs2.reshape(len(self.classes), -1).log_softmax(dim=-1)
                    probs2 = DividedTrainer.make_logits(torch.cat((_logits2, _logits3), -1),
                                                        conversion_dict=self.conversion_dict2,
                                                        _lengths=[l**2 for l in self.lengths],
                                                        i_labels=self.i_labels_m, matrix=True)
                    hidden2[k] = probs2.reshape(len(encodings), len(self.classes), -1)

        if self.model_path2 is not None:
            if divide_hidden_2:
                return hidden, hidden2, hidden2_2, hidden2_3, hidden2_4, hidden2_5
            return hidden, hidden2
        return hidden

    def _forward_b(self, texts, batch_size=4, batch_id=None):
        """

        :param texts: of shape (batch_size, num_segments)
        :return:
        """
        self.model.to(dev)
        b_texts = [texts[0][b_start: b_start+batch_size] for b_start in range(0, len(texts[0]), batch_size)]

        # shape (batch_size, seq_len, num_classes)
        # hidden = torch.zeros(max([len(t) for t in texts]), len(self.classes)).to(dev)

        if self.model_path2 is not None:
            if not self.multihead:
                self.model2.to(dev2)
            # shape (batch_size, seq_len, num_classes, num_classes)
            seq_len = max([len(t) for t in texts])

            # hidden2 = torch.zeros(len(texts), seq_len, len(self.classes),
            #                       len(self.classes)).to(dev2)

        if batch_id is None:
            _encodings = [self.tokenizer(b_text, truncation=True, padding=True, return_tensors='pt') for b_text in
                          b_texts]
            _outs = [self.model(**_e.to(dev)).logits for _e in _encodings]
            outs = torch.vstack(_outs)
            if self.model_path2 is not None:
                _outs2 = [self.model2(**_e.to(dev)).logits for _e in _encodings]
                outs2 = torch.vstack(_outs2)
        else:
            b_text = b_texts[batch_id]
            _encodings = self.tokenizer(b_text, truncation=True, padding=True, return_tensors='pt')
            outs = self.model(**_encodings.to(dev)).logits
            if self.model_path2 is not None:
                outs2 = self.model2(**_encodings.to(dev)).logits
        if self.normalize_crf:
            outs = outs.log_softmax(dim=-1)
            if self.model_path2 is not None:
                outs2 = outs2.reshape(len(self.classes), -1).log_softmax(dim=-1)

        if self.model_path2 is not None:
            return outs.unsqueeze(0), outs2.unsqueeze(0)
        return outs.unsqueeze(0), None

    def eval_loss(self, test_data=None):
        _eval = list(test_data.values())
        losses = []
        sys.stdout.flush()
        batch_size = 1
        if not self.multihead:
            self.model.eval()
            if self.model_path2 is not None:
                self.model2.eval()
        else:
            self.model0.eval()
            self.model1.eval()
            self.model2.eval()
            self.model3.eval()
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(_eval)), desc="Eval loss"):
                # for t, t_data in tqdm.tqdm(list(train_data.items())[:int(len(train_data) * 0.9)]):
                eval_batch = _eval[i: i+batch_size]
                _batch_size = len(eval_batch)

                texts = []
                label_mask = np.zeros((_batch_size, max([len(t) for t in eval_batch])), dtype=bool)
                labels = np.zeros((_batch_size, max([len(t) for t in eval_batch])), dtype=int)

                for j, _t in enumerate(eval_batch):
                    if not self.nba:
                        _texts, _labels = _make_texts({1: _t}, unused=[], out_path=("_all" if self.model_path.find(
                            "_all") >= 0 else "") + "1", conversion_dict=self.conversion_dict, ners=self.ner,
                                                      use_bins=self.use_bins)  # only current text

                    else:
                        _texts, _labels = _make_texts({1: _t}, unused=[], out_path=("_all" if self.model_path.find("_all") >= 0 else "") + "1", nba_data=True)  # only current text
                    # _texts, _labels = _make_texts({1: _t}, unused=[], out_path="1", conversion_dict=self.conversion_dict)  # only current text
                    label_mask[j, :len(_t)] = 1
                    labels[j, :len(_t)] = self.encoder.transform(_labels)
                    texts.append(_texts)
                    # labels.append(_labels)

                # hidden = self._forward(texts).detach()
                if self.model_path2 is not None:
                    if self.divide_hidden_2:
                        hidden, hidden2, hidden2_2, hidden2_3, hidden2_4, hidden2_5 = self._forward(texts)
                    else:
                        hidden, hidden2 = self._forward(texts)
                    # hidden2 = hidden2.to(dev)
                    hidden2 = hidden2.to(dev)
                else:
                    hidden, hidden2 = self._forward(texts), None

                # mask = torch.from_numpy(label_mask).to(dev)
                mask = torch.from_numpy(label_mask).to(dev)
                # labels = torch.from_numpy(labels).to(dev)
                labels = torch.from_numpy(labels).to(dev)

                # self.crf.eval()
                if self.divide_hidden_2:
                    loss = -self.crf.forward(hidden.to(dev), labels, mask, h2=[hidden2, hidden2_2, hidden2_3, hidden2_4, hidden2_5]).mean()
                else:
                    loss = -self.crf.forward(hidden.to(dev), labels, mask, h2=hidden2).mean()
                # self.crf.train()
                losses.append(loss.item())
                # print(loss.item())
        print("Eval loss:", np.mean(losses))
        wandb.log({'Eval loss': np.mean(losses)})

    def train(self, train_data, batch_size=4, epochs=30, test_dict=None, test_data=None, accu_grad=1, layers=None, layersb=None, wd=None):
        print(f"Training. Batch size: {batch_size}, epochs: {epochs}")
        print(f"Accu Grad: {accu_grad}")
        self.prior = self.prior + f"_b{batch_size}_e{epochs}"
        import torch.optim as optim
        # import torch.nn as nn
        # self.model.eval()  # don't train model
        print(f"******************* leaving layers {layers} unfrozen!!!!!!!!! *******************")

        _i = 0
        if self.model_path2 is not None:
            if not self.multihead:
                self.model2.train()
                _models = [self.model2]
                if self.divide_model2:
                    self.model2b.train()
                    _models = _models + [self.model2b]
            else:
                self.model2.train()
                self.model3.train()
                _models = [self.model2, self.model3]
                if self.divide_model2:
                    self.model2b.train()
                    self.model3b.train()
                    _models = _models + [self.model2b, self.model3b]
            for _model in _models:
                for name, param in _model.named_parameters():
                    # if name.find("11.output") == -1 and name.find("classifier") == -1:
                    # if np.any([name.find(str(_l)) >= 0 for _l in layersb + ["classifier", "pooler", "score"]]):
                    if np.any([name.find(str(_l)) >= 0 for _l in layersb]):
                        # if name.find("5") == -1 and name.find("classifier") == -1 and name.find("pooler") == -1:
                        # if name.find("classifier") == -1:
                        param.requires_grad = True
                        _i += 1
                    else:
                        param.requires_grad = False
                print(f"Trainable in model2: {_i}")

            # _i = 0
            # if self.divide_model2:
            #     self.model2b.train()
            #     for name, param in self.model2b.named_parameters():
            #         # if name.find("11.output") == -1 and name.find("classifier") == -1:
            #         if np.any([name.find(str(_l)) >= 0 for _l in layersb + ["classifier", "pooler"]]):
            #             # if name.find("5") == -1 and name.find("classifier") == -1 and name.find("pooler") == -1:
            #             # if name.find("classifier") == -1:
            #             param.requires_grad = True
            #             _i += 1
            #         else:
            #             param.requires_grad = False
            #     print(f"Trainable in model2b: {_i}")
            #     _i = 0

        # if self.model_path.find("deberta") >= 0 or self.model_path.find("luke") >= 0:  # TODO
        #     layers = []
        if not self.multihead:
            self.model.train()
            _models = [self.model]
            if self.divide_model:
                self.modelb.train()
                _models = _models + [self.modelb]
        else:
            self.model0.train()
            self.model1.train()
            _models = [self.model0, self.model1]
            if self.divide_model:
                self.model0b.train()
                self.model1b.train()
                _models = _models + [self.model0b, self.model1b]
        if not self.full_grad:
            for _model in _models:
                for name, param in _model.named_parameters():
                    if np.any([name.find(str(_l)) >= 0 for _l in layers + ["classifier", "pooler"]]):
                        param.requires_grad = True
                        _i += 1
                    else:
                        param.requires_grad = False
                print(f"Trainable in model: {_i}")
            _i = 0
        # if self.divide_model:
        #     self.modelb.train()
        #     for name, param in self.modelb.named_parameters():
        #         if np.any([name.find(str(_l)) >= 0 for _l in layers + ["classifier", "pooler"]]):
        #             param.requires_grad = True
        #             _i += 1
        #         else:
        #             param.requires_grad = False
        #             # param.requires_grad = False
        #     print(f"Trainable in modelb: {_i}")
        #     _i = 0

        # self.p_model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        if not self.multihead:
            to_train = list(self.crf.parameters()) + list(self.model.parameters())
            if self.model_path2 is not None:
                to_train = to_train + list(self.model2.parameters())
            if self.divide_model:
                to_train = to_train + list(self.modelb.parameters())
            if self.divide_model2:
                to_train = to_train + list(self.model2b.parameters())
            if self.train_vectors:
                to_train = to_train + list(self.encoder.parameters())
        else:
            to_train = list(self.crf.parameters()) + list(self.model0.parameters()) \
                       + list(self.model1.parameters()) + list(self.model2.parameters()) + list(self.model3.parameters())
            if self.divide_model:
                to_train = to_train + list(self.model0b.parameters()) + list(self.model1b.parameters())
            if self.divide_model2:
                to_train = to_train + list(self.model2b.parameters()) + list(self.model3b.parameters())
        # if self.model_path2 is None:
        #     # optimizer = optim.AdamW(list(self.crf.parameters()) + list(self.model.parameters()))
        #     optimizer = optim.AdamW(to_train, lr=5e-5, weight_decay=wd)
        # elif not self.train_vectors:
        #     optimizer = optim.AdamW(list(self.crf.parameters()) + list(self.model.parameters()) +
        #                             list(self.model2.parameters()), lr=5e-5, weight_decay=wd)
        # else:
        #     optimizer = optim.AdamW(list(self.crf.parameters()) + list(self.model.parameters()) +
        #                             list(self.model2.parameters()) + list(self.encoder.parameters()), lr=5e-5, weight_decay=wd)
        optimizer = optim.AdamW(to_train, lr=5e-5, weight_decay=wd)
        optimizer.zero_grad()

        # criterion = nn.CrossEntropyLoss()
        # self.crf = self.crf.to(dev)
        # criterion = criterion.to(dev)

        _train = list(train_data.values())  # list of lists
        # _train = list(train_data.items())[:int(len(train_data) * 0.9)]

        # first eval
        print("With random transitions")
        name = self.save_model(epoch=epochs)  # TODO:
        on_test = False
        if self.model_path2 is not None:
            crf_eval(data_path=test_dict['data_path'], model_path=test_dict['model_path'], model_path2=test_dict['model_path2'],
                     test_size=test_dict['test_size'], val_size=test_dict['val_size'], name=name, vectors=self.vectors,
                     ner=self.ner, use_test=test_dict['use_test'], test_set=on_test, conversion_dict=self.conversion_dict,
                     nba_data=test_dict['nba_data'])
        else:
            crf_eval(data_path=test_dict['data_path'], model_path=test_dict['model_path'],
                     test_size=test_dict['test_size'], val_size=test_dict['val_size'], name=name, vectors=self.vectors,
                     ner=self.ner, use_test=test_dict['use_test'], test_set=on_test, conversion_dict=self.conversion_dict,
                     nba_data=test_dict['nba_data'])

        random.seed()
        # self.model.train()
        for e in range(epochs):
            if self.model_path2 is not None:
                self.model2.train()
                if self.multihead:
                    self.model3.train()
            losses = []
            print("\n" + str(e))
            sys.stdout.flush()
            random.shuffle(_train)
            for i in tqdm.tqdm(range(0, len(_train), batch_size), desc="Train"):
                # for _d in [dev, dev2, dev3]:
                #     if _d != torch.device("cpu"):
                #         print(torch.cuda.memory_summary(_d, abbreviated=True))
                #         print("used:", print(torch.cuda.memory_allocated(_d) / torch.cuda.max_memory_allocated(_d)))
                # torch.cuda.empty_cache()

                # for t, t_data in tqdm.tqdm(list(train_data.items())[:int(len(train_data) * 0.9)]):
                train_batch = _train[i: i+batch_size]
                _batch_size = len(train_batch)

                texts = []
                label_mask = np.zeros((_batch_size, max([len(t) for t in train_batch])), dtype=bool)
                labels = np.zeros((_batch_size, max([len(t) for t in train_batch])), dtype=int)

                for j, _t in enumerate(train_batch):
                    if not self.nba:
                        _texts, _labels = _make_texts({1: _t}, unused=[], out_path=self.model_path,
                                                    conversion_dict=self.conversion_dict, ners=self.ner,
                                                    use_bins=self.use_bins)  # only current text
                    else:
                        _texts, _labels = _make_texts({1: _t}, unused=[], out_path=self.model_path, nba_data=True)
                    label_mask[j, :len(_t)] = 1
                    labels[j, :len(_t)] = self.encoder.transform(_labels)
                    texts.append(_texts)
                    # labels.append(_labels)

                if self.model_path2 is not None:
                    # with torch.no_grad():
                    if self.divide_hidden_2:
                        hidden, hidden2, hidden2_2, hidden2_3, hidden2_4, hidden2_5 = self._forward(texts)
                    else:
                        hidden, hidden2 = self._forward(texts)
                        # hidden2 = hidden2.to(dev)
                    hidden = hidden.to(dev)
                    hidden2 = hidden2.to(dev)
                else:
                    if self.full_grad:
                        _inner_batch_size = 4
                        num_batches = len(labels[0]) // _inner_batch_size + int(len(labels[0]) % _inner_batch_size != 0)
                        seed = torch.randint(1000, (1,))
                        torch.manual_seed(seed.item())
                        with torch.no_grad():
                            hidden, hidden2 = self._forward_b(texts)
                        hidden.requires_grad = True
                    else:
                        hidden, hidden2 = self._forward(texts), None

                # mask = torch.from_numpy(label_mask).to(dev)
                mask = torch.from_numpy(label_mask).to(dev)
                # mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size. sequence_size)
                # labels = torch.from_numpy(labels).to(dev)
                labels = torch.from_numpy(labels).to(dev)
                # labels = torch.LongTensor(self.encoder.transform(labels)).unsqueeze(0).to(dev)  # (batch_size, sequence_size)
                if self.divide_hidden_2:
                    loss = -self.crf.forward(hidden.to(dev2), labels, mask, h2=[hidden2, hidden2_2, hidden2_3, hidden2_4, hidden2_5]).mean() / accu_grad
                else:
                    loss = -self.crf.forward(hidden.to(dev), labels, mask, h2=hidden2).mean() / accu_grad
                # loss = criterion(y_pred, label)

                loss.backward()

                if self.full_grad:
                    grads = torch.clone(hidden.grad)
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.manual_seed(seed.item())
                    for b_i in range(num_batches):
                        outs, outs2 = self._forward_b(texts, batch_size=_inner_batch_size, batch_id=b_i)
                        outs.backward(grads[:, b_i * _inner_batch_size: (b_i + 1) * _inner_batch_size, :])
                        if outs2 is not None:
                            outs2.backward(grads[:, b_i * _inner_batch_size: (b_i + 1) * _inner_batch_size, :])

                if self.divide_model:
                    if not self.multihead:
                        _mergeGrad(self.model, self.modelb)
                    else:
                        _mergeGrad(self.model0, self.model0b)
                        _mergeGrad(self.model1, self.model1b)
                        if self.divide_model2:
                            _mergeGrad(self.model2, self.model2b)
                            _mergeGrad(self.model3, self.model3b)

                if accu_grad == 1 or (i+1) % accu_grad == 0 or i+accu_grad >= len(_train) or _batch_size < batch_size:
                    optimizer.step()
                    optimizer.zero_grad()

                losses.append(loss.item())
                wandb.log({'loss': loss})
                # print(loss.item())
            print("Epoch loss:", np.mean(losses))
            if self.crf.theta is not None:
                print("Theta: ", torch.sigmoid(self.crf.theta).item())
            wandb.log({'Epoch loss': np.mean(losses)})
            if test_data is not None:
                self.eval_loss(test_data)
            if e % 3 == 0:
                name = self.save_model(epoch=epochs)  # TODO:
                # crf_eval(data_path=test_dict['data_path'], model_path=test_dict['model_path'],
                #          test_size=test_dict['test_size'], val_size=test_dict['val_size'], name=name, vectors=self.vectors,
                #          nba_data=test_dict['nba_data'])
                crf_eval(data_path=test_dict['data_path'], model_path=test_dict['model_path'],
                         test_size=test_dict['test_size'], val_size=test_dict['val_size'], name=name,
                         vectors=self.vectors,
                         ner=self.ner, use_test=test_dict['use_test'], test_set=on_test,
                         conversion_dict=self.conversion_dict,
                         nba_data=test_dict['nba_data'])

        return self

    def decode(self, test_data, labels=None):
        """
        If labels are provided then test_data is only the texts
        :param test_data:
        :param labels:
        :return:
        """

        if labels is None:
            if not self.nba:
                texts, labels = _make_texts(test_data, unused=[], out_path=self.model_path, conversion_dict=self.conversion_dict, ners=self.ner, use_bins=self.use_bins)  # only current text
            else:
                texts, labels = _make_texts(test_data, unused=[], out_path=self.model_path, nba_data=True)
        else:
            texts = test_data
        with torch.no_grad():
            if self.multihead:
                self.model0.eval()
                self.model1.eval()
            else:
                self.model.eval()
            if self.model_path2 is None:
                if self.divide_hidden_2:
                    self.divide_hidden_2 = False
                    hidden, hidden2 = self._forward([texts]), None
                    self.divide_hidden_2 = True
                else:
                    hidden, hidden2 = self._forward([texts]), None
                # hidden = self._forward([texts])
                # hidden.detach()
                # mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size. sequence_size)
                mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size. sequence_size)

                return self.crf.viterbi_decode(hidden.to(dev), mask, 
                                               h2=hidden2.to(dev) if hidden2 is not None else None)[0], \
                       labels  # predictions and labels
            else:
                self.model2.eval()
                if self.multihead:
                    self.model3.eval()
                if self.divide_hidden_2:
                    self.divide_hidden_2 = False
                    hidden, hidden2 = self._forward([texts])
                    self.divide_hidden_2 = True
                else:
                    hidden, hidden2 = self._forward([texts])
                    hidden, hidden2 = hidden.to(dev), hidden2.to(dev)
                # hidden2.detach()
                # mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size, sequence_size)
                mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size, sequence_size)
                return self.crf.viterbi_decode(hidden, mask, h2=hidden2)[0], labels  # predictions and labels

    def save_model(self, epoch=None):
        if epoch is not None:
            print("Saving model " + f'/crf_{self.prior[:-3]}e{epoch}.pkl')
            torch.save(self, self.model_path + f'/crf_{self.prior[:-3]}e{epoch}.pkl')
            # joblib.dump(self, self.model_path + f'/crf_{self.prior[:-3]}e{epoch}.pkl')
            return f'/crf_{self.prior[:-3]}e{epoch}.pkl'
        else:
            print("Saving model " + f'/crf_{self.prior}.pkl')
            torch.save(self, self.model_path + f'/crf_{self.prior}.pkl')
            # joblib.dump(self, self.model_path + f'/crf_{self.prior}.pkl')
            return f'/crf_{self.prior}.pkl'

    def save_transitions(self, prompt=""):
        if self.model_path2 is None:
            matrix = self.crf.trans_matrix.detach().cpu().numpy()
            with open(args.base_path + "data/" + "sorted_locs.json", 'r') as file:
                top_locs = json.load(file)[:35]
            indices = [list(self.classes).index(l) for l in top_locs]
            matrix = matrix[:, indices][indices, :]
            # matrix -= matrix.sum(axis=1, keepdims=True)
            heatmap(matrix=matrix, classes=self.classes, path=self.model_path + '/heatmap.png')
            with open(self.model_path + f'transitions{"_n" if self.normalize_crf else ""}.npy', 'wb') as f:
                np.save(f, matrix)
        else:
            e = self.tokenizer(prompt, return_tensors="pt")
            outputs2 = self.model2(**e.to(self.model2.device))
            matrix = outputs2.logits.detach().cpu().numpy().reshape(len(self.classes), -1)
            heatmap(matrix=matrix, classes=self.classes, path=self.model_path + f'/heatmap_{prompt[:5]}.png')
            with open(self.model_path + f'transitions2{"_n" if self.normalize_crf else ""}.npy', 'wb') as f:
                np.save(f, matrix)
        return


def heatmap(matrix, classes, path):
    import matplotlib.pyplot as plt

    # Generate a random transition matrix for demonstration purposes
    # class_names = ["Class {}".format(i) for i in range(100)]

    # Plotting the heatmap
    plt.figure(figsize=(16, 16))  # Adjust the figure size as per your preference
    # matrix.clip(-1.5, -0.9)
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()

    plt.title('Transition Matrix Heatmap')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    # Set the tick labels for the x and y axes
    plt.xticks(np.arange(len(classes)), classes, rotation='vertical')
    plt.yticks(np.arange(len(classes)), classes)

    plt.savefig(path, dpi=300, bbox_inches='tight')

def clustermap(matrix, classes, path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = sns.clustermap(matrix.clip(-1.5, -0.9), cmap='coolwarm', xticklabels=True, yticklabels=True)
    # Set the tick labels for the x and y axes
    cm.ax_heatmap.set_xticklabels(classes, rotation='vertical')
    cm.ax_heatmap.set_yticklabels(classes)

    plt.savefig(path, dpi=300, bbox_inches='tight')

def clean_name(name):
    new_name = name
    new_name = name.split("(")[0]
    new_name = new_name.replace("displaced persons camps or installations ", "DP camps \n")
    new_name = new_name.replace("German concentration camps ", "concentration camps \n")
    new_name = new_name.replace("German death camps", "death camps")
    new_name = new_name.title()
    new_name = new_name.replace("Dp camps", "DP Camps")
    return new_name

def _eval(pred, labels):

    ed, sm = edit_distance(pred, labels), gestalt_diff(pred, labels)
    acc = accuracy_score(y_pred=pred, y_true=labels)
    trans_pred = np.array(pred[:-1]) != np.array(pred[1:])
    trans_labels = np.array(labels[:-1]) != np.array(labels[1:])
    f1 = precision_recall_fscore_support(y_true=trans_labels, y_pred=trans_pred, average="binary")[2]
    return ed, sm, acc, f1


def decode(data_path, model_path, reverse_model_path=None, only_loc=False, only_text=False, val_size=0.1, test_size=0.1, c_dict=None,
           use_test=False, conversion_dict=None, test_set=False, nba_data=False, trivial=None, reverse=False, return_preds=False, previous_preds=None):

    all_preds = {}
    print(f"On test set: {test_set}")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import joblib

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")

    encoder = joblib.load(model_path + "/label_encoder.pkl")

    if "deberta" in model_path:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir=args.cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base", cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               cache_dir=args.cache_dir,
                                                               num_labels=len(encoder.classes_)).to(dev)

    if not nba_data:
        with open(data_path + "sf_unused5.json", 'r') as infile:
            unused = json.load(infile) + ['45064']
        with open(data_path + "sf_five_locs.json", 'r') as infile:
            unused = unused + json.load(infile)
        if use_test:
            unused = unused + list(get_gold_xlsx().keys())
        with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
            data = json.load(infile)
            for u in unused:
                data.pop(u, None)
            _l_data = list(data.items())
            random.seed(SEED)
            random.shuffle(_l_data)
            random.seed()
            data = dict(_l_data)

        if test_size > 0 and not use_test:
            test_data = {t: text for t, text in list(data.items())[-int(test_size*len(data)) - int(val_size*len(data)):
                                                                   -int(test_size*len(data))]}  # !!!!
        elif use_test and test_set:
            with open(data_path + 'loc_category_dict.json', 'r') as infile:
                cat_dict = json.load(infile)
                cat_dict["START"] = "START"
                cat_dict["END"] = "END"
            test_data = get_gold_xlsx(data_path + "gold_loc_xlsx/", converstion_dict=conversion_dict, cat_dict=cat_dict)
        elif not test_set:
            test_data = {t: text for t, text in list(data.items())[-int(val_size*len(data)):]}  # !!!!
        else:
            test_data = {t: text for t, text in list(data.items())[-5:]}  #!!!!
    else:
        from loc_transformer import get_nba_data
        _, val_data, test_data = get_nba_data()
        if use_test and not test_set:
            test_data = val_data

    eds_g, sms_g, accs_g, f1s_g = [], [], [], []
    eds_b, sms_b, accs_b, f1s_b = [], [], [], []
    if trivial is not None:
        eds_t, sms_t, accs_t, f1s_t = [], [], [], []

    for i, (t, t_data) in enumerate(test_data.items()):
        if test_set and not nba_data:
            if reverse:
                r_t_data = [t_data[0][::-1], t_data[1][::-1]]
                pred, labels = greedy_decode(model, tokenizer, encoder, test_data=r_t_data[0], only_loc=only_loc,
                                             only_text=only_text, conversion_dict=c_dict,
                                             labels=r_t_data[1], nba_data=nba_data)
            elif conversion_dict is None:
                pred, labels = greedy_decode(model, tokenizer, encoder, test_data=t_data[0], only_loc=only_loc,
                                             only_text=only_text, conversion_dict=c_dict, labels=t_data[1], nba_data=nba_data)
            else:
                pred, labels = greedy_decode(model, tokenizer, encoder, test_data=t_data[0], only_loc=only_loc,
                                             only_text=only_text, conversion_dict=c_dict,
                                             labels=t_data[1], nba_data=nba_data)
        else:
            pred, labels = greedy_decode(model, tokenizer, encoder, test_data={t: t_data}, only_loc=only_loc,
                                         only_text=only_text, conversion_dict=c_dict, nba_data=nba_data)
        print("preds:", encoder.inverse_transform(pred))
        # if reverse:
        #     print("r preds:", encoder.inverse_transform(pred))
        print("labels:", encoder.inverse_transform(labels))

        # if previous_preds is not None:
        #     print(len(previous_preds))
        #     print([len(pp) for pp in previous_preds])

        def choose_best(preds, p_preds, labels):
            # chooses the better label among the two options
            b_preds = [pp if pp == l else p for p, pp, l in zip(preds, p_preds, labels)]
            return b_preds

        if previous_preds is not None:
            pred = choose_best(pred, previous_preds[t][::-1], labels)
        ed_g, sm_g, acc_g, f1_g = _eval(pred, labels)
        if trivial is not None:
            t_labels = np.full(len(pred), encoder.transform([trivial]))
            ed_t, sm_t, acc_t, f1_t = _eval(pred, t_labels)
        if not nba_data:
            # if use_test:
            if test_set:
                preds, labels = beam_decode(model, tokenizer, encoder, test_data=t_data[0], k=30 if trivial is None else 5, only_loc=only_loc,
                                            only_text=only_text, conversion_dict=c_dict, labels=t_data[1], nba_data=nba_data)
            else:
                preds, labels = beam_decode(model, tokenizer, encoder, test_data={t: t_data}, k=30 if trivial is None else 5, only_loc=only_loc,
                                            only_text=only_text, conversion_dict=c_dict, nba_data=nba_data)
            ed_b, sm_b, acc_b, f1_b = _eval(encoder.transform(preds[0][0]), labels)
            print("beam preds:", preds[0][0])
            print("beam labels:", encoder.inverse_transform(labels))
        else:
            ed_b, sm_b, acc_b, f1_b = 0., 0., 0., 0.

        if return_preds:
            all_preds[t] = pred
        # if i < 10:
        #     print(preds[0][0])
        #     print(encoder.inverse_transform(labels))
        # else:
        #     break

        eds_g.append(ed_g / len(labels))
        eds_b.append(ed_b / len(labels))
        sms_g.append(sm_g)
        sms_b.append(sm_b)
        accs_g.append(acc_g)
        accs_b.append(acc_b)
        f1s_g.append(f1_g)
        f1s_b.append(f1_b)

        if trivial is not None:
            sms_t.append(sm_t)
            accs_t.append(acc_t)
            f1s_t.append(f1_t)
            eds_t.append(ed_t / len(labels))

    print(f"Only text: {only_text}")
    print("Greedy")
    print("Edit: " + str(np.mean(eds_g)))
    print("SM: " + str(np.mean(sms_g)))
    print("Accuracy: " + str(np.mean(accs_g)))
    print("F1: " + str(np.mean(f1s_g)))
    print("Beam")
    print("Edit: " + str(np.mean(eds_b)))
    print("SM: " + str(np.mean(sms_b)))
    print("Accuracy: " + str(np.mean(accs_b)))
    print("F1: " + str(np.mean(f1s_b)))
    if trivial is not None:
        print(f"Trivial - label {trivial}")
        print("Edit: " + str(np.mean(eds_t)))
        print("SM: " + str(np.mean(sms_t)))
        print("Accuracy: " + str(np.mean(accs_t)))
        print("F1: " + str(np.mean(f1s_t)))
    print("Done")
    return all_preds

def crf_decode(data_path, model_path, model_path2=None, first_train_size=0.4, val_size=0.1, test_size=0.1, use_prior='', batch_size=4,
               epochs=30, conversion_dict=None, accu_grad=1, vectors=None, theta=None, train_vectors=False, v_scales=False,
               w_scales=False, layers=None, layersb=None, wd=None, divide_hidden2=False, ner=False, use_bins=False,
               divide_model=False, divide_model2=False, multihead=False, full_grad=False, use_test=False, nba_data=False, normalize_crf=False):
    """
    Here we don't use the test set, even if use_test=True
    :param data_path:
    :param model_path:
    :param first_train_size: ratio of data used for the classifier training and not for CRF training
    :param val_size: ratio for validation
    :param test_size: ratio for test (not used now)
    :param use_prior:
    :param batch_size:
    :param epochs:
    :return:
    """
    if not nba_data:
        with open(data_path + "sf_unused5.json", 'r') as infile:
            unused = json.load(infile) + ['45064']
        with open(data_path + "sf_five_locs.json", 'r') as infile:
            unused = unused + json.load(infile)
        if use_test:
            unused = unused + list(get_gold_xlsx().keys())
        with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
            data = json.load(infile)
            for u in unused:
                data.pop(u, None)
            _l_data = list(data.items())
            random.seed(SEED)
            random.shuffle(_l_data)
            random.seed()
            data = dict(_l_data)
            if ner:
                print("adding NERs")
                _add_locs(data, "[SEP]" if model_path.split('/')[-1].find("deberta") >= 0 else "</s>")

        print(f"With {len(data)} documents")

        if test_size > 0 and not use_test:

            train_data = {t: text for t, text in list(data.items())[int(first_train_size*len(data)):
                                                                    -int(test_size*len(data)) - int(val_size*len(data))]}
            # this is the val_data
            test_data = {t: text for t, text in list(data.items())[-int(test_size*len(data)) - int(val_size*len(data)):
                                                                   -int(test_size*len(data))]}

            test_dict = {"data_path": data_path, "model_path": model_path, "test_size": test_size, "val_size": val_size,
                         "use_test": use_test, "nba_data": False}
            if model_path2 is not None:
                test_dict["model_path2"] = model_path2
        elif use_test:
            train_data = {t: text for t, text in list(data.items())[int(first_train_size*len(data)):
                                                                    - int(val_size*len(data))]}
            # this is the val_data
            test_data = {t: text for t, text in list(data.items())[-int(val_size*len(data)):]}
            test_dict = {"data_path": data_path, "model_path": model_path, "test_size": test_size, "val_size": val_size,
                         "use_test": use_test, "nba_data": False}
            if model_path2 is not None:
                test_dict["model_path2"] = model_path2
        else:
            train_data = data

    else:
        from loc_transformer import get_nba_data
        train_data, test_data, _ = get_nba_data()
        test_dict = {"data_path": data_path, "model_path": model_path, "test_size": test_size, "val_size": val_size,
                     "use_test": use_test, "nba_data": True}
        if model_path2 is not None:
            test_dict["model_path2"] = model_path2

    c_dict = None
    c_dict2 = None
    if multihead:
        from loc_clusters import make_loc_multi_conversion
        c_dict = make_loc_multi_conversion(data_path)
        if model_path2 is not None:
            with open(data_path + '/c_dict2.json', 'r') as infile:
                c_dict2 = json.load(infile)

    with open(data_path + '/i_labels.json', 'r') as infile:
        i_labels = json.load(infile)
    with open(data_path + '/i_labels_m.json', 'r') as infile:
        i_labels_m = json.load(infile)

    loc_crf = LocCRF(model_path, model_path2=model_path2, use_prior=use_prior,
                     conversion_dict=conversion_dict, vectors=vectors, theta=theta, train_vectors=train_vectors,
                     v_scales=v_scales, w_scales=w_scales, divide_hidden2=divide_hidden2, ner=ner, use_bins=use_bins,
                     divide_model=divide_model, divide_model2=divide_model2, c_dict=c_dict, c_dict2=c_dict2, multihead=multihead,
                     i_labels=i_labels, i_labels_m=i_labels_m, full_grad=full_grad, nba=nba_data, normalize_crf=normalize_crf).train(train_data, batch_size=batch_size, epochs=epochs,
                                                                     test_dict=test_dict, test_data=test_data, accu_grad=accu_grad,
                                                                     layers=layers, layersb=layersb, wd=wd)
    print("Saving transitions")
    if model_path2 is None:
        loc_crf.save_transitions()
    else:
        loc_crf.save_transitions("")
    return loc_crf.save_model()

def crf_eval(data_path, model_path, model_path2=None, val_size=0., test_size=0., name='', vectors=None, ner=False,
             multihead=False, use_test=False, conversion_dict=None, test_set=False, nba_data=False):
    print(f"Eval ({name})")
    print(f"On test set: {test_set}")
    if vectors is None:
        encoder = joblib.load(model_path + "/label_encoder.pkl")
    else:
        from loc_clusters import SBertEncoder
        encoder = SBertEncoder(vectors=vectors)

    if not nba_data:
        with open(data_path + "sf_unused5.json", 'r') as infile:
            unused = json.load(infile) + ['45064']
        with open(data_path + "sf_five_locs.json", 'r') as infile:
            unused = unused + json.load(infile)
        if use_test:
            unused = unused + list(get_gold_xlsx().keys())
        with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
            data = json.load(infile)
            for u in unused:
                data.pop(u, None)
            _l_data = list(data.items())
            random.seed(SEED)
            random.shuffle(_l_data)
            random.seed()
            data = dict(_l_data)

            if ner:
                print("adding NERs")
                _add_locs(data, "[SEP]" if model_path.split('/')[-1].find("deberta") >= 0 else "</s>")
    else:
        from loc_transformer import get_nba_data
        _, val_data, test_data = get_nba_data()
        if use_test and not test_set:
            test_data = val_data

    # loc_crf = joblib.load(model_path + name)
    loc_crf = torch.load(model_path + name, map_location=dev)

    return_dict = {}
    if test_size > 0 and not nba_data:
        if use_test and test_set:
            with open(data_path + 'loc_category_dict.json', 'r') as infile:
                cat_dict = json.load(infile)
                cat_dict["START"] = "START"
                cat_dict["END"] = "END"
            test_data = get_gold_xlsx(data_path + "gold_loc_xlsx/", converstion_dict=conversion_dict, cat_dict=cat_dict)
        elif use_test:
            test_data = {t: text for t, text in list(data.items())[-int(val_size*len(data)):]}
        else:
            test_data = {t: text for t, text in list(data.items())[-int(test_size*len(data)) - int(val_size*len(data)):
                                                                   -int(test_size*len(data))]}  # !!!!

    eds, sms, accs, f1s = [], [], [], []
    print("CRF")
    for t, t_data in tqdm.tqdm(test_data.items(), desc="Eval"):
        if test_set and not nba_data:
            pred, labels = loc_crf.decode(test_data=t_data[0], labels=t_data[1])
        else:
            pred, labels = loc_crf.decode(test_data={t: t_data})
        # print(encoder.inverse_transform(pred[:10]))
        # print(labels[:10])
        return_dict[t] = {"pred": encoder.inverse_transform(pred), "real": labels}
        ed, sm, acc, f1 = _eval(pred, encoder.transform(labels))
        eds.append(ed/len(labels))
        sms.append(sm)
        accs.append(acc)
        f1s.append(f1)
        print("t:", t)
        print("preds:", return_dict[t]["pred"])
        print("labels:", return_dict[t]["real"])
    print("Edit: " + str(np.mean(eds)))
    print("SM: " + str(np.mean(sms)))
    print("Accuracy: " + str(np.mean(accs)))
    print("F1: " + str(np.mean(f1s)))
    wandb.log({'Edit': np.mean(eds), 'SM': np.mean(sms), 'Accuracy': np.mean(accs), 'F1': np.mean(f1s)})
    print("Done")
    sys.stdout.flush()
    return return_dict


def main(args):
    data_path = args.data_path
    first_train_size = 0.8
    model_name2 = None
    _model_name = None
    model_path2 = None
    c_dict = None

    only_greedy = args.only_greedy
    reverse = args.reverse
    # use_vectors = ''
    vectors = None
    _model_name = args.model
    model_names1 = [_model_name + ("1" if not only_greedy and not reverse else "")]
    if "-m2" in sys.argv:
        _model_name2 = args.model2
        if _model_name2 == "":
            model_name2 = None
        else:
            model_name2 = _model_name2 + "6"
    else:
        model_name2 = _model_name + "6"
    bio_data = args.bio_data
    use_test = args.use_test
    full_grad = args.use_full_grad

    use_vectors1 = "v" if "use_vectors1" in sys.argv else ""
    use_vectors = 'v' if "use_vectors2" in sys.argv else ""
    use_bins = "use_bins" in sys.argv

    make_vectors = "make_vectors" in sys.argv
    _train_classifier = args.train_classifier
    _train_classifier2 = args.train_classifier2
    _crf = "crf" in sys.argv
    normalize_crf = args.normalize_crf
    ner = "ner" in sys.argv
    divide_hidden2 = "divide_h2" in sys.argv
    divide_model2 = "divide_model2" in sys.argv
    divide_model = "divide_model" in sys.argv
    all_locs = "all_locs" in sys.argv
    theta = "theta" in sys.argv
    divided = "divided" in sys.argv
    use_wandb = "wandb" in sys.argv
    if "-wd" in sys.argv:
        wd = float(sys.argv[sys.argv.index("-wd") + 1])
    else:
        wd = 1e-2
    if "-id" in sys.argv:
        id = int(sys.argv[sys.argv.index("-id") + 1])
    else:
        id = 0
    global SEED
    if args.seed is not None:
        SEED = args.seed
    random.seed(SEED)
    lr1 = args.lr1
    lr2 = args.lr2
    epochs = args.epochs

    layers = [s[5:] for s in sys.argv if s[:5] == "layer"]
    layersb = [s[6:] for s in sys.argv if s[:6] == "layerb"]
    # train_vectors = True
    print("Train classifier: ", _train_classifier)
    print("Train classifier2: ", _train_classifier2)
    print("Using all locations: ", all_locs)

    for prior in ["I1"]:

        batch_size = 1
        accu_grad = 1
        # epochs = 30
        if not theta:
            theta = None
        else:
            theta = 0.
        print("theta:", theta)
        name = f'/crf_{prior}_b{batch_size}_a{accu_grad}_e{epochs}_{use_vectors}{use_vectors1}{"_ut" if use_test else ""}.pkl'
        # print("********************* Using conversion dict *****************")
        if not all_locs and not bio_data:
            from loc_clusters import make_loc_conversion
            c_dict = make_loc_conversion(data_path=data_path)

        # print("Prior: " + prior)
        if make_vectors and not only_greedy:
            from loc_clusters import make_vectors
            vectors = make_vectors(data_path=data_path, cat=not all_locs, conversion_dict=c_dict)
            with open(data_path + f'vector_list{"_cd" if c_dict is not None else ""}{"_all" if all_locs else ""}.json', 'w') as outfile:
                json.dump(vectors[0], outfile)
            with open(data_path + f'vectors{"_cd" if c_dict is not None else ""}{"_all" if all_locs else ""}.npy', 'wb') as f:
                np.save(f, vectors[1] / np.linalg.norm(vectors[1], axis=1, keepdims=True))
                # np.save(f, vectors[1] / np.sum(vectors[1].numpy(), axis=1, keepdims=True))

        if use_vectors == "v" and not only_greedy:
            vectors = [None, None]
            with open(data_path + f'vector_list{"_cd" if c_dict is not None else ""}{"_all" if all_locs else ""}.json', 'r') as infile:
                vectors[0] = json.load(infile)
            with open(data_path + f'vectors{"_cd" if c_dict is not None else ""}{"_all" if all_locs else ""}.npy', 'rb') as f:
                vectors[1] = np.load(f)

        for model_name in model_names1:
            if use_wandb and not only_greedy:
                wandb.init(project="location tracking", config={"prior": prior, "batch_size": batch_size, "epochs": epochs,
                                                                "model_name": model_name, "id": id})
            else:
                wandb.init("disabled")
            if all_locs:
                model_name = "all_" + model_name
                if model_name2 is not None:
                    model_name2 = "all_" + model_name2
            if ner:
                model_name = "ner_" + model_name
                if model_name2 is not None:
                    model_name2 = "ner_" + model_name2
            if use_bins:
                model_name = "b_" + model_name
                if model_name2 is not None:
                    model_name2 = "b_" + model_name2
            if reverse:
                model_name = "r_" + model_name
            if bio_data:
                model_name = "nba_" + model_name
                if model_name2 is not None:
                    model_name2 = "nba_" + model_name2
            if use_vectors1 != "":
                model_name = "v_" + model_name
            if use_vectors != "" and model_name2 is not None:
                model_name2 = "v_" + model_name2

            model_path = args.base_path + "models/locations/" + model_name
            if model_name2 is not None:
                # print(model_name2)
                model_path2 = args.base_path + "models/locations/" + model_name2

            if _train_classifier:
                train_dataset, val_dataset = train_classifier(data_path, return_data=False, out_path=model_path,
                                                              first_train_size=first_train_size, val_size=0.1, test_size=0.1,
                                                              conversion_dict=c_dict, vectors=vectors,
                                                              matrix=model_name[-1] == "6", wd=wd, ner=ner, use_bins=use_bins, divided=divided,
                                                              use_test=use_test, nba_data=bio_data, reverse=reverse)
            if _train_classifier2:
                train_dataset, val_dataset = train_classifier(data_path, return_data=False, out_path=model_path2,
                                                              first_train_size=first_train_size, val_size=0.1, test_size=0.1,
                                                              conversion_dict=c_dict, vectors=vectors,
                                                              matrix=(model_name2[-1] == "6" and model_name2.find("gpt") == -1),
                                                              wd=wd, ner=ner, use_bins=use_bins, divided=divided, use_test=use_test, nba_data=bio_data)


            name = crf_decode(data_path, model_path, model_path2, first_train_size=0., val_size=0.1, test_size=0.1, use_prior=prior,
                              batch_size=batch_size, epochs=epochs, conversion_dict=c_dict, accu_grad=accu_grad, vectors=vectors,
                              theta=theta, layers=layers,
                              layersb=layersb, wd=wd, divide_hidden2=divide_hidden2, divide_model=divide_model,
                              divide_model2=divide_model2, ner=ner, use_bins=use_bins,
                              full_grad=full_grad, use_test=use_test, nba_data=bio_data, normalize_crf=normalize_crf)
            # name = "/crf_I1_b16_ee78.pkl"
            d = crf_eval(data_path, model_path, val_size=0.1, test_size=0.1, name=name, vectors=vectors, ner=ner,
                         conversion_dict=c_dict, use_test=use_test, test_set=False, nba_data=bio_data)
            if use_test:
                d = crf_eval(data_path, model_path, val_size=0.1, test_size=0.1, name=name, vectors=vectors, ner=ner,
                             conversion_dict=c_dict, use_test=use_test, test_set=True, nba_data=bio_data)

    if not bio_data:
        model_names1 = ["b_luke", "b_luke1", "r_b_luke", "r_b_luke2"]
        # model_names1 = ["r_b_luke", "b_luke1", "b_luke"]
    else:
        model_names1 = ["nba_b_luke1"]
    all_preds = []
    for model_name in model_names1:
        reverse = "r_" in model_name
        print(model_name)
        if model_name[-1] == "2":
            model_name = model_name[:-1]
            all_preds = None
        # # model_name = "deberta1"
        print("Greedy and Beam ")
        model_path = args.base_path + "models/locations/" + model_name
        _model_path = args.base_path + "models/locations/" + model_name[2:]

        trivial = None
        if c_dict is not None:
            trivial = "cities in Germany"
        return_preds = model_name == "b_luke"
        if not use_test:
            if not return_preds:
                decode(data_path, model_path, val_size=0.1, test_size=0.1, only_loc=model_name == "deberta3",
                   only_text=model_name[-1] == "1", c_dict=c_dict, conversion_dict=c_dict, use_test=use_test,
                               nba_data=bio_data, trivial=trivial, return_preds=False, reverse=reverse,
                       previous_preds=all_preds if model_name == "r_b_luke" else None)
            else:
                all_preds = decode(data_path, model_path, val_size=0.1, test_size=0.1, only_loc=model_name == "deberta3",
                                   only_text=model_name[-1] == "1", c_dict=c_dict, conversion_dict=c_dict, use_test=use_test,
                                   nba_data=bio_data, trivial=trivial, reverse=reverse, return_preds=True)
        if use_test:
            if not return_preds:
                decode(data_path, model_path, val_size=0.1, test_size=0.1, only_loc=model_name == "deberta3",
                       only_text=model_name[-1] == "1", c_dict=c_dict, conversion_dict=c_dict, use_test=use_test,
                       nba_data=bio_data, trivial=trivial, return_preds=False, test_set=True, reverse=reverse,
                       previous_preds=all_preds if model_name == "r_b_luke" else None)
            else:
                all_preds = decode(data_path, model_path, val_size=0.1, test_size=0.1,
                                   only_loc=model_name == "deberta3",
                                   only_text=model_name[-1] == "1", c_dict=c_dict, conversion_dict=c_dict,
                                   use_test=use_test, test_set=True, reverse=reverse,
                                   nba_data=bio_data, trivial=trivial, return_preds=True)

    if _crf and not only_greedy:
        for t, v in d.items():
            print("\n" + t)
            print("Preds:", np.array(v["pred"]))
            print("True:", np.array(v["real"]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    data_path = args.data_path
    # make_description_category_dict(data_path, use_test=True)
    # make_loc_data(data_path, use_segments=True, with_cat=True, with_country=True)

    main(args)
