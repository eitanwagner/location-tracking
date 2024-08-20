

import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
import json
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

from loc_evaluation import edit_distance
from sentence_transformers import SentenceTransformer, util

from utils import parse_args
args = parse_args()

# **************

def make_rand_vectors(data_path):
    with open(data_path + 'loc_description_dict.json', 'r') as infile:
        desc_dict = json.load(infile)
        desc_dict["START"] = "The beginning of the testimony."
        desc_dict["END"] = "The end of the testimony."
    embeddings = torch.rand(len(desc_dict), 64) * 2 - 1
    return list(desc_dict.keys()), embeddings

def make_vectors(data_path, cat=False, conversion_dict=None, normalize=False):
    with open(data_path + 'loc_description_dict.json', 'r') as infile:
        desc_dict = json.load(infile)
        desc_dict["START"] = "The beginning of the testimony."
        desc_dict["END"] = "The end of the testimony."
    if cat:
        with open(data_path + 'loc_category_dict.json', 'r') as infile:
            cat_dict = json.load(infile)
            cat_dict["START"] = "START"
            cat_dict["END"] = "END"
        cat_list = {conversion_dict.get(c): [] for c in set(cat_dict.values())}
        for l, c in cat_dict.items():
            cat_list[conversion_dict.get(c)].append(l)
        cat_list.sort()

    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    if cat:
        locs = []
        vecs = []
        for c, l in cat_list.items():
            embeddings = model.encode([_l + desc_dict.get(_l, "") for _l in l], convert_to_tensor=False, normalize_embeddings=normalize)
            vecs.append(embeddings.mean(axis=0))
            locs.append(c)
        return locs, np.stack(vecs, axis=0)

    locs = list(desc_dict.keys())
    locs.sort()
    embeddings = model.encode([l + ": " + desc_dict[l] for l in locs], convert_to_tensor=False, normalize_embeddings=normalize)
    return locs, embeddings


def find_closest(locs=None, vectors=None, c_vector=None, tensor=False, v_scales=False):
    if v_scales:
        sims = util.dot_score(vectors, c_vector)
    else:
        sims = util.cos_sim(vectors, c_vector)
    # return locs[np.argmax(sims)]
    if tensor:
        return torch.argmax(sims)
    return int(np.argmax(sims))


class SBertEncoder(nn.Module):
    def __init__(self, vectors=None, data_path=None, cat=False, conversion_dict=None, train_vectors=False):
        super().__init__()
        if vectors is not None:
            self.classes_ = vectors[0]
            self.vectors = vectors[1]
            if train_vectors:
                self.vectors = nn.Parameter(torch.from_numpy(self.vectors))
                self.vectors.requires_grad = True
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            if data_path is None:
                base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
                data_path = base_path + 'data/'
            with open(data_path + 'loc_description_dict.json', 'r') as infile:
                desc_dict = json.load(infile)
                desc_dict["START"] = "The beginning of the testimony."
                desc_dict["END"] = "The end of the testimony."
                self.desc_dict = desc_dict
            if cat:
                with open(data_path + 'loc_category_dict.json', 'r') as infile:
                    cat_dict = json.load(infile)
                    cat_dict["START"] = "START"
                    cat_dict["END"] = "END"
                cat_list = {conversion_dict.get(c): [] for c in set(cat_dict.values())}
                for l, c in cat_dict.items():
                    cat_list[conversion_dict.get(c)].append(l)
                self.cat_list = cat_list
                dict = {}
                for c, l in cat_list.items():
                    # embeddings = model.encode([_l + desc_dict.get(_l, "") for _l in l], convert_to_tensor=True)
                    embeddings = self.model.encode([_l + desc_dict.get(_l, "") for _l in l], convert_to_tensor=True)
                    dict[c] = embeddings.mean(dim=0)
                self.classes_, self.vectors = list(dict.keys()), torch.stack(list(dict.values()), dim=0)
                self.vectors.requires_grad = True

        # self.vector_dict = {c: v for c, v in zip(vectors)}

    def inverse_transform(self, v_labels):
        """
        from label numbers to label strings
        :param labels:
        :param tensors:
        :return:
        """
        return [self.classes_[id] for id in v_labels]

    def inverse_transform2(self, v_labels, tensors=True):
        """
        from label vectors to label strings
        :param labels:
        :param tensors:
        :return:
        """
        label_ids = torch.stack([find_closest(vectors=self.vectors, c_vector=v, tensor=True) for v in v_labels])
        return [self.classes_[id] for id in label_ids]

    def transform(self, labels):
        """
        from label strings to label number
        :return:
        """
        return [self.classes_.index(l) for l in labels]

    def transform2(self, labels):
        """
        from label strings to label vectors
        :return:
        """
        return [self.vectors[self.classes_.index(l)] for l in labels]



# *************

def make_loc_conversion(data_path):
    df = pd.read_csv(data_path + "loc_conversion.csv", names=["from", "to"])
    return dict(zip(df["from"], df["to"]))

def make_loc_conversion2(data_path):
    df = pd.read_csv(data_path + "loc_conversion2.csv", names=["from", "to"])
    return dict(zip(df["from"], df["to"]))

def make_loc_multi_conversion(data_path):
    df = pd.read_csv(data_path + "classes_type_country.csv", names=["Label", "Type", "Country"])
    return {l: [t, c] for l, t, c in zip(df["Label"], df["Type"], df["Country"]) if l != "Label"}

def remove_duplicates(sequences):
    from itertools import groupby
    return [[_s[0] for _s in groupby(s)] for s in sequences]

def clustering(sequences, dist_func):
    similarity = -1 * np.array([[dist_func(s1, s2) for s1 in sequences] for s2 in tqdm(sequences)])

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, preference=-80)
    affprop.fit(similarity)

    centroids = {}
    for cluster_id in np.unique(affprop.labels_):
        exemplar = sequences[affprop.cluster_centers_indices_[cluster_id]]
        # cluster = np.unique(sequences[np.nonzero(affprop.labels_ == cluster_id)])
        in_cluster_ids = np.nonzero(affprop.labels_ == cluster_id)[0]

        centroids[cluster_id] = in_cluster_ids[np.argmax([np.mean(similarity[id, in_cluster_ids]) for id in in_cluster_ids])]

    print("Centriods:")
    for id, c in centroids.items():
        print(id)
        print(sequences[c])
    print("Done")

def load_data(data_path, train_size=0.8, conversion_dict=None):
    from location_tracking import _make_texts
    with open(data_path + "sf_unused5.json", 'r') as infile:
        unused = json.load(infile) + ['45064']
    with open(data_path + "sf_five_locs.json", 'r') as infile:
        unused = unused + json.load(infile)
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
        for u in unused:
            data.pop(u, None)

        train_data = {t: text for t, text in list(data.items())[:int(train_size * len(data))][:]}  # !!!!

    labels = [_make_texts({t: t_data}, unused, "1", conversion_dict=conversion_dict)[1] for t, t_data in train_data.items()]
    return labels


problematic = []
def _parse(cat, desc):
    from lat_lon_parser import parse
    import wptools

    _d = desc.split()
    if len(_d) > 0 and _d[0] == "Definition:":
        _d = _d[1:]
    c_camp = cat.find("Concentration Camp") >= 0
    d_camp = cat.find("Death Camp") >= 0
    r_camp = cat.find("Refugee Camp") >= 0
    dp_camp = cat.find("DP Camp") >= 0
    pow_camp = cat.find("POW Camp") >= 0
    ghetto = cat.find("Ghetto") >= 0
    generic = cat.find("generic") >= 0

    lat, long = None, None
    if len(_d) > 0 and _d[0] == "Coordinates:":
        if _d[1][-1] in ["N", "S"]:
            try:
                lat = parse(_d[1])
                long = parse(_d[2])
            except ValueError:
                lat, long = None, None
            if _d[2][-1] in ["N", "S"]:
                lat, long = None, None
        elif _d[1][-2] in ["N", "S"]:
            lat = parse(_d[1][:-1])
            long = parse(_d[2])
        elif _d[2][-1] in ["N", "S"]:
            lat = parse(_d[1] + _d[2])
            long = parse(_d[3] + _d[4])
        elif _d[3][-1] in ["N", "S"]:
            lat = parse(_d[1] + _d[2] + _d[3])
            long = parse(_d[4] + _d[5] + _d[6])
        else:
            lat, long = None, None


    if lat is None:
        if cat[2] == ")":
            cat = cat[3:]
        _cat = cat[:cat.find("(")-1]

        try:
            so = wptools.page(_cat).get_parse()
            d = wptools.page(wikibase=so.data['wikibase']).get_wikidata()
            if isinstance(d.data['wikidata']['coordinate location (P625)'], list):
                lat = d.data['wikidata']['coordinate location (P625)'][0]['latitude']
                long = d.data['wikidata']['coordinate location (P625)'][0]['longitude']
            else:
                lat = d.data['wikidata']['coordinate location (P625)']['latitude']
                long = d.data['wikidata']['coordinate location (P625)']['longitude']
        except (LookupError, TypeError):
            try:
                _cat = cat[cat.find("(")+1:]
                _cat = _cat[:_cat.find(")")]
                so = wptools.page(_cat[:_cat.find(":")]).get_parse()
                d = wptools.page(wikibase=so.data['wikibase']).get_wikidata()
                lat = d.data['wikidata']['coordinate location (P625)']['latitude']
                long = d.data['wikidata']['coordinate location (P625)']['longitude']
            except (LookupError, TypeError):
                try:
                    _cat2 = _cat[:_cat.find(",")]
                    so = wptools.page(_cat2).get_parse()
                    d = wptools.page(wikibase=so.data['wikibase']).get_wikidata()
                    lat = d.data['wikidata']['coordinate location (P625)']['latitude']
                    long = d.data['wikidata']['coordinate location (P625)']['longitude']
                except (LookupError, TypeError):
                    # try:
                    #     _cat = cat[:cat.find("(")-1]
                    #     so = wptools.page(_cat).get_parse()
                    #     d = wptools.page(wikibase=so.data['wikibase']).get_wikidata()
                    # except (LookupError, TypeError):
                        problematic.append(cat)
                        return None, None, [c_camp, d_camp,  r_camp, dp_camp, pow_camp, ghetto]

    return lat, long, [c_camp, d_camp,  r_camp, dp_camp, pow_camp, ghetto]

def tabular_parse(data_path):
    with open(data_path + 'loc_description_dict.json', 'r') as infile:
        desc_dict = json.load(infile)
    with open(data_path + 'correction_dict.json', 'r') as infile:
        correction_dict = json.load(infile)
    desc_dict["START"] = "The beginning of the testimony."
    desc_dict["END"] = "The end of the testimony."
    # lats, longs = [], []


    t_dict = {}
    for i, (c, d) in enumerate(desc_dict.items()):
        _c = c
        if c in correction_dict:
            _c = correction_dict[c]
            if _c is None:
                t_dict[c] = [None, None, []]
                continue
        lat, long, attr = _parse(cat=_c, desc=d)
        t_dict[c] = [lat, long, attr]
    print(t_dict)
    with open(data_path + 't_dict.json', 'w') as outfile:
        json.dump(t_dict, outfile)


def main():
    base_path = args.base_path
    data_path = base_path + 'data/'

    model_name = "deberta11"
    model_path = base_path + "models/locations/" + model_name

    encoder = joblib.load(model_path + "/label_encoder.pkl")
    c_dict = None
    c_dict = make_loc_conversion(data_path=data_path)
    data = remove_duplicates(load_data(data_path, conversion_dict=c_dict))
    clustering(sequences=data, dist_func=edit_distance)

def main2():
    base_path = args.base_path
    data_path = base_path + 'data/'
    c_dict = make_loc_conversion(data_path=data_path)
    locs, vectors = make_vectors(data_path, cat=True, conversion_dict=c_dict)
    # locs, vectors = make_vectors(data_path)
    cat2id = {c: i for i, c in enumerate(locs)}
    cor_matrix = util.cos_sim(vectors, vectors).detach().numpy()
    # cor_matrix.detach().numpy()
    _c_m = {(s1, s2): cor_matrix[cat2id[s1], cat2id[s2]] for s1 in locs for s2 in locs}
    data = remove_duplicates(load_data(data_path, conversion_dict=c_dict))
    clustering(sequences=data, dist_func=lambda s1, s2: edit_distance(s1, s2, cor_matrix=_c_m))
    print("Done")

def main3():
    base_path = args.base_path
    data_path = base_path + 'data/'
    tabular_parse(data_path)

if __name__ == "__main__":
    main3()