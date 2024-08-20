
import spacy
from transformers import pipeline
import numpy as np
import json
from loc_transformer import get_data, _make_texts
import sys
import torch
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

from location_tracking import _eval
from utils import parse_args
args = parse_args()

class EntityLocator:
    """
    Class for a location classifier that is based on NER
    """
    def __init__(self, classes):
        self.nlp = spacy.load(args.base_path + "ner/model-best")
        self.classifier = pipeline(model="facebook/bart-large-mnli", device=dev)  # zshot classification
        self.classes = classes

    def get_segments_ents(self, text):
        """
        Extract all entities using Spacy
        """
        doc = self.nlp(text)
        return doc.ents

    def predict_segment(self, text, ents, threshold=0.):
        """
        Predict location for segment, after adding the entities to the input
        """
        s_token = self.classifier.tokenizer.sep_token
        input = s_token.join([e.text for e in ents])
        out = self.classifier(input, candidate_labels=self.classes)
        if np.all(np.array(out["scores"]) < threshold):
            return None
        return self.classes.index(out["labels"][0])
        # return np.argmax(out["scores"])

    def predict(self, texts):
        locs = [self.classes.index("START")]

        for text in texts:
            ents = self.get_segments_ents(text)
            if len(ents) == 0:  # stay in same place
                locs.append(locs[-1])
            else:
                pred = self.predict_segment(text, ents)
                locs.append(pred if pred is not None else locs[-1])
        return locs[:-1]  # ???

    def convert_to_labels(self, preds):
        return [self.classes[p] for p in preds]


def evaluate(model, val_data=None):
    """
    Get scores for the model
    """
    eds, sms, accs, f1s = [], [], [], []
    for t, v in val_data.items():
        print(f"Testimony: {t}")
        texts, labels = _make_texts({t: v}, [], out_path="_1", use_bins=False)
        _labels = [model.classes.index(l) for l in labels]
        preds = model.predict(texts)
        print("Labels:")
        print(labels)
        print("Preds:")
        print(model.convert_to_labels(preds))
        ed, sm, acc, f1 = _eval(preds, _labels)
        eds.append(ed / len(_labels))
        sms.append(sm)
        accs.append(acc)
        f1s.append(f1)
    print(f"means (ed, sm, acc, f1): {np.mean(eds)}, {np.mean(sms)}, {np.mean(accs)}, {np.mean(f1s)}")
    sys.stdout.flush()


def main():
    base_path = args.base_path
    data_path = base_path + 'data/'
    train_data, val_data, _ = get_data(data_path=data_path)

    with open(data_path + 'loc_category_dict.json', 'r') as infile:
        cat_dict = json.load(infile)
        cat_dict["START"] = "START"
        cat_dict["END"] = "END"
        classes = list(cat_dict.keys())
        classes.sort()
        cats = list(set(cat_dict.values()))
        cats.sort()

    with torch.no_grad():
        el = EntityLocator(cats)
        evaluate(el, val_data=val_data)
    print("Done")


if __name__ == "__main__":
    main()
