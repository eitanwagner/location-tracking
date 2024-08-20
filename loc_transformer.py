
import argparse
import numpy as np
import json
import torch
from torch import nn
import sys
from sentence_transformers import SentenceTransformer, util
from tqdm.autonotebook import trange

import spacy
if torch.cuda.is_available():
    spacy.require_gpu()

dev1 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dev = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu")
import random

SEED = 1
TESTING = False

from loc_clusters import find_closest
from loc_evaluation import get_gold_xlsx
from location_tracking import _make_texts, Dataset, _eval
from sklearn.preprocessing import LabelEncoder

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    DebertaModel, DebertaConfig, DebertaForTokenClassification, DebertaForSequenceClassification
from transformers import LukeModel, LukeForSequenceClassification
from transformers import T5ForConditionalGeneration, T5Config
# from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

from sklearn.metrics import accuracy_score
from datasets import load_metric
import joblib
import tqdm
import math

from utils import parse_args
args = parse_args()

def _get_ent_spans(nlp, text, return_ents=False):
    doc = nlp(text)
    ent_spans = [(e.start_char, e.end_char) for e in doc.ents]
    ents = [e.text for e in doc.ents]
    if return_ents:
        return (ent_spans, ents) if len(ent_spans) > 0 else ([(0, 0)], [""])
    return ent_spans if len(ent_spans) > 0 else [(0, 0)]

# *******************

def evaluation(vectors=None, evaluate1=False, val_data=None, cat_dict=None, loc_transformer=None, conversion_dict=None,
               use_test=False):
    """
    Perform evaluation
    :param vectors:
    :param evaluate1:
    :param val_data:
    :param cat_dict: dictionary to convert location to category
    :param loc_transformer:
    :param conversion_dict: dictionary to base category to the combined one
    :param use_test: whether to use the test set. This will override val_data
    :return:
    """
    use_cats = True
    classes = list(cat_dict.keys())
    classes.sort()
    cats = list(set(cat_dict.values()))
    cats.sort()
    if evaluate1:
        eds1, sms1, accs1, f1s1 = [], [], [], []

    if use_test:
        val_data = get_gold_xlsx(cat_dict=cat_dict, converstion_dict=None)  # in this case, the cat_dict included the conversion_dict

    preds = []
    preds1 = []
    eds, sms, accs, f1s = [], [], [], []
    for t, v in val_data.items():
        print(f"Testimony: {t}")
        if not use_test:
            texts, labels = _make_texts({t: v}, [], out_path=f"{'_all' if not use_cats else ''}_1", use_bins=False,
                                        conversion_dict=conversion_dict, nba_data="nba_data" in sys.argv)
        else:
            # if conversion_dict is None:
            texts, labels = v
            # elif use_cats:
            #     texts, labels = v[0], [cat_dict[_v] for _v in v[1]]
            # else:
            #     texts, labels = [conversion_dict[_v] for _v in v]
        if vectors is not None:
            _labels = [vectors[0].index(l) for l in labels]
        else:
            _labels = [(cats if use_cats else classes).index(l) for l in labels]
        # preds.append(loc_transformer.decode(v))
        preds.append(loc_transformer.decode2(texts))
        print("Labels:")
        print(labels)
        print("Preds:")
        print(loc_transformer.convert_to_labels(preds[-1]))
        if not loc_transformer.use_ents and loc_transformer.s_transformer is None:
            preds1.append(loc_transformer.decode1(texts))
            print("Preds1:")
            print(loc_transformer.convert_to_labels(preds1[-1]))
        ed, sm, acc, f1 = _eval(preds[-1], _labels)
        eds.append(ed / len(labels))
        sms.append(sm)
        accs.append(acc)
        f1s.append(f1)
        print(f"ed, sm, acc, f1: {ed}, {sm}, {acc}, {f1}")

        if evaluate1 and not use_cats:
            _preds = loc_transformer.convert_to_labels(preds[-1])
            _preds = [cats.index(cat_dict.get(p, None)) for p in _preds]
            _labels = [cats.index(cat_dict.get(l, None)) for l in labels]
            ed, sm, acc, f1 = _eval(_preds, _labels)
            eds1.append(ed / len(_labels))
            sms1.append(sm)
            accs1.append(acc)
            f1s1.append(f1)

    print(f"means (ed, sm, acc, f1): {np.mean(eds)}, {np.mean(sms)}, {np.mean(accs)}, {np.mean(f1s)}")
    if evaluate1:
        print(f"category means (ed, sm, acc, f1): {np.mean(eds1)}, {np.mean(sms1)}, {np.mean(accs1)}, {np.mean(f1s1)}")


# *******************

def encode(s_transformer, sentences,
           batch_size: int = 32,
           convert_to_tensor: bool = True,
           output_value: str = 'sentence_embedding',
           device: str = None,
           normalize_embeddings: bool = False):
    """
    Computes sentence embeddings - with gradients!
    Adapted from the original implementation in sentence-transformers

    :return:
       By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
    """
    input_was_string = False
    if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
        sentences = [sentences]
        input_was_string = True

    if device is None and torch.cuda.is_available():
        device = s_transformer._target_device

    s_transformer.to(device)

    all_embeddings = []
    length_sorted_idx = np.argsort([-s_transformer._text_length(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    for start_index in range(0, len(sentences), batch_size):
        sentences_batch = sentences_sorted[start_index:start_index+batch_size]
        features = s_transformer.tokenize(sentences_batch)
        features = util.batch_to_device(features, device)

        out_features = s_transformer.forward(features)

        embeddings = out_features[output_value]
        # embeddings = embeddings.detach()
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
        all_embeddings.extend(embeddings)

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

    if convert_to_tensor:
        all_embeddings = torch.stack(all_embeddings)
    if input_was_string:
        all_embeddings = all_embeddings[0]

    return all_embeddings

def make_sentence_vectors(model_path='/cs/snapless/oabend/eitan.wagner/locations/output/training_nli_v2_studio-ousia-luke-base',
                          classes=None, use_categories=True, nba=False):
    """
    Make sentence vectors based on a pretrained sentence transformer
    :param model_path:
    :param classes:
    :return:
    """
    model = SentenceTransformer(model_path + ("_nba_locs" if nba else ""))
    if not nba:
        _classes = [f"The event location {'category' if use_categories else ''} is {l}" for l in classes]
    else:
        _classes = [f"The professional location is {l}" for l in classes]
    vectors = model.encode(_classes, convert_to_tensor=True, normalize_embeddings=True)
    return model, (classes, vectors)


class LocTransformer(nn.Module):
    def __init__(self, vectors=None, ner=False, use_bins=False, with_adapter=False, initialize_model=False, classes=None,
                 cats=None, cat_dict=None, use_ents=False, model_name="luke", conversion_dict=None,
                 i_loss=False, s_transformer=None):
        super().__init__()
        self.model_path = "_all_1"
        self.s_transformer = s_transformer
        self.vectors = vectors
        if vectors is not None:
            self._vectors = nn.Embedding.from_pretrained(torch.tensor(vectors[1]))
        self.ner = ner
        self.use_bins = use_bins
        self.use_ents = use_ents
        self.model_name = model_name
        self.conversion_dict = conversion_dict
        self.i_loss = i_loss

        if classes is not None:
            self.classes = classes
            classes.sort()
        else:
            self.classes = vectors[0]
        self.cats = cats
        self.cat_dict = cat_dict
        if cats is not None:
            self.classes = cats
            classes.sort()
        # self.start_id = self.classes.index("START")
        self.with_adapter = with_adapter

        if vectors is not None:
            from loc_clusters import SBertEncoder
            self.encoder = SBertEncoder(vectors=vectors)

        if model_name == "deberta":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base",
                                                           cache_dir=args.cache_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base",
                                                           cache_dir=args.cache_dir)

        # v_dim = len(vectors[1][0])

        if initialize_model and s_transformer is None:
            if model_name == "deberta":
                self._model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base",
                                                                               cache_dir=args.cache_dir,
                                                                               num_labels=len(self.classes),
                                                                               problem_type="single_label_classification")
                self.model = self._model.deberta
                self._model.pooler.to(dev1)
            elif model_name == "luke":
                self._model = LukeForSequenceClassification.from_pretrained("studio-ousia/luke-base",
                                                                      cache_dir=args.cache_dir,
                                                                      num_labels=len(self.classes),
                                                                      problem_type="single_label_classification")
                self.model = self._model.luke
            self.model.to(dev)
            # self._model.dropout = nn.Dropout(0.)
            self._model.dropout.to(dev1)
            self._model.classifier.to(dev1)

        elif s_transformer is not None:
            self.model = s_transformer
            self._model = self.model
            self.model.to(dev)
        else:
            self.model = None

        if use_ents:
            # self.intermediate_size = self.model.config.hidden_size + self.model.config.entity_emb_size
            self.intermediate_size = self.model.config.hidden_size * 2
        elif s_transformer is None:
            self.intermediate_size = self.model.config.hidden_size

        if use_ents:
            self.nlp = spacy.load(args.base_path + "ner/model-best")
            # self.ent_pooler = nn.Sequential(nn.Linear(self.model.config.entity_emb_size,
            #                                           self.model.config.entity_emb_size), nn.Tanh())
        # self.l_transformer = nn.Transformer(d_model=v_dim)
        # config = T5Config(vocab_size=len(vectors[0]), d_model=self.model.config.hidden_size)
        # config = T5Config.from_pretrained("t5-small")

        config = DebertaConfig.from_pretrained("microsoft/deberta-base")
        config.vocab_size = len(self.classes)  # not important in the encoder case
        config.num_labels = len(self.classes)  # important in the encoder case

        # config.d_model = self.model.config.hidden_size
        self.intermediate = None
        if self.vectors is not None:
            config.d_model = vectors[1].shape[1]

            if s_transformer is None and config.d_model != self.intermediate_size:
                self.intermediate = nn.Linear(self.intermediate_size, config.d_model)
                self.intermediate.to(dev1)
        elif use_ents:
            config.d_model = self.intermediate_size
            config.hidden_size = self.intermediate_size

        # self.l_transformer = T5ForConditionalGeneration(config=config)
        self.l_transformer = DebertaForTokenClassification(config=config)
        # lm_head = nn.Linear(vectors[1].shape[1], vectors[1].shape[0])
        if self.vectors is not None:
            with torch.no_grad():
                # lm_head.weight.copy_(torch.tensor(vectors[1]))
                self.l_transformer.classifier.weight.copy_(torch.tensor(vectors[1]))
        # self.l_transformer.set_output_embeddings(lm_head)
        self.l_transformer.to(dev1)

        # what about positional embeddings?? - no need

    def forward1(self, texts, labels=None, batch_size=4, batch_id=None, return_logits=False, changes=None):
        """
        First part of the forward
        :param texts:
        :param labels: if None then used for inference
        :return:
        """
        # b_texts = np.split(texts[0], np.arange(batch_size, len(texts[0]), batch_size))
        b_texts = [texts[0][b_start: b_start+batch_size] for b_start in range(0, len(texts[0]), batch_size)]
        if labels is not None:
            b_labels = [labels[0][b_start: b_start+batch_size] for b_start in range(0, len(labels[0]), batch_size)]
        if changes is not None:
            b_changes = [changes[b_start: b_start+batch_size] for b_start in range(0, len(changes), batch_size)]

        if self.s_transformer:
            if batch_id is None:
                # return torch.vstack([self.model.encode(b_text, convert_to_tensor=True, normalize_embeddings=True) for b_text in b_texts])
                return torch.vstack([encode(self.model, b_text, convert_to_tensor=True, normalize_embeddings=True) for b_text in b_texts])
            else:
                # return self.model.encode(b_texts[batch_id], convert_to_tensor=True, normalize_embeddings=True), 0.
                return encode(self.model, b_texts[batch_id], convert_to_tensor=True, normalize_embeddings=True), 0.

        if self.use_ents:
            if batch_id is None:
                # entity_spans = [[_get_ent_spans(self.nlp, text) for text in b_text] for b_text in b_texts]
                _entity_spans = [[_get_ent_spans(self.nlp, text, return_ents=True) for text in b_text] for b_text in b_texts]
                entity_spans = [[_e_s[0] for _e_s in e_s] for e_s in _entity_spans]
                entities = [[_e_s[1] for _e_s in e_s] for e_s in _entity_spans]
                _encodings = [self.tokenizer(b_text, entity_spans=ent_spans, entities=ents, truncation=True, padding=True,
                                             return_tensors='pt') for b_text, ents, ent_spans in zip(b_texts, entities, entity_spans)]
            else:
                b_text = b_texts[batch_id]
                entity_spans = [_get_ent_spans(self.nlp, text) for text in b_text]
                _entity_spans = [_get_ent_spans(self.nlp, text, return_ents=True) for text in b_text]
                entity_spans = [e_s[0] for e_s in _entity_spans]
                entities = [e_s[1] for e_s in _entity_spans]
                _encodings = self.tokenizer(b_text, entity_spans=entity_spans, entities=entities, truncation=True, padding=True, return_tensors='pt')
                # _encodings = self.tokenizer(b_text, entity_spans=entity_spans, truncation=True, padding=True, return_tensors='pt')
        else:
            # _encodings = self.tokenizer(texts[0], truncation=True, padding=True, return_tensors='pt')
            if batch_id is None:
                _encodings = [self.tokenizer(b_text, truncation=True, padding=True, return_tensors='pt') for b_text in b_texts]
            else:
                b_text = b_texts[batch_id]
                _encodings = self.tokenizer(b_text, truncation=True, padding=True, return_tensors='pt')

        i_loss = 0.
        if self.intermediate is not None:
            if batch_id is None:
                _outs = [self.model(**_e.to(dev)) for _e in _encodings]
                if self.model_name == "deberta":
                    p_outs = [self._model.pooler(_out.last_hidden_state.to(dev1)) for _out in _outs]
                else:
                    p_outs = [_out.pooler_output.to(dev1) for _out in _outs]
                ent_outs = [_out.entity_last_hidden_state.mean(dim=1).to(dev1) for _out in _outs]
                outs = torch.vstack([self.intermediate(torch.hstack((p_out, ent_out))) for p_out, ent_out in zip(p_outs, ent_outs)])
            else:
                _outs = self.model(**_encodings.to(dev))
                if self.model_name == "deberta":
                    p_outs = self._model.pooler(_outs.last_hidden_state.to(dev1))
                else:
                    p_outs = _outs.pooler_output.to(dev1)
                ent_outs = _outs.entity_last_hidden_state.mean(dim=1).to(dev1)
                outs = self.intermediate(torch.hstack((p_outs, ent_outs)))
        else:
            if batch_id is None:
                if return_logits:
                    self._model.to(dev)
                    outs = torch.vstack([self._model(**_e.to(dev)).logits for _e in _encodings])
                    if self.model_name == "deberta":
                        self._model.pooler.to(dev1)
                    self._model.dropout.to(dev1)
                    self._model.classifier.to(dev1)
                    return outs
                _outs = [self.model(**_e.to(dev)) for _e in _encodings]
                if self.model_name == "deberta":
                    outs = torch.vstack([self._model.pooler(_out.last_hidden_state.to(dev1)) for _out in _outs])
                elif not self.use_ents:
                    outs = torch.vstack([_out.pooler_output.to(dev1) for _out in _outs])
                else:
                    p_outs = [_out.pooler_output.to(dev1) for _out in _outs]
                    ent_outs = [_out.entity_last_hidden_state.mean(dim=1).to(dev1) for _out in _outs]
                    outs = torch.vstack([torch.hstack((p_out, ent_out)) for p_out, ent_out in zip(p_outs, ent_outs)])
            else:
                _outs = self.model(**_encodings.to(dev))
                if self.model_name == "deberta":
                    outs = self._model.pooler(_outs.last_hidden_state.to(dev1))
                elif not self.use_ents:
                    outs = _outs.pooler_output.to(dev1)
                else:
                    p_outs = _outs.pooler_output.to(dev1)
                    ent_outs = _outs.entity_last_hidden_state.mean(dim=1).to(dev1)
                    outs = torch.hstack((p_outs, ent_outs))
                if labels is not None and self.i_loss:
                    pooled_output = self._model.dropout(outs)
                    logits = self._model.classifier(pooled_output)
                    if changes is None:
                        loss_fct = nn.CrossEntropyLoss()
                        i_loss = loss_fct(logits.view(-1, len(self.classes)),
                                          torch.LongTensor(b_labels[batch_id]).to(dev1).view(-1))

                    else:
                        loss_fct = nn.CrossEntropyLoss(reduction='none')
                        i_loss = loss_fct(logits.view(-1, len(self.classes)),
                                          torch.LongTensor(b_labels[batch_id]).to(dev1).view(-1)) @ \
                                 torch.tensor(b_changes[batch_id], dtype=torch.float32).to(dev1)

        return (outs, i_loss) if batch_id is not None else outs

    def forward2(self, outs, labels=None, print_test=False):
        """
        Second part
        :param outs:
        :param labels:
        :param i_loss:
        :param print_test:
        :return:
        """
        if labels is not None:
            out2 = self.l_transformer(inputs_embeds=outs.unsqueeze(0).to(dev1),
                                      labels=torch.LongTensor(labels).to(dev1))
            # out2.loss = out2.loss
        else:
            out2 = self.l_transformer(inputs_embeds=outs.unsqueeze(0).to(dev1))
        return out2

    def forward(self, texts, labels=None, attention_mask=None, intermediate_loss=False, print_test=False, return_iloss=False):
        """

        :param texts:
        :param labels: if None then used for inference
        :param attention_mask:
        :return:
        """
        if self.s_transformer:
            outs = encode(self.model, texts[0], convert_to_tensor=True, normalize_embeddings=True)
            if labels is not None:
                out2 = self.l_transformer(inputs_embeds=outs.unsqueeze(0).to(dev1),
                                          labels=torch.LongTensor(labels).to(dev1))
            else:
                out2 = self.l_transformer(inputs_embeds=outs.unsqueeze(0).to(dev1))
            return out2

        _batch_size = 4
        if self.use_ents:
            _entity_spans = [_get_ent_spans(self.nlp, text, return_ents=True) for text in texts[0]]
            entity_spans = [e_s[0] for e_s in _entity_spans]
            entities = [e_s[1] for e_s in _entity_spans]
            _encodings = self.tokenizer(texts[0], entity_spans=entity_spans, truncation=True, entities=entities, padding=True, return_tensors='pt')
        else:
            _encodings = self.tokenizer(texts[0], truncation=True, padding=True, return_tensors='pt')

        i_loss = 0.
        if self.intermediate is not None:
            _outs = self.model(**_encodings.to(dev))
            p_outs = _outs.pooler_output.to(dev1)
            ent_outs = _outs.entity_last_hidden_state.mean(dim=1).to(dev1)
            outs = self.intermediate(torch.hstack((p_outs, ent_outs)))
        else:
            _outs = self.model(**_encodings.to(dev))
            if self.model_name == "deberta":
                outs = self._model.pooler(_outs.last_hidden_state.to(dev1))
            elif not self.use_ents:
                outs = _outs.pooler_output.to(dev1)
            else:
                p_outs = _outs.pooler_output.to(dev1)
                ent_outs = _outs.entity_last_hidden_state.mean(dim=1).to(dev1)
                outs = torch.hstack((p_outs, ent_outs))

            if return_iloss:
                pooled_output = self._model.dropout(outs)
                logits = self._model.classifier(pooled_output)
                loss_fct = nn.CrossEntropyLoss()
                i_loss = loss_fct(logits.view(-1, len(self.classes)), torch.LongTensor(labels).to(dev1).view(-1))
                return i_loss

        if intermediate_loss:
            _vectors = self._vectors(torch.LongTensor(labels)).squeeze()
            _loss = nn.MSELoss(reduction='sum')
            if self.use_ents:
                i_loss = _loss(p_outs.squeeze(), _vectors.to(dev1))
            else:
                i_loss = _loss(outs.squeeze(), _vectors.to(dev1))

        if labels is not None:
            out2 = self.l_transformer(inputs_embeds=outs.unsqueeze(0).to(dev1),
                                      labels=torch.LongTensor(labels).to(dev1))
            out2.loss = out2.loss + i_loss

        else:
            out2 = self.l_transformer(inputs_embeds=outs.unsqueeze(0).to(dev1))

        return out2

    def save_model(self, path):
        torch.save(self, path)

    def eval_loss(self, eval_data, batch_size=1):
        with torch.no_grad():
            losses = []
            _losses = []
            self.model.to(dev)
            self.eval()
            _eval = list(eval_data.values())  # list of lists
            for i in tqdm.tqdm(range(0, len(_eval), batch_size), desc="Eval"):
                eval_batch = _eval[i: i + batch_size]
                _batch_size = len(eval_batch)

                texts = []
                label_mask = np.zeros((_batch_size, max([len(t) for t in eval_batch])), dtype=bool)
                labels = np.zeros((_batch_size, max([len(t) for t in eval_batch])), dtype=int)

                for j, _t in enumerate(eval_batch):
                    _texts, _labels = _make_texts({1: _t}, unused=[], out_path=f"{'_all' if self.cats is None else ''}_1", ners=self.ner,
                                                  use_bins=self.use_bins, conversion_dict=self.conversion_dict, nba_data="nba_data" in sys.argv)  # only current text
                    _labels = [self.classes.index(l) for l in _labels]
                    label_mask[j, :len(_t)] = 1
                    labels[j, :len(_t)] = _labels
                    texts.append(_texts)

                loss = self.forward(texts, labels).loss
                if not self.use_ents and self.s_transformer is None:
                    _loss = self.forward(texts, labels, return_iloss=True)
                    _losses.append(_loss.item())
                else:
                    _losses.append(0.)
                losses.append(loss.item())
            return np.mean(losses), np.mean(_losses)

    def train_transformer(self, train_data, batch_size=1, epochs=30, eval_data=None, accu_grad=1, layers=None, wd=0.,
                          lr1=5e-6, lr2=1e-5, full_grad=False):
        """
        Train the transformer
        :param train_data:
        :param batch_size:
        :param epochs:
        :param eval_data:
        :param accu_grad:
        :param layers:
        :param wd:
        :param lr1:
        :param lr2:
        :param full_grad:
        :return:
        """
        only_changes = "changes" in sys.argv
        if TESTING:
            epochs = 1
        print(f"Training. Batch size: {batch_size}, epochs: {epochs}, lr1: {lr1}, lr2: {lr2}")
        print(f"Accu Grad: {accu_grad}")
        print(f"FullGrad: {full_grad}")

        import torch.optim as optim
        print(f"******************* leaving layers {layers} unfrozen *******************")

        _i = 0
        self.model.to(dev)
        self.train()
        if self.with_adapter:
            self.model.train_adapter("transitions")

        to_train = [{"params": [p for n, p in self.l_transformer.named_parameters() if n.find("classifier") == -1]
                    if self.vectors is not None else self.l_transformer.parameters()},
                    {"params": self._model.parameters(), "lr": lr1}]
                    # {"params": self.model.parameters(), "lr": 1e-3}]
        if self.intermediate is not None:
            to_train = to_train + [{"params": self.intermediate.parameters()}]

        # to_train = list(self.parameters())
        if self.with_adapter:
            to_train = to_train + list(self.model.adapter("transitions").parameters())  # !!!!!

        optimizer = optim.AdamW(to_train, lr=lr2, weight_decay=wd)
        max_steps = math.ceil(epochs * 200.)
        def lr_lambda(current_step: int):
            return max(0.0, float(max_steps - current_step) / float(max(1, max_steps)))
        optimizer.zero_grad()

        _train = list(train_data.values())  # list of lists

        random.seed()
        if eval_data is not None:
            eval_loss = self.eval_loss(eval_data)
            print("Eval loss:", eval_loss)
            evaluation(vectors=self.vectors, val_data=eval_data, cat_dict=self.cat_dict, loc_transformer=self,
                       conversion_dict=self.conversion_dict)
            self.train()

        for e in range(epochs):
            losses = []
            print("\n" + str(e))
            random.shuffle(_train)
            for i in tqdm.tqdm(range(0, len(_train), batch_size), desc="Train"):
                train_batch = _train[i: i + batch_size]
                _batch_size = len(train_batch)

                texts = []
                label_mask = np.zeros((_batch_size, max([len(t) for t in train_batch])), dtype=bool)
                labels = np.zeros((_batch_size, max([len(t) for t in train_batch])), dtype=int)

                for j, _t in enumerate(train_batch):
                    _texts, _labels = _make_texts({1: _t}, unused=[], out_path=f"{'_all' if self.cats is None else ''}_1", ners=self.ner,
                                                  use_bins=self.use_bins, conversion_dict=self.conversion_dict, nba_data="nba_data" in sys.argv)  # only current text
                    _labels = [self.classes.index(l) for l in _labels]
                    label_mask[j, :len(_t)] = 1
                    labels[j, :len(_t)] = _labels
                    texts.append(_texts)

                _inner_batch_size = 4
                num_batches = len(labels[0]) // _inner_batch_size + int(len(labels[0]) % _inner_batch_size != 0)
                if only_changes:
                    changes = (labels[0] != np.insert(labels[0][:-1], 0, -1)) + 1e-5
                seed = torch.randint(1000, (1,))
                torch.manual_seed(seed.item())
                with torch.no_grad():
                    outs = self.forward1(texts, labels, batch_size=_inner_batch_size, batch_id=None)
                outs.requires_grad = True
                loss = self.forward2(outs, labels).loss
                losses.append(loss.item())
                loss.backward()
                grads = torch.clone(outs.grad)
                #
                optimizer.step()
                optimizer.zero_grad()

                torch.manual_seed(seed.item())
                if full_grad:
                    for b_i in range(num_batches):
                        outs, i_loss = self.forward1(texts, labels, batch_size=_inner_batch_size, batch_id=b_i,
                                                     changes=changes if only_changes else None)
                        if self.i_loss:
                            outs.backward(grads[b_i * _inner_batch_size: (b_i + 1) * _inner_batch_size], retain_graph=True)
                            (i_loss / num_batches).backward()
                        else:
                            outs.backward(grads[b_i * _inner_batch_size: (b_i + 1) * _inner_batch_size])

                if accu_grad == 1 or (i + 1) % accu_grad == 0 or i + accu_grad >= len(
                        _train) or _batch_size < batch_size:
                    optimizer.step()
                    optimizer.zero_grad()

            print("Epoch loss:", np.mean(losses))
            if eval_data is not None:
                eval_loss = self.eval_loss(eval_data)
                print("Eval loss:", eval_loss)

                if e % 5 == 0:
                    evaluation(vectors=self.vectors, val_data=eval_data, cat_dict=self.cat_dict, loc_transformer=self,
                               conversion_dict=self.conversion_dict)
                    self.train()

        return self

    def train_classifier(self, train_data, val_data, save_data=False, out_path=None, wd=0.0, ner=False, use_bins=False):
        """
        Train the classifier
        :param data_path:
        :param save_data:
        :param out_path:
        :param first_train_size:
        :param val_size:
        :param test_size:
        :param conversion_dict:
        :param wd:
        :param ner:
        :param use_bins:
        :return:
        """

        metric = load_metric("accuracy")
        def compute_metrics_v(eval_pred):
            vs, labels = eval_pred
            predictions = [find_closest(None, self.vectors[1], v) for v in vs]
            eval_labels = [find_closest(None, self.vectors[1], v) for v in labels]
            return metric.compute(predictions=predictions, references=eval_labels)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        train_texts, train_labels = _make_texts(train_data, [], out_path, conversion_dict=self.conversion_dict,
                                                vectors=self.vectors, ners=ner, use_bins=use_bins, nba_data="nba_data" in sys.argv)
        val_texts, val_labels = _make_texts(val_data, [], out_path, conversion_dict=self.conversion_dict,
                                            vectors=self.vectors, ners=ner, use_bins=use_bins, nba_data="nba_data" in sys.argv)

        train_labels = [self.classes.index(l) for l in train_labels]
        val_labels = [self.classes.index(l) for l in val_labels]

        print("made data")
        print(f"Use entities: {self.use_ents}")
        print(f"*************** {out_path.split('/')[-1]} **************")

        tokenizer = self.tokenizer

        if self.use_ents:
            train_entity_spans = [_get_ent_spans(self.nlp, text) for text in train_texts]
            val_entity_spans = [_get_ent_spans(self.nlp, text) for text in val_texts]
            train_encodings = tokenizer(train_texts[:], entity_spans=train_entity_spans, truncation=True, padding=True)
            val_encodings = tokenizer(val_texts[:], entity_spans=val_entity_spans, truncation=True, padding=True)
        else:
            train_encodings = tokenizer(train_texts[:], truncation=True, padding=True)
            val_encodings = tokenizer(val_texts[:], truncation=True, padding=True)
        print("made encodings")

        train_dataset = Dataset(train_encodings, train_labels[:])
        val_dataset = Dataset(val_encodings, val_labels[:])

        if save_data:
            self.train_dataset, self.val_dataset = train_dataset, val_dataset

        lr = 5e-5
        print("Learning rate: ")
        print(lr)

        training_args = TrainingArguments(
            output_dir='/cs/labs/oabend/eitan.wagner/checkpoints/results',  # output directory
            num_train_epochs=3,  # total number of training epochs
            learning_rate=lr,
            per_device_train_batch_size=4,  # batch size per device during training
            per_device_eval_batch_size=4,  # batch size for evaluation
            gradient_accumulation_steps=1,
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=wd,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # load_best_model_at_end=True,

        )

        model = self._model
        model.to(dev)
        print("Training")

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics_v if self.vectors is not None else compute_metrics,
        )

        trainer.train()
        self._model = model
        if self.model_name == "deberta":
            self.model = self._model.deberta
            self._model.pooler.to(dev1)
        elif self.model_name == "luke":
            self.model = self._model.luke
        self.model.to(dev)
        self._model.dropout.to(dev1)
        self._model.classifier.to(dev1)

        if self.with_adapter:
            from transformers import DebertaAdapterModel, AdapterTrainer
            self.model.add_adapter("transitions")
        print("Trained classifier")

    def _decode(self, text):
        """

        :param text: sequence of segments
        :return:
        """
        self.eval()
        # TODO
        with torch.no_grad():
            outs = torch.stack([self.forward(texts=t).squeeze() for t in text])
            preds = self.l_transformer.generate(inputs_embeds=outs.unsqueeze(0), max_new_tokens=len(text), min_new_tokens=len(text))
        return preds.squeeze().tolist()[1:]

    def decode1(self, text):
        """

        :param text: sequence of segments
        :return:
        """
        self.eval()
        with torch.no_grad():
            outs = self.forward1(texts=[text], return_logits=True).detach().cpu().numpy()
            preds = np.argmax(outs, axis=-1)
        return preds.squeeze().tolist()

    def decode2(self, text):
        """

        :param text: sequence of segments
        :return:
        """
        self.eval()
        # TODO
        with torch.no_grad():
            # outs = torch.stack([self.forward(texts=t).logits.squeeze() for t in text]).detach().cpu().numpy()
            outs = self.forward(texts=[text]).logits.detach().cpu().numpy()
            preds = np.argmax(outs, axis=-1)
        return preds.squeeze().tolist()

    def convert_to_labels(self, preds):
        return [self.classes[p] for p in preds]


def get_nba_data(data_path="/cs/labs/oabend/eitan.wagner/location tracking/data/", val_size=0.1, test_size=0.1):
    """
    Get the NBA data
    :param data_path:
    :param val_size:
    :param test_size:
    :return:
    """
    data_path = "/cs/labs/oabend/eitan.wagner/location tracking/data/"
    with open(data_path + "nba_data.json", 'r') as infile:
        data = json.load(infile)
    _l_data = list(data.items())
    random.seed(SEED)
    random.shuffle(_l_data)
    random.seed()
    data = dict(_l_data)

    train_data = {t: text for t, text in list(data.items())[:-int(val_size * len(data))-int(test_size * len(data))]}
    val_data = {t: text for t, text in list(data.items())[-int(val_size * len(data))-int(test_size * len(data)):-int(test_size * len(data))]}
    test_data = {t: text for t, text in list(data.items())[-int(test_size * len(data)):]}
    print(f"Training on {len(train_data)} documents")
    return train_data, val_data, test_data

def get_data(data_path, first_train_size=0.8, val_size=0.1, test_size=0.1, use_test=False):
    """
    Get the data
    :param data_path:
    :param first_train_size:
    :param val_size:
    :param test_size:
    :return:
    """
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

    if not use_test:
        train_data = {t: text for t, text in list(data.items())[:int(first_train_size * len(data)) if not TESTING else 5]}
        val_data = {t: text for t, text in list(data.items())[-int(test_size * len(data)) - int(val_size * len(data)):
                                                              -int(test_size * len(data)) if not TESTING
                                                              else -int(test_size * len(data)) - int(val_size * len(data)) + 1]}
    else:
        train_data = {t: text for t, text in list(data.items())[:-int(val_size*len(data))]}
        # this is the val_data
        val_data = {t: text for t, text in list(data.items())[-int(val_size*len(data)):]}
    print(f"Training on {len(train_data)} documents")
    return train_data, val_data, None


def main(args):
    print(sys.argv)
    nba_data = args.bio_data
    pretrain = args.train_classifier
    evaluate = True
    evaluate1 = True  # evaluate categories
    use_vectors = args.use_vectors
    st_vectors = args.st_vector
    use_cats = args.use_cats
    conv_dict = True
    use_ents = args.use_ents
    full_grad = args.use_full_grad
    use_test = args.use_test
    load_model = args.load_model
    i_loss = args.use_i_loss

    model_name = args.model
    lr1 = args.lr1
    lr2 = args.lr2
    epochs = args.epochs
    suffix = f"{model_name}_{'conv_' if conv_dict else ''}{'fg_' if full_grad else ''}{'ents_' if use_ents else ''}" \
             f"{'v_' if use_vectors else ''}{'st_' if st_vectors else ''}lr1_{lr1}_lr2_{lr2}_e_{epochs}{'_nba' if nba_data else ''}"

    global TESTING
    TESTING = "testing" in sys.argv
    layers = [s[5:] for s in sys.argv if s[:5] == "layer"]

    base_path = args.base_path
    data_path = base_path + "data/"

    if nba_data:
        train_data, val_data, test_data = get_nba_data(data_path=data_path)
    else:
        train_data, val_data, test_data = get_data(data_path=data_path, use_test=use_test)
    cats = None
    vectors = None
    c_dict = None
    if not nba_data:
        # make conversion dict
        if conv_dict:
            from loc_clusters import make_loc_conversion
            c_dict = make_loc_conversion(data_path=data_path)

        with open(data_path + 'loc_category_dict.json', 'r') as infile:
            cat_dict = json.load(infile)
            cat_dict["START"] = "START"
            cat_dict["END"] = "END"
            if conv_dict:
                cat_dict = {k: c_dict[v] for k, v in cat_dict.items()}
            classes = list(cat_dict.keys())
            classes.sort()
            print("Classes: ", classes)
            if use_cats:
                cats = list(set(cat_dict.values()))
                cats.sort()
                print("Cats: ", cats)
    else:
        # get the classes
        classes = _make_texts(data=train_data, nba_data=True)[1]
        classes = classes + _make_texts(data=val_data, nba_data=True)[1]
        classes = classes + _make_texts(data=test_data, nba_data=True)[1]
        classes = list(set(classes))
        cat_dict = {c: c for c in classes}
        use_cats = False

    st_model = None
    if st_vectors:
        st_model, vectors = make_sentence_vectors(classes=cats if use_cats else classes, use_categories=use_cats, nba=nba_data)
    elif use_vectors:
        from loc_clusters import make_vectors
        vectors = make_vectors(data_path=data_path, normalize=True)

    if load_model:
        loc_transformer = torch.load(base_path + f"models/loc_transformer/lt_{suffix}.pkl", map_location=dev)
        if model_name == "deberta":
            loc_transformer.model = loc_transformer._model.deberta
            loc_transformer._model.pooler.to(dev1)
        elif model_name == "luke":
            if not st_vectors:
                loc_transformer.model = loc_transformer._model.luke
                loc_transformer._model.dropout.to(dev1)
                loc_transformer._model.classifier.to(dev1)
            loc_transformer.model.to(dev)
            loc_transformer.l_transformer.to(dev1)

    else:
        # train a model
        loc_transformer = LocTransformer(vectors=vectors, initialize_model=True, classes=classes, cats=cats,
                                         cat_dict=cat_dict, use_ents=use_ents, model_name=model_name,
                                         conversion_dict=c_dict, i_loss=i_loss, s_transformer=st_model)
        if pretrain:
            # use classification pretraining
            loc_transformer.train_classifier(train_data, val_data, out_path=f"{'' if use_cats else '_all'}_1", use_bins=False)
            # save only base model
            torch.save(loc_transformer._model, base_path + f"models/loc_transformer/_model{'_c' if use_cats else ''}{'_nba' if nba_data else ''}.pkl")
        elif not st_vectors:
            print("Loading _model")
            loc_transformer._model = torch.load(base_path + f"models/loc_transformer/_model{'_c' if use_cats else ''}{'_nba' if nba_data else ''}.pkl", map_location=dev)
            if model_name == "deberta":
                loc_transformer.model = loc_transformer._model.deberta
                loc_transformer._model.pooler.to(dev1)
            elif model_name == "luke":
                loc_transformer.model = loc_transformer._model.luke
                loc_transformer.model.to(dev)
                # self._model.dropout = nn.Dropout(0.)
                loc_transformer._model.dropout.to(dev1)
                loc_transformer._model.classifier.to(dev1)

        loc_transformer.train_transformer(train_data=train_data, eval_data=val_data, layers=layers, full_grad=full_grad,
                                          lr1=lr1, lr2=lr2, epochs=epochs)
        loc_transformer.save_model(path=base_path + f"models/loc_transformer/lt_{suffix}.pkl")

    # evaluation
    print("Evaluation on the val set")
    evaluation(vectors=vectors, evaluate1=evaluate1, val_data=val_data, cat_dict=cat_dict,
               loc_transformer=loc_transformer, conversion_dict=c_dict, use_test=False)
    if use_test and test_data is None:
        print("Evaluation on the test set")
        evaluation(vectors=vectors, evaluate1=evaluate1, val_data=val_data, cat_dict=cat_dict,
                   loc_transformer=loc_transformer, conversion_dict=c_dict, use_test=use_test)
    elif test_data is not None:
        print("Evaluation on the test set")
        evaluation(vectors=vectors, evaluate1=evaluate1, val_data=test_data, cat_dict=cat_dict,
                   loc_transformer=loc_transformer, conversion_dict=c_dict, use_test=False)


if __name__ == "__main__":

    main(args)
