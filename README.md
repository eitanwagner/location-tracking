
# Location-Tracking Repository

Code for the paper "Event-Location Tracking in Narratives: A Case Study on Holocaust Testimonies"
https: https://aclanthology.org/2023.emnlp-main.544/

---

## Install
```bash
git clone https://github.com/eitanwagner/location-tracking.git
cd location-tracking
pip install -e requirements.txt

```


---

General arguments:
- **--base_path**, type=str - path to project"
- **--cache_dir**, type=str, default=None

- **--use_len**, action="store_true" - whether do use length regularization
- **--train_classifier**, action="store_true"
- **--train_classifier2**, action="store_true"
- **--use_full_grad**, action="store_true" - whether to use gradients for first transformer
- **--use_test**, action="store_true" - whether to evaluate on the test set
- **--use_bins**, action="store_true" - whether to add the bin number to the segment
- **--use_cats**, action="store_true" - whether to use categories as labels
- **--bio_data**, action="store_true" - whether to use the biography data
- **--use_vectors**, action="store_true" - whether to vectors
- **--use_ents**, action="store_true" - whether to use entities in the pipelie
- **--use_i_loss**, action="store_true" - whether to use intermediate loss
- **--load_model**, action="store_true" - whether to use a pre-trained model
- **--st_vectors**, action="store_true" - whether to use sentence-transformer vectors
- **--only_greedy**, action="store_true"
- **--reverse**, action="store_true"

---

## Hierarchical Transformers

Arguments for the Hierarchical Transformers method:
- **--model**, type=str, default='luke' - name of model to use as text encoder.
- **--lr1**, type=float, default=5e-6 - learning rate for first transformer (text encoder).
- **--lr2**, type=float, default=1e-5 - learning rate for second transformer.


For example, 
```bash
python loc_transformer.py --base_path <base_path> --model luke --lr1 5e-6 --lr2 1e-5 --use_len True
```


---
 
## CRF
 
Arguments for the CRF method:
- **--model2**, type=str, default='' - name of model to use for transition weights.
- **--lr**, type=float, default=5e-6 - learning rate.
- **--wd**, type=float, default=1e-2 - help="weight decay"

For example, 
```bash
python location_tracking.py --base_path <base_path> --model2 luke --lr 5e-6 --wd 1e-2
```

# Citation

```bibtex
@inproceedings{wagner-etal-2023-event,
    title = "Event-Location Tracking in Narratives: A Case Study on Holocaust Testimonies",
    author = "Wagner, Eitan  and
      Keydar, Renana  and
      Abend, Omri",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.544",
    doi = "10.18653/v1/2023.emnlp-main.544",
    pages = "8789--8805",
    abstract = "This work focuses on the spatial dimension of narrative understanding and presents the task of event-location tracking in narrative texts. The task intends to extract the sequence of locations where the narrative is set through its progression. We present several architectures for the task that seeks to model the global structure of the sequence, with varying levels of context awareness. We compare these methods to several baselines, including the use of strong methods applied over narrow contexts. We also develop methods for the generation of location embeddings and show that learning to predict a sequence of continuous embeddings, rather than a string of locations, is advantageous in terms of performance. We focus on the test case of Holocaust survivor testimonies. We argue for the moral and historical importance of studying this dataset in computational means and that it provides a unique case of a large set of narratives with a relatively restricted set of location trajectories. Our results show that models that are aware of the larger context of the narrative can generate more accurate location chains. We further corroborate the effectiveness of our methods by showing similar trends from experiments on an additional domain.",
}

```
