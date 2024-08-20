
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # for location transformer
    parser.add_argument('--model', type=str, default='luke', help="name of model to use as text encoder")
    parser.add_argument('--lr1', type=float, default=5e-6, help="learning rate for first transformer (text encoder)")
    parser.add_argument('--lr2', type=float, default=1e-5, help="learning rate for second transformer")

    # for CRF
    parser.add_argument('--model2', type=str, default='', help="name of model to use for transition weights")
    parser.add_argument('--lr', type=float, default=5e-6, help="learning rate")
    parser.add_argument('--wd', type=float, default=1e-2, help="weight decay")
    parser.add_argument('--normalize', action="store_true", help="whether to normalize the CRF matrix as probabilities")

    # for all
    parser.add_argument('--train_classifier', action="store_true")
    parser.add_argument('--train_classifier2', action="store_true")
    parser.add_argument('--use_full_grad', action="store_true", help="whether to use gradients for first transformer")
    parser.add_argument('--use_test', action="store_true", help="whether to evaluate on the test set")
    parser.add_argument('--use_bins', action="store_true", help="whether to add the bin number to the segment")
    parser.add_argument('--use_cats', action="store_true", help="whether to use categories as labels")
    parser.add_argument('--bio_data', action="store_true", help="whether to use the biography data")
    parser.add_argument('--use_vectors', action="store_true", help="whether to vectors")
    parser.add_argument('--use_ents', action="store_true", help="whether to use entities in the pipeline")
    parser.add_argument('--use_i_loss', action="store_true", help="whether to use intermediate loss")
    parser.add_argument('--load_model', action="store_true", help="whether to use a pre-trained model")
    parser.add_argument('--st_vectors', action="store_true", help="whether to use sentence-transformer vectors")
    parser.add_argument('--only_greedy', action="store_true")
    parser.add_argument('--reverse', action="store_true")

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--base_path', type=str, help="path to project (name included)")
    parser.add_argument('--cache_dir', type=str, default=None, help="cache directory")
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    return args
