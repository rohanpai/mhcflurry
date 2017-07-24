"""
Train Class1 single allele models.

"""
import sys
import argparse
import json

import pandas

from .class1_affinity_predictor import Class1AffinityPredictor
from ..common import configure_logging
from sklearn.model_selection import train_test_split
from mhcflurry.scoring import make_scores
import operator


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "--data",
    metavar="FILE.csv",
    required=True,
    help=(
        "Training data CSV. Expected columns: "
        "allele, peptide, measurement_value"))
parser.add_argument(
    "--out-models-dir",
    metavar="DIR",
    required=True,
    help="Directory to write models and manifest")
parser.add_argument(
    "--hyperparameters",
    metavar="FILE.json",
    required=True,
    help="JSON of hyperparameters")
parser.add_argument(
    "--allele",
    default=None,
    nargs="+",
    help="Alleles to train models for. If not specified, all alleles with "
    "enough measurements will be used.")
parser.add_argument(
    "--min-measurements-per-allele",
    type=int,
    metavar="N",
    default=50,
    help="Train models for alleles with >=N measurements.")
parser.add_argument(
    "--verbosity",
    type=int,
    help="Keras verbosity. Default: %(default)s",
    default=1)
parser.add_argument(
    "--use-kubeface",
    default=False,
    action="store_true",
    help="Use Kubeface: %(default)s")
parser.add_argument(
    "--ensemble-size",
    default=8,
    type=int,
    help="Ensemble Size: %(default)s")


try:
    import kubeface
    kubeface.Client.add_args(parser)
except Exception as e:
    pass

def allele_fn(arguments):
    from sklearn.model_selection import train_test_split
    from mhcflurry.scoring import make_scores
    import numpy as np        
    allele, allele_data, args, hyperparameters_lst = arguments 
    train_data, model_train_data = train_test_split(allele_data, 
        test_size=0.2, 
        random_state=42)            
    
    ## this is the training data
    alleles = train_data.allele.values
    peptides = train_data.peptide.values
    affinities = train_data.measurement_value.values

    # this is model comparison data
    model_alleles = model_train_data.allele.values
    model_peptides = model_train_data.peptide.values
    model_affinities = model_train_data.measurement_value.values            
    
    best_score = 0
    best_predictor = None

    for (h, hyperparameters) in enumerate(hyperparameters_lst):
        if "n_models" in hyperparameters:
            n_models = hyperparameters.pop("n_models")
            
        tmp_model = Class1AffinityPredictor()
        
        # train model
        tmp_model.fit_allele_specific_predictors(
            n_models=args.ensemble_size,
            architecture_hyperparameters=hyperparameters,
            peptides=peptides,
            allele=allele,
            affinities=affinities)

        # compare tmp_model with best score
        tmp_predictions = tmp_model.predict(alleles=model_alleles.tolist(), peptides=model_peptides.tolist())
        tmp_scores = make_scores(np.asarray(tmp_predictions), np.asarray(model_affinities))
        tmp_score = tmp_scores['auc'] + tmp_scores['f1'] + tmp_scores['tau']
        
        if best_predictor == None:
            best_predictor = tmp_model
        if tmp_score > best_score:
            best_predictor = tmp_model
            best_score = tmp_score
    return best_predictor


def run_model_selection(argv=sys.argv[1:]):

    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbosity > 1)

    hyperparameters_lst = json.load(open(args.hyperparameters))
    assert isinstance(hyperparameters_lst, list)
    print("Loaded hyperparameters list: %s" % str(hyperparameters_lst))


    df = pandas.read_csv(args.data)
    
    print("Loaded training data: %s" % (str(df.shape)))

    df = df.ix[
        (df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)
    ]
    print("Subselected to 8-15mers: %s" % (str(df.shape)))

    allele_counts = df.allele.value_counts()

    if args.allele:
        alleles = args.allelle
        df = df.ix[df.allele.isin(alleles)]
    else:
        alleles = list(allele_counts.ix[
            allele_counts > args.min_measurements_per_allele
        ].index)

    print("Selected %d alleles: %s" % (len(alleles), ' '.join(alleles)))
    print("Training data: %s" % (str(df.shape)))
    
    final_predictor = Class1AffinityPredictor()
    best_predictors = []



    from multiprocessing import Pool
    from multiprocessing.dummy import Pool as ThreadPool

    pool = None

    map_fn = map
    if args.use_kubeface:
        client = kubeface.Client.from_args(args)            
        map_fn = client.map
    else:
        pool = Pool(4)
        map_fn = pool.map
            

    inputs = [ (allele, df.ix[df.allele == allele].dropna().sample(frac=1.0), args, hyperparameters_lst) for allele in alleles[:4]]
    results = map_fn(allele_fn, inputs)
    best_predictors = [r for r in results]
    print(best_predictors)
    
    final_predictor.merge(best_predictors)
    return final_predictor


def compare_predictors(predictor_1, predictor_2):

    
    df = pandas.read_csv("abelin_peptides.mhcflurry.csv.bz2")

    def ppv(is_hit, predictions):
        df = pandas.DataFrame({"prediction": predictions, "is_hit": is_hit})
        return df.sort_values("prediction", ascending=True)[:int(is_hit.sum())].is_hit.mean()
    print(ppv(df.hit.values, df.mhcflurry.values))
    #predictor_1.predict(allele=df.allele.tolist(), peptides=df.peptide.tolist())
    
def run():
    best_predictor = Class1AffinityPredictor.load()
    model_selected_predictor = run_model_selection()
    
    print(model_selected_predictor)
    compare_predictors(model_selected_predictor, best_predictor)

if __name__ == '__main__':
    run()
