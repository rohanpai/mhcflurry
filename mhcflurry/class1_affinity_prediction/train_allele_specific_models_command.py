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

import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import random
import string
import random


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

parser.add_argument(
    "--test",
    default=False,
    action="store_true")

parser.add_argument(
    "--parallelize-hyperparameters",
    default=False,
    action="store_true")


try:
    import kubeface
    kubeface.Client.add_args(parser)
except Exception as e:
    pass


# these are the alleles that the mass spec data is trained on that we can use to train our models
alleles_of_interest = ['HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*02:03', 'HLA-A*03:01', 'HLA-A*11:01', 'HLA-A*24:02', 'HLA-A*29:02', 'HLA-A*31:01', 'HLA-A*68:02', 'HLA-B*07:02', 'HLA-B*15:01', 'HLA-B*35:01', 'HLA-B*44:02', 'HLA-B*44:03', 'HLA-B*51:01', 'HLA-B*54:01', 'HLA-B*57:01']
#alleles_of_interest = ['HLA-A*01:01']
alleles_of_interest = ['HLA-A*02:01', 'HLA-A*02:03', 'HLA-A*03:01', 'HLA-A*11:01']

def train_model(arguments):
    print("doing something")
    hyperparameters, peptides, allele, affinities = arguments
    if "n_models" in hyperparameters:
        n_models = hyperparameters.pop("n_models")
        
    tmp_model = Class1AffinityPredictor()
    tmp_model.fit_allele_specific_predictors(
        n_models=1,
        architecture_hyperparameters=hyperparameters,
        peptides=peptides,
        allele=allele,
        affinities=affinities,
        verbose=1)
    return tmp_model



def allele_fn(arguments):
    from sklearn.model_selection import train_test_split
    from mhcflurry.scoring import make_scores
    import numpy as np   
    allele, allele_data, args, hyperparameters_lst = arguments
    best_predictors = []
    map_fn = map
    pool = None
    if args.parallelize_hyperparameters:
        pool = Pool(20)
        map_fn = pool.map
    for i in range(args.ensemble_size):
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
        
        inputs = [ (hyperparameters, peptides, allele, affinities) for hyperparameters in hyperparameters_lst]
        tmp_models = map_fn(train_model, inputs)
        
        best_score = 0
        best_predictor = None
        
        for tmp_model in tmp_models:
            # compare tmp_model with best score
            tmp_predictions = tmp_model.predict(alleles=model_alleles.tolist(), peptides=model_peptides.tolist())
            tmp_scores = make_scores(np.asarray(tmp_predictions), np.asarray(model_affinities))
            tmp_score = tmp_scores['auc'] + tmp_scores['f1'] + tmp_scores['tau']
            
            if best_predictor == None:
                best_predictor = tmp_model
            if tmp_score > best_score:
                best_predictor = tmp_model
                best_score = tmp_score
        best_predictors.append(best_predictor)
    
    inputs = [(predictor.manifest_df.model[0].hyperparameters, allele_data.peptide.values, allele, allele_data.measurement_value.values) for predictor in best_predictors]
    final_best_models = map_fn(train_model, inputs)

    best_model = Class1AffinityPredictor()
    best_model.merge(final_best_models)

    pool.terminate()
    return best_model 


def run_model_selection(argv=sys.argv[1:]):
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbosity > 1)

    hyperparameters_lst = json.load(open(args.hyperparameters))
    random.shuffle(hyperparameters_lst)
    hyperparameters_lst = hyperparameters_lst[:10]
    
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

    pool = None

    map_fn = map
    #pool = Pool(20)
    #map_fn = pool.map

    inputs = [(allele, df.ix[df.allele == allele].dropna().sample(frac=1.0), args, hyperparameters_lst) for allele in alleles_of_interest]
    print(len(inputs))

    results = map_fn(allele_fn, inputs)
    best_predictors = [r for r in results]
    print(best_predictors)
    
    final_predictor.merge(best_predictors)
    return final_predictor

def ppv(is_hit, predictions):
    df = pandas.DataFrame({"prediction": predictions, "is_hit": is_hit})
    return df.sort_values("prediction", ascending=True)[:int(is_hit.sum())].is_hit.mean()

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def run():
    best_predictor = Class1AffinityPredictor.load()
    model_selected_predictor = run_model_selection()
    random_str_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    os.mkdir(random_str_name)
    model_selected_predictor.save(random_str_name)
    print("Created Model " + random_str_name) 
    
    #print(model_selected_predictor)
    #just_trained_model = Class1AffinityPredictor.load('./models')
    #compare_predictors(just_trained_model)

if __name__ == '__main__':
    run()
