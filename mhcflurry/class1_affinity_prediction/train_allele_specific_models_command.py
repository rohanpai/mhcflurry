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
import numpy as np
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



def run(argv=sys.argv[1:]):
    
    def train(arguments):
        n_models, hyperparameters, allele, peptides, affinities, Class1AffinityPredictor = arguments
        predictor = Class1AffinityPredictor()
        for model_group in range(n_models):
            print(
                "[%2d / %2d hyperparameters] "
                "[%2d / %2d replicates] "
                "[%4d / %4d alleles]: %s" % (
                    h + 1,
                    len(hyperparameters_lst),
                    model_group + 1,
                    n_models,
                    i + 1,
                    len(alleles), allele))

            model = predictor.fit_allele_specific_predictors(
                n_models=n_models,
                architecture_hyperparameters=hyperparameters,
                allele=allele,
                peptides=train_data.peptide.values,
                affinities=train_data.measurement_value.values)
            
            return model

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
    for (i, allele) in enumerate(alleles[:3]):
        allele_data = df.ix[df.allele == allele].dropna().sample(
            frac=1.0)
        
        train_data, model_train_data = train_test_split(allele_data.values, 
                test_size=0.2, 
                random_state=42)

        all_predictors = []
        for (h, hyperparameters) in enumerate(hyperparameters_lst):
            print("hello------------------")
            if "n_models" in hyperparameters:
                n_models = hyperparameters.pop("n_models")
            alleles = [x[0] for x in train_data]
            peptides = [x[1] for x in train_data]
            affinities = [x[2] for x in train_data]
            for i in range(args.ensemble_size):
                tmp_predictor = Class1AffinityPredictor()                
                tmp_predictor.fit_allele_specific_predictors(
                    n_models=1,
                    architecture_hyperparameters=hyperparameters,
                    peptides=peptides,
                    allele=allele,
                    affinities=affinities,
                    models_dir_for_save=args.out_models_dir)
                all_predictors.append(tmp_predictor)
        model_alleles = [x[0] for x in model_train_data]
        model_peptides = [x[1] for x in model_train_data]
        model_affinities = [x[2] for x in model_train_data]

        best_scores_dict = {}
        
        for i, predictor in enumerate(all_predictors):
            predictor_predictions = predictor.predict(alleles=model_alleles, peptides=model_peptides)
            print(i, make_scores(np.asarray(predictor_predictions), np.asarray(model_affinities)))
            score = make_scores(np.asarray(predictor_predictions), np.asarray(model_affinities))
            score_sum = score['auc'] + score['f1'] + score['tau']
            best_scores_dict[i] = score_sum
        
        sorted_scores = sorted(best_scores_dict.items(), key=operator.itemgetter(1), reverse=True)
        final_predictor.merge([all_predictors[score[0]] for score in sorted_scores[:args.ensemble_size]])
        
        import ipdb;
        ipdb.set_trace()

        '''
        map_fn = map
        
        if args.use_kubeface:
            client = kubeface.Client.from_args(args)            
            map_fn = client.map

        inputs = inputs
        results = map_fn(train, inputs)
 
        final_predictor = Class1AffinityPredictor()
        final_predictor.merge(results)
        final_predictor.save(args.out_models_dir)
        '''


if __name__ == '__main__':
    run()
