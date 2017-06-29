'''
To run this file execute:
    >> mhcflurry-downloads fetch data_curated
    >> python kubeface_class1_allele_specific_models.py 1 --kubeface-backend kubernetes --out-csv /tmp/result --kubeface-worker-pip-packages mhcflurry
'''

import pandas
import numpy
import seaborn
import logging
import mhcflurry
import argparse
import sys
import csv
from matplotlib import pyplot
from mhcflurry import Class1AffinityPredictor

# check that the user has kubeface installed
try:
    import kubeface
except Exception as e:
    raise ImportError("Install the `kubeface` package to continue. `pip install kubeface`")

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument("n", type=int)
parser.add_argument("--out-csv")
kubeface.Client.add_args(parser)

def main_2(argv):
    args = parser.parse_args(argv)
    client = kubeface.Client.from_args(args)
    data_path = mhcflurry.downloads.get_path("data_curated", "curated_training_data.csv.bz2")
    data_df = pandas.read_csv(data_path)
    affinity_predictor = Class1AffinityPredictor()
    
    train_data = data_df.ix[
        (data_df.allele == "HLA-B*57:01") &
        (data_df.peptide.str.len() >= 8) &
        (data_df.peptide.str.len() <= 15)
    ]

    affinity_predictor.fit_allele_specific_predictors(
        n_models=args.n,
        architecture_hyperparameters={"layer_sizes": [16], "max_epochs": 10},
        peptides=train_data.peptide.values,
        affinities=train_data.measurement_value.values,
        allele="HLA-B*57:01",
        map_fn=client.map
    )


if __name__ == '__main__':
    import time
    start = time.time()
    main_2(sys.argv[1:])
    end = time.time()
    print(end - start)
