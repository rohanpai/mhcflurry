#!/bin/bash

set -e
set -x

DOWNLOAD_NAME=models_class1
SCRATCH_DIR=${TMPDIR-/tmp}/mhcflurry-downloads-generation
SCRIPT_ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "$SCRIPT_ABSOLUTE_PATH")

mkdir -p "$SCRATCH_DIR"
rm -rf "$SCRATCH_DIR/$DOWNLOAD_NAME"
mkdir "$SCRATCH_DIR/$DOWNLOAD_NAME"

# Send stdout and stderr to a logfile included with the archive.
exec >  >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt")
exec 2> >(tee -ia "$SCRATCH_DIR/$DOWNLOAD_NAME/LOG.txt" >&2)

# Log some environment info
date
pip freeze
git status

cd $SCRATCH_DIR/$DOWNLOAD_NAME

mkdir models

cp $SCRIPT_DIR/hyperparameters.json .

time mhcflurry-class1-train-allele-specific-models \
    --data "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" \
    --hyperparameters hyperparameters.json \
    --out-models-dir models \
    --min-measurements-per-allele 100

cp $SCRIPT_ABSOLUTE_PATH .
bzip2 LOG.txt
tar -cjf "../${DOWNLOAD_NAME}.tar.bz2" *

echo "Created archive: $SCRATCH_DIR/$DOWNLOAD_NAME.tar.bz2"
