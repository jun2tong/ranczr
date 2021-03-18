# RANCZR

This repo is for my attempt at Kaggle's RANCZR clip competition.

Our team came at 15-th position. Main strategy is pseudo label NIH and MIMIC dataset and use those dataset to pre-train a model which then be used to train on competition dataset.

Models are ensembled through weighted average. Stacking models provided best score sufficient for a gold, but we didn't trust our CV. *sad face*

