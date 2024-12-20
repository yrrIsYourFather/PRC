# PRL: Can Parallel Reservoir Computing with Linear Modules Support DDM?

![overview](/fig/overview-1.jpg)

This is our final project for the course "Introduction to Neuroscience" (2024 Fall), supervised by Prof. T.Yang. For more details, we refer to [our report](/PRL.pdf).

Before running our code, use `conda env create -f environment.yml` to create your conda environment.

## Model

[./model](/model) is our code for PRL model, which contains the model itself, training and evaluation subparts.

## Data

Our dataset is developed based on https://github.com/tyangLab/Sequence_learning. We appreciate this great work! 

You can use it to generate datasets and specify shape sequence for analysis.

## Analysis

[./analysis](/analysis) and [./fig](/fig) are our code for model analysis. 

When running the [figD.py](/fig/figD.py) file, you need to specify the generated shape sequence in [dataset.py](/data/dataset.py). Other code can be run directly.
