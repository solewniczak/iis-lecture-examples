# How to run the examples
1. Install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
2. Create new environment from environment.yaml: `conda env create -f environment.yml`
3. For Yelp examples:
   1. Download the Yelp reviews dataset: https://www.yelp.com/dataset
   2. Provide the filepath to Yelp reviews dataset in `args.py`
4. For Machine Translation examples:
   1. Download the translation pairs from: https://www.manythings.org/anki/ to `data/{lang}.txt`
   It must be English to other language dataset.
   2. Set the language code in `args.py`
5. Train the model: `python train.py`
6. Test the model: `python test.py`
7. Plot the learning epochs: `python plot.py`