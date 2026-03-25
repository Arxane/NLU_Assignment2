# NLU Assignment 2

this repo contains two parts:
- problem 1: word2vec from scratch pipeline (crawl -> train -> analyze -> visualize)
- problem 2: character-level name generation with rnn, blstm, and attention

## problem 1
problem 1 builds custom cbow and skip-gram embeddings without gensim.

flow:
1. collect and clean corpus data from urls
2. train multiple cbow/skip-gram models with hyperparameter combinations
3. run semantic checks (neighbors + analogies)
4. visualize embedding space in 2d (pca/tsne)

main entry:
- `main_problem1.py`

## problem 2
problem 2 trains sequence models on indian names and compares generation quality.

flow:
1. load names and build character vocabulary
2. train three models: rnn, blstm, attention
3. generate sample names
4. evaluate novelty/diversity and save plots + metrics

main entry:
- `main_problem2.py`

## file guide

core scripts:
- `main_problem1.py`: runs complete problem 1 pipeline
- `main_problem2.py`: runs complete problem 2 pipeline

problem 1 task files:
- `problem1_task1.py`: dataset preparation (crawl, clean, corpus outputs)
- `problem1_task2.py`: word2vec training + hyperparameter search
- `problem1_task3.py`: semantic analysis (neighbors + analogies)
- `problem1_task4.py`: embedding visualization (pca/tsne)

shared and support:
- `shared_utils.py`: common utilities + from-scratch word2vec implementation
- `requirements.txt`: python dependencies
- `check_cuda.py`: quick cuda availability check

data and outputs:
- `TrainingNames.txt`: training data used by problem 2
- `sources.txt`: list of all links that have been used for crawling
- `deliverables/`: problem 1 outputs by task
- `models/`: saved word2vec models
- `problem2_outputs/`: generated samples, metrics, and plots for problem 2
- `raw_downloads/`: downloaded raw pages/pdfs from crawling

## how to run

### 1) setup
```bash
pip install -r requirements.txt
```

### 2) run problem 1
```bash
python main_problem1.py
```

optional flags:
```bash
python main_problem1.py --crawl-depth 2 --max-pages 120 --viz pca
```

### 3) run problem 2
```bash
python main_problem2.py
```

### 4) run individual problem 1 tasks (optional)
```bash
python problem1_task1.py
python problem1_task2.py
python problem1_task3.py
python problem1_task4.py
```

## where to check results
- problem 1: `deliverables/task1`, `deliverables/task2`, `deliverables/task3`, `deliverables/task4`
- problem 2: `problem2_outputs`


## external help used
i used ai support mainly to speed up repetitive work, like:
- refactoring repeated code blocks
- cleaning up comments and making style consistent
- drafting and polishing documentation text

i also used it for a few more complex parts as a thinking partner, such as:
- checking edge cases in crawling and text cleaning
- validating logic for cbow/skip-gram training pipeline
- improving readability of pipeline

all final code decisions, testing, and output checks were done manually in this project setup.
