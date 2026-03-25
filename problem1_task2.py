# task 2: train cbow and skip-gram with hyperparameter experiments

import itertools
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from shared_utils import (
    deliverables_dir,
    ensure_dirs,
    load_corpus,
    models_dir,
    root_dir,
    train_scratch_word2vec,
    write_json,
)


def train_model(
    corpus: List[List[str]],
    model_type: str,
    vector_size: int,
    window: int,
    negative: int,
    epochs: int,
    min_count: int,
) -> dict:
    # train one word2vec model
    model = train_scratch_word2vec(
        corpus=corpus,
        model_type=model_type,
        vector_size=vector_size,
        window=window,
        negative=negative,
        epochs=epochs,
        min_count=min_count,
        seed=42,
    )
    return model


def run_grid_search(
    corpus: List[List[str]],
    model_type: str,
    dimensions: List[int],
    windows: List[int],
    negatives: List[int],
    epochs: int,
    min_count: int,
    models_dir: Path,
) -> List[Dict]:
    # run a hyperparameter grid search
    records = []
    # full cartesian product over window, dimension, negative samples
    for vector_size, window, negative in itertools.product(dimensions, windows, negatives):
        start = time.time()
        model = train_model(
            corpus=corpus,
            model_type=model_type,
            vector_size=vector_size,
            window=window,
            negative=negative,
            epochs=epochs,
            min_count=min_count,
        )
        elapsed = time.time() - start
        model_name = f"{model_type}_dim{vector_size}_win{window}_neg{negative}.model"
        model_path = models_dir / model_name
        model.save(str(model_path))

        # keep enough metadata to compare and reload later
        records.append(
            {
                "model_type": model_type,
                "vector_size": vector_size,
                "window": window,
                "negative": negative,
                "epochs": epochs,
                "vocab_size": len(model.wv.key_to_index),
                "train_seconds": round(elapsed, 3),
                "model_path": str(model_path),
            }
        )
        print(f"Trained {model_name} in {elapsed:.2f}s")
    return records


def choose_best(records: List[Dict], model_type: str) -> Dict:
    # pick the best model by vocab size and train time
    subset = [r for r in records if r["model_type"] == model_type]
    # prefer larger vocab coverage, then faster train time
    subset = sorted(subset, key=lambda r: (-r["vocab_size"], r["train_seconds"]))
    return subset[0]


def task2_train_word2vec(
    corpus_rel: str = "deliverables/task1/clean_corpus.txt",
    dimensions: List[int] = None,
    windows: List[int] = None,
    negatives: List[int] = None,
    epochs: int = 20,
    min_count: int = 2,
) -> Dict:
    # train cbow and skip-gram over the search grid
    # defaults match the assignment hyperparameter sweep
    dimensions = dimensions or [100, 200, 300]
    windows = windows or [3, 5, 8]
    negatives = negatives or [5, 10]

    root = root_dir()
    corpus_path = root / corpus_rel
    model_output_dir = models_dir()
    task2_dir = deliverables_dir() / "task2"
    ensure_dirs([model_output_dir, task2_dir])

    corpus = load_corpus(corpus_path)
    if not corpus:
        raise RuntimeError("Corpus is empty. Run task1 first.")

    # searches for best cbow and skip-gram models seperately, then saves all records and best config
    all_records: List[Dict] = []
    for model_type in ["cbow", "skipgram"]:
        all_records.extend(
            run_grid_search(
                corpus=corpus,
                model_type=model_type,
                dimensions=dimensions,
                windows=windows,
                negatives=negatives,
                epochs=epochs,
                min_count=min_count,
                models_dir=model_output_dir,
            )
        )

    df = pd.DataFrame(all_records)
    csv_path = task2_dir / "task2_hyperparameter_results.csv"
    df.to_csv(csv_path, index=False)

    best = {
        "cbow": choose_best(all_records, "cbow"),
        "skipgram": choose_best(all_records, "skipgram"),
    }
    json_path = task2_dir / "task2_best_models.json"
    write_json(best, json_path)

    print(f"Saved experiment table: {csv_path}")
    print(f"Saved best-model config: {json_path}")
    return best


if __name__ == "__main__":
    task2_train_word2vec()
