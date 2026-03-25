# task 4: 2d embedding plots using pca or t-sne

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from shared_utils import (
    SEED_WORDS,
    ScratchWord2Vec,
    deliverables_dir,
    ensure_dirs,
    load_json,
    root_dir,
    write_json,
)


def collect_words(model: ScratchWord2Vec, limit_top_vocab: int = 120) -> List[str]:
    # start from curated seed words, then add top vocab for context
    words = [w for w in SEED_WORDS if w in model.wv.key_to_index]
    words.extend(list(model.wv.index_to_key)[:limit_top_vocab])
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def reduce_vectors(vectors: np.ndarray, method: str = "pca") -> np.ndarray:
    # use pca for speed, t-sne for non-linear neighborhood view
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(vectors)
    reducer = TSNE(n_components=2, random_state=42, perplexity=20, max_iter=1200, init="pca")
    return reducer.fit_transform(vectors)


def plot_embedding(model: ScratchWord2Vec, output_png: Path, model_label: str, method: str = "pca") -> None:
    # annotate only seed words to avoid label clutter
    words = collect_words(model)
    vectors = np.array([model.wv[w] for w in words])
    reduced = reduce_vectors(vectors, method=method)

    plt.figure(figsize=(13, 9))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=40, alpha=0.75)
    for i, word in enumerate(words):
        if word in SEED_WORDS:
            plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=9)

    plt.title(f"{model_label} Word Embeddings ({method.upper()} 2D Projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def task4_visualization(best_models_rel: str = "deliverables/task2/task2_best_models.json", method: str = "pca") -> Dict:
    # generate one plot per best model
    root = root_dir()
    best_path = root / best_models_rel
    task4_dir = deliverables_dir() / "task4"
    ensure_dirs([task4_dir])
    best_models = load_json(best_path)

    cbow_model = ScratchWord2Vec.load(best_models["cbow"]["model_path"])
    skip_model = ScratchWord2Vec.load(best_models["skipgram"]["model_path"])

    out_cbow = task4_dir / f"task4_{method}_cbow.png"
    out_skip = task4_dir / f"task4_{method}_skipgram.png"

    plot_embedding(cbow_model, out_cbow, "CBOW", method=method)
    plot_embedding(skip_model, out_skip, "Skip-gram", method=method)

    interp = {
        "method": method,
        "note": (
            "Visually compare local cluster compactness and semantic neighborhood spread. "
            "Skip-gram often separates rare or technical words better, while CBOW may form smoother dense clusters."
        ),
        "files": {
            "cbow": str(out_cbow),
            "skipgram": str(out_skip),
        },
    }

    out_json = task4_dir / "task4_interpretation.json"
    write_json(interp, out_json)

    print(f"Saved: {out_cbow}")
    print(f"Saved: {out_skip}")
    print(f"Saved: {out_json}")
    return interp


if __name__ == "__main__":
    task4_visualization()
