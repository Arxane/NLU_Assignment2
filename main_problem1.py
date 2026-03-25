import argparse
from typing import List

from problem1_task1 import task1_prepare_dataset
from problem1_task2 import task2_train_word2vec
from problem1_task3 import task3_semantic_analysis
from problem1_task4 import task4_visualization


def run_all(
    sources: str = "sources.txt",
    viz: str = "pca",
    dimensions: List[int] = None,
    windows: List[int] = None,
    negatives: List[int] = None,
    epochs: int = 20,
    min_count: int = 2,
    crawl_depth: int = 2,
    max_pages: int = 120,
) -> None:
    # run the full task 1-4 pipeline
    print("\n" + "=" * 70)
    print("TASK 1: Prepare Dataset (Crawl, Download, Tokenize)")
    print("=" * 70)
    task1_prepare_dataset(sources_rel=sources, crawl_depth=crawl_depth, max_pages=max_pages)

    # fall back to assignment grid if cli args are omitted
    dimensions = dimensions or [100, 200, 300]
    windows = windows or [3, 5, 8]
    negatives = negatives or [5, 10]

    print("\n" + "=" * 70)
    print("TASK 2: Train Word2Vec Models (CBOW & Skip-gram)")
    print("=" * 70)
    task2_train_word2vec(
        corpus_rel="deliverables/task1/clean_corpus.txt",
        dimensions=dimensions,
        windows=windows,
        negatives=negatives,
        epochs=epochs,
        min_count=min_count,
    )

    print("\n" + "=" * 70)
    print("TASK 3: Semantic Analysis (Neighbors & Analogies)")
    print("=" * 70)
    task3_semantic_analysis(best_models_rel="deliverables/task2/task2_best_models.json")

    print("\n" + "=" * 70)
    print("TASK 4: Visualization (2D Embeddings)")
    print("=" * 70)
    task4_visualization(best_models_rel="deliverables/task2/task2_best_models.json", method=viz)

    print("\n" + "=" * 70)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("Check the 'deliverables/' folder for all outputs.")


def parse_args() -> argparse.Namespace:
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Word2Vec from-scratch pipeline: All tasks")

    parser.add_argument("--sources", default="sources.txt", help="Path to sources file")
    parser.add_argument("--crawl-depth", type=int, default=2, help="Crawl depth")
    parser.add_argument("--max-pages", type=int, default=120, help="Max pages to crawl")
    parser.add_argument("--viz", choices=["pca", "tsne"], default="pca", help="Visualization method")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[100, 200, 300], help="Embedding dimensions")
    parser.add_argument("--windows", nargs="+", type=int, default=[3, 5, 8], help="Context window sizes")
    parser.add_argument("--negatives", nargs="+", type=int, default=[5, 10], help="Negative sampling counts")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--min-count", type=int, default=2, help="Min word count threshold")

    return parser.parse_args()


if __name__ == "__main__":
    # entrypoint for running the full pipeline from terminal
    args = parse_args()
    run_all(
        sources=args.sources,
        viz=args.viz,
        dimensions=args.dimensions,
        windows=args.windows,
        negatives=args.negatives,
        epochs=args.epochs,
        min_count=args.min_count,
        crawl_depth=args.crawl_depth,
        max_pages=args.max_pages,
    )
