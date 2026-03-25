# task 3: semantic analysis with neighbors and analogies

from pathlib import Path
from typing import Dict, List, Tuple

from shared_utils import (
    ANALOGIES,
    TARGET_WORDS,
    ScratchWord2Vec,
    deliverables_dir,
    ensure_dirs,
    load_json,
    root_dir,
    write_json,
)


def get_neighbors(model: ScratchWord2Vec, word: str, k: int = 5) -> List[Tuple[str, float]]:
    # return [] instead of raising for out-of-vocab words
    if word not in model.wv.key_to_index:
        return []
    return model.wv.most_similar(word, topn=k)


def solve_analogy(model: ScratchWord2Vec, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
    # analogy vector is c + b - a
    for w in (a, b, c):
        if w not in model.wv.key_to_index:
            return []
    return model.wv.most_similar(positive=[c, b], negative=[a], topn=k)


def analyze_model(model_path: Path) -> Dict:
    # run both analyses so reports stay comparable across models
    model = ScratchWord2Vec.load(str(model_path))
    neighbors = {word: get_neighbors(model, word, k=5) for word in TARGET_WORDS}

    analogies = {}
    for a, b, c in ANALOGIES:
        key = f"{a}:{b}::{c}:?"
        analogies[key] = solve_analogy(model, a, b, c, k=5)

    return {
        "model_path": str(model_path),
        "neighbors": neighbors,
        "analogies": analogies,
    }


def format_semantic_report(results: Dict[str, Dict]) -> str:
    # get statistical insights 
    lines = ["TASK-3 SEMANTIC ANALYSIS", ""]
    for label, payload in results.items():
        lines.append(f"Model: {label}")
        lines.append(f"Path: {payload['model_path']}")
        lines.append("Nearest neighbors:")
        for word, items in payload["neighbors"].items():
            if not items:
                lines.append(f"  {word}: [OOV]")
                continue
            joined = ", ".join(f"{w} ({score:.4f})" for w, score in items)
            lines.append(f"  {word}: {joined}")

        lines.append("Analogies:")
        for expr, items in payload["analogies"].items():
            if not items:
                lines.append(f"  {expr} -> [insufficient vocabulary]")
                continue
            joined = ", ".join(f"{w} ({score:.4f})" for w, score in items)
            lines.append(f"  {expr} -> {joined}")
        lines.append("")
    return "\n".join(lines)


def task3_semantic_analysis(best_models_rel: str = "deliverables/task2/task2_best_models.json") -> Dict:
    # load best paths selected in task 2
    root = root_dir()
    best_path = root / best_models_rel
    task3_dir = deliverables_dir() / "task3"
    ensure_dirs([task3_dir])
    if not best_path.exists():
        raise RuntimeError("Best model config not found. Run task2 first.")

    best_models = load_json(best_path)
    cbow_path = Path(best_models["cbow"]["model_path"])
    skip_path = Path(best_models["skipgram"]["model_path"])

    results = {
        "CBOW": analyze_model(cbow_path),
        "Skip-gram": analyze_model(skip_path),
    }

    out_json = task3_dir / "task3_semantic_results.json"
    write_json(results, out_json)

    report_text = format_semantic_report(results)
    out_txt = task3_dir / "task3_semantic_results.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_txt}")
    return results


if __name__ == "__main__":
    task3_semantic_analysis()
