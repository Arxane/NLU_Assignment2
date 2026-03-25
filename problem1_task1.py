# task 1: crawl, download, and clean the corpus

from collections import Counter
from pathlib import Path
from typing import Dict, List

from shared_utils import (
    crawl_urls,
    corpus_stats,
    deliverables_dir,
    ensure_dirs,
    fetch_url,
    generate_wordcloud,
    load_json,
    process_document,
    raw_downloads_dir,
    read_lines,
    root_dir,
    save_raw,
    write_json,
)


def task1_prepare_dataset(sources_rel: str = "sources.txt", crawl_depth: int = 2, max_pages: int = 120) -> Dict:
    # prepare the dataset and tokenize documents
    root = root_dir()
    sources_path = root / sources_rel
    raw_dir = raw_downloads_dir()
    task1_dir = deliverables_dir() / "task1"
    corpus_file = task1_dir / "clean_corpus.txt"
    ensure_dirs([raw_dir, task1_dir, corpus_file.parent])

    # load seed urls from sources file
    seed_urls = read_lines(sources_path)
    urls_with_depth = crawl_urls(seed_urls, max_depth=crawl_depth, max_pages=max_pages)
    urls = [url for url, _ in urls_with_depth]

    print(
        f"Crawl complete: {len(urls)} URLs discovered from {len(seed_urls)} seeds "
        f"(depth={crawl_depth}, max_pages={max_pages})"
    )

    documents = []
    doc_tokens: List[List[str]] = []
    metadata = []

    # keep crawl depth for each url for reporting
    url_depth_map = {url: depth for url, depth in urls_with_depth}

    for url in urls:
        try:
            content = fetch_url(url)
            raw_file = save_raw(raw_dir, url, content)
            text, tokens = process_document(url, content)
            # skip docs that become empty after cleaning
            if not tokens:
                continue
            documents.append(text)
            doc_tokens.append(tokens)
            metadata.append(
                {
                    "url": url,
                    "depth": url_depth_map.get(url, 0),
                    "raw_file": str(raw_file.relative_to(root)),
                    "token_count": len(tokens),
                }
            )
            print(f"Processed: {url} ({len(tokens)} tokens)")
        except Exception as exc:
            print(f"Skipped: {url} ({exc})")

    if not doc_tokens:
        raise RuntimeError("No documents were processed. Update sources.txt with reachable IITJ URLs.")

    with corpus_file.open("w", encoding="utf-8") as f:
        for tokens in doc_tokens:
            f.write(" ".join(tokens) + "\n")

    # base corpus stats + crawl metadata
    stats = corpus_stats(doc_tokens)
    stats["sources_used"] = len(metadata)
    stats["seed_sources"] = len(seed_urls)
    stats["crawl_depth"] = crawl_depth
    stats["max_pages"] = max_pages
    stats["discovered_urls"] = len(urls)
    stats["documents"] = metadata

    stats_file = task1_dir / "dataset_stats.json"
    write_json(stats, stats_file)

    freq = Counter(tok for doc in doc_tokens for tok in doc)
    wordcloud_file = task1_dir / "wordcloud_task1.png"
    generate_wordcloud(freq, wordcloud_file)

    summary_file = task1_dir / "task1_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("TASK-1 DATASET SUMMARY\n")
        f.write(f"Total documents: {stats['total_documents']}\n")
        f.write(f"Total tokens: {stats['total_tokens']}\n")
        f.write(f"Vocabulary size: {stats['vocabulary_size']}\n")
        f.write(f"Sources used: {stats['sources_used']}\n")
        f.write(f"Seed URLs: {stats['seed_sources']}\n")
        f.write(f"Discovered URLs: {stats['discovered_urls']}\n")
        f.write(f"Crawl depth: {stats['crawl_depth']}\n")
        f.write("Top words:\n")
        for word, count in stats["top_30_words"][:20]:
            f.write(f"  {word}: {count}\n")

    print(f"Saved corpus: {corpus_file}")
    print(f"Saved stats: {stats_file}")
    print(f"Saved wordcloud: {wordcloud_file}")
    print(f"Saved summary: {summary_file}")
    return stats


if __name__ == "__main__":
    task1_prepare_dataset()
