#!/usr/bin/env python3
import re
import unicodedata
from collections import Counter
from typing import List, Tuple


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # normalize unicode and replace non-breaking spaces which often break regex
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")
    return text


def split_articles_by_header(text: str) -> List[Tuple[int, str]]:
    """
    Return list of (article_number, chunk_text).
    The header regex is anchored to line-start (MULTILINE) to avoid accidental matches inside paragraphs.
    """

    # header_re = re.compile(r'\nArticle\s*(\d+)\n | ##\s*Article\s*(\d+)\n', flags=re.I | re.M)

    header_re = re.compile(r'(?:\n|##\s*)Article\s*(\d+)\s*\n', flags=re.I | re.M)
    matches = list(header_re.finditer(text))
    if not matches:
        return []

    articles = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].rstrip()
        articles.append((num, chunk))
    return articles


def verify_articles(articles: List[Tuple[int, str]]):
    if not articles:
        print("No article headers found.")
        return

    nums = [n for n, _ in articles]
    cnt = Counter(nums)
    duplicates = [n for n, c in cnt.items() if c > 1]

    mn, mx = min(nums), max(nums)
    expected = list(range(mn, mx + 1))
    missing = [n for n in expected if n not in cnt]

    order_issues = []
    for i in range(1, len(nums)):
        prev, cur = nums[i - 1], nums[i]
        if cur != prev + 1:
            order_issues.append((i, prev, cur))

    print(f"Found {len(articles)} chunks. Header numbers span {mn} .. {mx}.")
    if duplicates:
        print("Duplicate article numbers:", duplicates)
    if missing:
        print("Missing article numbers:", missing)
    if order_issues:
        print("Order problems (chunk_index, previous_number -> current_number):")
        for idx, prev, cur in order_issues[:50]:
            prev_title = articles[idx - 1][1].splitlines()[0][:80]
            cur_title = articles[idx][1].splitlines()[0][:80]
            print(
                f"  chunk {idx}: {prev} -> {cur}; prev header starts: {prev_title!r}; cur header starts: {cur_title!r}")
    if not (duplicates or missing or order_issues):
        print("All article headers look sequential and in order.")


if __name__ == "__main__":
    path = "../data/processed/AIACT.md"
    text = load_text(path)
    articles = split_articles_by_header(text)
    verify_articles(articles)

    # optional: show the last chunk header and first line for quick inspection
    if articles:
        last_num, last_chunk = articles[-1]
        print("\nLast chunk header number:", last_num)
        print("Last chunk first line:", last_chunk.splitlines()[0] if last_chunk.splitlines() else "<empty>")

    print(articles[0])
    print(articles[1])

