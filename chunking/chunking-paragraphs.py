"""
Chunk EU-style legal PDFs (e.g. EU AI Act) into structured JSON.

Each chunk contains metadata:
{
  "id": "article-6-para-2",
  "type": "recital" | "article" | "annex" | "other",
  "paragraph_number": 2,
  "page_range": "5/144",
  "chapter_number": "III",
  "chapter_name": "HIGH-RISK AI SYSTEMS",
  "section_number": "1",
  "section_name": "Obligations of providers and deployers...",
  "article_number": 6,
  "article_name": "Classification rules for high-risk AI systems",
  "annex_number": "V",
  "annex_name": "EU declaration of conformity",
  "text": "Cleaned text"
}

Dependencies:
    pip install pymupdf tqdm
"""

import fitz
import re
import json
import collections
from tqdm import tqdm

# ---------- CONFIG ----------
PDF_PATH = "chunking/AIACT_new.pdf"
OUTPUT_JSON = "chunking/aia_chunks.json"

RECITALS_TO_DROP = {"180"}
CHAPTERS_TO_DROP = {"XIII"}
ANNEXES_TO_DROP = {"I", "X"}

TOP_SCAN_LINES = 3
BOTTOM_SCAN_LINES = 3
HF_FREQ_THRESHOLD = 0.55

# ---------- REGEX ----------
RE_CHAPTER = re.compile(r'^\s*CHAPTER\s+([IVXLCDM]+)\b(?:\s*[-–—:]\s*(.*))?$', re.I)
RE_ARTICLE = re.compile(r'^\s*Article\s+(\d+)\b(?:\s*[-–—:]\s*(.*))?$', re.I)
RE_RECITAL = re.compile(r'^\(\s*(\d+)\s*\)\s*(.*)$')
RE_SECTION = re.compile(r'^\s*SECTION\s+(\d+)\b(?:\s*[-–—:]\s*(.*))?$', re.I)
RE_ANNEX = re.compile(r'^\s*ANNEX\s+([IVXLCDM]+)\b(?:\s*[-–—:]\s*(.*))?$', re.I)

RE_ARTICLE_PARAGRAPH = re.compile(r'^(\d+)\.\s*')
RE_ARTICLE_PAREN_NUM = re.compile(r'^\(\s*(\d+)\s*\)\s*')
RE_ANNEX_PARAGRAPH_NUMBER = re.compile(r'^(?:\(\s*(\d+)\s*\)|(\d+(?:\.\d+)*))\s*')

RE_OJ_LINE = re.compile(r'(OJ\s+L\b|Official Journal|Regulation\s*\(EU\)\s*No|Directive\s*\d+/\d+|Commission\s+(Decision|Implementing))', re.I)
RE_ELI = re.compile(r'ELI\s*:\s*https?://', re.I)
RE_INLINE_REF_NUMBER_ONLY = re.compile(r'^\s*\(\s*\d+\s*\)\s*$')
RE_LONG_REG_REF = re.compile(r'(Regulation\s*\(EU\).*?OJ\s+L\s*\d+,\s*\d+\.\d+\.\d+|\bDirective\s*\(\s*EU\s*\).+OJ)', re.I | re.S)
RE_PAGE_NUMBER = re.compile(r'^\d+/\d+$')
RE_FOOTNOTE = re.compile(r'^\(\d+\)\s')

# ---------- UTILITIES ----------
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    return [{"page_num": i + 1,
             "text": (p := doc.load_page(i)).get_text("text"),
             "dict": p.get_text("dict")} for i in range(doc.page_count)], doc.page_count

def guess_headers_footers(pages):
    top, bottom = collections.Counter(), collections.Counter()
    n = len(pages)
    for p in pages:
        lines = [ln.strip() for ln in p["text"].splitlines() if ln.strip()]
        for ln in lines[:TOP_SCAN_LINES]:
            if not RE_PAGE_NUMBER.match(ln):
                top[ln] += 1
        for ln in lines[-BOTTOM_SCAN_LINES:]:
            if not RE_PAGE_NUMBER.match(ln):
                bottom[ln] += 1
    headers = {t for t, c in top.items() if c >= n * HF_FREQ_THRESHOLD and not re.match(r'^[\W_]*\d+[\W_]*$', t)}
    footers = {b for b, c in bottom.items() if c >= n * HF_FREQ_THRESHOLD and not re.match(r'^[\W_]*\d+[\W_]*$', b)}
    return headers, footers

def strip_headers_footers(text, headers, footers):
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s in headers or s in footers or RE_ELI.search(s):
            continue
        out.append(ln)
    return "\n".join(out)

def clean_reference_lines(text):
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or RE_OJ_LINE.search(s) or RE_LONG_REG_REF.search(s) or RE_INLINE_REF_NUMBER_ONLY.match(s) or "OJ L" in s:
            continue
        out.append(ln)
    return "\n".join(out)

def remove_footnotes(text):
    cleaned, skip = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            cleaned.append("")
            continue
        if RE_FOOTNOTE.match(s) and any(k in s for k in ("Regulation", "Directive", "OJ L", "Decision")):
            skip = True
            continue
        if skip:
            if not s or s.endswith("."):
                skip = False
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)

def remove_superscript_refs(page_dict):
    out = []
    for block in page_dict["blocks"]:
        for line in block.get("lines", []):
            txt = "".join(span["text"] for span in line.get("spans", []) if span["size"] >= 7 and span["text"].strip() not in {"*", "**"})
            if txt.strip():
                out.append(txt.strip())
    return "\n".join(out)

def normalize_whitespace(s):
    return re.sub(r'\n{3,}', '\n\n', s.strip())

def roman_upper(s):
    return s.strip().upper() if s else None

def is_bold_line(line, page_dict):
    line = line.strip()
    if not line:
        return False
    for b in page_dict["blocks"]:
        for l in b.get("lines", []):
            span_txt = " ".join(s["text"].strip() for s in l.get("spans", []) if s["text"].strip())
            if span_txt.strip() == line:
                return all("Bold" in s["font"] for s in l["spans"] if s["text"].strip())
    return False

def collect_bold_title_lines(i, stream, pages):
    title, j = [], i
    while j < len(stream):
        pnum, cand = stream[j]
        if cand.strip() and is_bold_line(cand.strip(), pages[pnum - 1]["dict"]):
            title.append(cand.strip())
            j += 1
        else:
            break
    return (" ".join(title), j - 1) if title else (None, i)

# ---------- MAIN ----------
def parse_pdf_to_chunks(pdf_path, output_json):
    pages, total = extract_pages(pdf_path)
    headers, footers = guess_headers_footers(pages)
    print(f"Detected headers: {list(headers)[:3]} | footers: {list(footers)[:3]}")

    chunks, stream = [], []
    for p in pages:
        text = remove_superscript_refs(p["dict"])
        text = strip_headers_footers(text, headers, footers)
        text = remove_footnotes(text)
        for ln in text.splitlines():
            if not RE_PAGE_NUMBER.match(ln.strip()):
                stream.append((p["page_num"], ln.strip()))

    def emit_chunk(kind, lines, pages_set, meta):
        if not lines:
            return
        txt = normalize_whitespace(clean_reference_lines("\n".join(lines)))
        if not txt:
            return
        start, end = min(pages_set), max(pages_set)
        prange = f"{start}/{total}" if start == end else f"{start}-{end}/{total}"
        chunks.append({
            "id": meta.get("id", f"{kind}-{len(chunks)+1}"),
            "type": kind,
            "paragraph_number": meta.get("paragraph_number"),
            "page_range": prange,
            "chapter_number": meta.get("chapter_number"),
            "chapter_name": meta.get("chapter_name"),
            "section_number": meta.get("section_number"),
            "section_name": meta.get("section_name"),
            "article_number": meta.get("article_number"),
            "article_name": meta.get("article_name"),
            "annex_number": meta.get("annex_number"),
            "annex_name": meta.get("annex_name"),
            "text": txt
        })

    # --- state ---
    buf, pages_set, mode = [], set(), "recital"
    cur_ch, cur_ch_name = None, None
    cur_sec, cur_sec_name = None, None
    cur_art_meta, cur_ann_num, cur_ann_name = {}, None, None
    cur_rec = None
    skip_block = False
    i = 0

    while i < len(stream):
        page, ln = stream[i]
        s = ln.strip()
        if not s:
            if buf and buf[-1] != "":
                buf.append("")
                pages_set.add(page)
            i += 1
            continue

        # --- ANNEX ---
        m = RE_ANNEX.match(s)
        if m:
            if buf and mode == "recital" and cur_rec:
                emit_chunk("recital", buf, pages_set, {"paragraph_number": int(cur_rec)})
            elif buf and cur_art_meta.get("article_number"):
                emit_chunk("article", buf, pages_set, cur_art_meta)
            buf, pages_set = [], set()
            cur_ann_num = roman_upper(m.group(1))
            cur_ann_name = (m.group(2) or "").strip()
            if not cur_ann_name:
                t, new_i = collect_bold_title_lines(i + 1, stream, pages)
                if t:
                    cur_ann_name, i = t, new_i
            mode, skip_block = "annex", cur_ann_num in ANNEXES_TO_DROP
            i += 1
            continue

        # --- CHAPTER ---
        m = RE_CHAPTER.match(s)
        if m:
            if mode == "recital" and cur_rec:
                emit_chunk("recital", buf, pages_set, {"paragraph_number": int(cur_rec)})
            elif buf and cur_art_meta.get("article_number"):
                emit_chunk("article", buf, pages_set, cur_art_meta)
            buf, pages_set = [], set()
            cur_ch, cur_ch_name = roman_upper(m.group(1)), (m.group(2) or "").strip()
            if not cur_ch_name:
                t, new_i = collect_bold_title_lines(i + 1, stream, pages)
                if t:
                    cur_ch_name, i = t, new_i
            mode, skip_block = "chapter", cur_ch in CHAPTERS_TO_DROP
            i += 1
            continue

        # --- SECTION ---
        m = RE_SECTION.match(s)
        if m:
            if buf and cur_art_meta.get("article_number"):
                emit_chunk("article", buf, pages_set, cur_art_meta)
                buf, pages_set = [], set()
            cur_sec, cur_sec_name = m.group(1), (m.group(2) or "").strip()
            if not cur_sec_name:
                t, new_i = collect_bold_title_lines(i + 1, stream, pages)
                if t:
                    cur_sec_name, i = t, new_i
            mode, skip_block = "section", False
            i += 1
            continue

        # --- ARTICLE ---
        m = RE_ARTICLE.match(s)
        if m and mode != "annex":
            if buf and cur_art_meta.get("article_number"):
                emit_chunk("article", buf, pages_set, cur_art_meta)
            if mode == "recital" and cur_rec and buf:
                emit_chunk("recital", buf, pages_set, {"paragraph_number": int(cur_rec)})
            buf, pages_set = [], set()
            num, name = int(m.group(1)), (m.group(2) or "").strip()
            if not name:
                t, new_i = collect_bold_title_lines(i + 1, stream, pages)
                if t:
                    name, i = t, new_i
            cur_art_meta = {
                "id": f"article-{num}",
                "paragraph_number": None,
                "chapter_number": cur_ch,
                "chapter_name": cur_ch_name,
                "section_number": cur_sec,
                "section_name": cur_sec_name,
                "article_number": num,
                "article_name": name
            }
            mode, skip_block = "article", False
            i += 1
            continue

        # --- RECITAL ---
        m = RE_RECITAL.match(s)
        if mode == "recital" and m:
            if cur_rec and buf and cur_rec not in RECITALS_TO_DROP:
                emit_chunk("recital", buf, pages_set, {"paragraph_number": int(cur_rec)})
            cur_rec, buf = m.group(1), [m.group(2).strip()] if m.group(2).strip() else []
            pages_set, skip_block = {page}, cur_rec in RECITALS_TO_DROP
            i += 1
            continue

        if skip_block:
            i += 1
            continue

        # --- ARTICLE PARAGRAPH ---
        if mode == "article":
            m1, m2 = RE_ARTICLE_PARAGRAPH.match(s), RE_ARTICLE_PAREN_NUM.match(s)
            if m1 or m2:
                if buf:
                    emit_chunk("article", buf, pages_set, cur_art_meta)
                after = s[(m1.end() if m1 else m2.end()):].strip()
                buf, pages_set = ([after] if after else []), {page}
                para_no = int(m1.group(1) if m1 else m2.group(1))
                cur_art_meta.update({"paragraph_number": para_no, "id": f"article-{cur_art_meta['article_number']}-para-{para_no}"})
                i += 1
                continue

        # --- ANNEX PARAGRAPH ---
        if mode == "annex":
            m = RE_ANNEX_PARAGRAPH_NUMBER.match(s)
            if m:
                para_raw = m.group(1) or m.group(2)
                try:
                    para_no = int(para_raw)
                except ValueError:
                    para_no = para_raw  # keep composite like "3.1"
                if buf:
                    emit_chunk("annex", buf, pages_set, {
                        "id": f"annex-{cur_ann_num}-para-{para_no}",
                        "annex_number": cur_ann_num,
                        "annex_name": cur_ann_name,
                        "paragraph_number": para_no
                    })
                after = s[m.end():].strip()
                buf, pages_set = ([after] if after else []), {page}
                i += 1
                continue


        # --- default accumulate ---
        buf.append(ln)
        pages_set.add(page)
        i += 1

    # --- finalize ---
    if mode == "recital" and cur_rec and cur_rec not in RECITALS_TO_DROP:
        emit_chunk("recital", buf, pages_set, {"paragraph_number": int(cur_rec)})
    elif mode == "article" and cur_art_meta.get("article_number") and buf:
        emit_chunk("article", buf, pages_set, cur_art_meta)
    elif mode == "annex" and cur_ann_num:
        emit_chunk("annex", buf, pages_set, {"annex_number": cur_ann_num, "annex_name": cur_ann_name})

    final = [
        {**c, "text": normalize_whitespace(clean_reference_lines(c["text"]))}
        for c in chunks
        if not (
            (c["type"] == "recital" and str(c.get("paragraph_number")) in RECITALS_TO_DROP)
            or (c["type"] == "annex" and c.get("annex_number") in ANNEXES_TO_DROP)
            or (c.get("chapter_number") in CHAPTERS_TO_DROP)
        )
    ]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"source_file": pdf_path, "total_pages": total, "chunks": final}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(final)} chunks to {output_json}")
    return final

# ---------- RUN ----------
if __name__ == "__main__":
    parse_pdf_to_chunks(PDF_PATH, OUTPUT_JSON)
