"""
Chunk EU-style PDF into JSON chunks with metadata.

Outputs a JSON file with one object per chunk:
{
  "id": "recital-1" | "article-6-para-2" | "annex-V-3",
  "type": "recital" | "article" | "annex" | "other",
  "paragraph_number": 1,                   # when applicable (recitals / article paras)
  "page_range": "5/144" or "5-7/144",
  "chapter_number": "III",                 # Roman (when present)
  "chapter_name": "HIGH-RISK AI SYSTEMS",  # when present
  "section_number": "1",                   # when present (normal numbers)
  "section_name": "...",
  "article_number": 6,
  "article_name": "Classification rules for high-risk AI systems",
  "annex_number": "V",
  "annex_name": "EU declaration of conformity",
  "text": "Cleaned chunk text"
}

Notes:
- This script uses PyMuPDF (fitz) to extract page text.
- Install dependencies: pip install pymupdf tqdm
"""

import fitz            # PyMuPDF
import re
import json
import collections
from tqdm import tqdm

# ---------- CONFIG / RULES ----------
PDF_PATH = "chunking/AIACT_no_hdr_keep_pn.pdf"   # change if different
OUTPUT_JSON = "chunking/aia_chunks.json"

# Strict lists of things the user asked to strip entirely:
RECITALS_TO_DROP = {"180"}           # recital numbers (strings)
CHAPTERS_TO_DROP = {"XIII"}          # chapter roman numerals to skip
ANNEXES_TO_DROP = {"I", "X"}         # annex roman numerals to skip

# Header/footer detection parameters
TOP_SCAN_LINES = 3
BOTTOM_SCAN_LINES = 3
HF_FREQ_THRESHOLD = 0.55   # if a top/bottom line appears on >= this fraction of pages, treat as header/footer

# Regex patterns to recognise structure (tweak if your PDF uses different wording/case)
RE_CHAPTER = re.compile(r'^\s*CHAPTER\s+([IVXLCDM]+)\b(?:\s*[\-–—:]\s*(.*))?$', re.IGNORECASE)
RE_ARTICLE = re.compile(r'^\s*Article\s+(\d+)\b(?:\s*[\-–—:]\s*(.*))?$', re.IGNORECASE)
RE_RECITAL = re.compile(r'^\(\s*(\d+)\s*\)\s*(.*)$')   # lines that start "(1) ..."
RE_ARTICLE_PARAGRAPH = re.compile(r'^(\d+)\.\s*')        # matches "1. ", "2. "
RE_ARTICLE_PAREN_NUM = re.compile(r'^\(\s*(\d+)\s*\)\s*') # matches "(1)", "(2)"
RE_SECTION = re.compile(r'^\s*SECTION\s+(\d+)\b(?:\s*[\-–—:]\s*(.*))?$', re.IGNORECASE)
RE_ANNEX = re.compile(r'^\s*ANNEX\s+([IVXLCDM]+)\b(?:\s*[\-–—:]\s*(.*))?$', re.IGNORECASE)


# Patterns to identify paragraph starts inside articles/annexes e.g. "(1)" or "(a)"
# Only split annexes on numbers like (1), (2), 3., 3.1 etc. — NOT (a), (b)
RE_PARAGRAPH_NUMBER = re.compile(r'^\(\s*([0-9]+)\s*\)\s*|^([0-9]+(\.[0-9]+)*)\s*')
# Matches only numeric paragraph markers for annexes
# Examples: "(1)", "1.", "3.1.", "4.2.3."
RE_ANNEX_PARAGRAPH_NUMBER = re.compile(r'^(?:\(\s*(\d+)\s*\)|(\d+(?:\.\d+)*))\s*')


# Patterns to remove references and OJ citations
RE_OJ_LINE = re.compile(r'(OJ\s+L\b|Official Journal|Regulation\s*\(EU\)\s*No|Directive\s*\d+/\d+|Commission\s+Decision|Commission\s+Implementing)', re.IGNORECASE)
RE_ELI = re.compile(r'ELI\s*:\s*http[s]?:\/\/', re.IGNORECASE)
RE_INLINE_REF_NUMBER_ONLY = re.compile(r'^\s*\(\s*\d+\s*\)\s*$')  # standalone "(5)" lines
RE_LONG_REG_REF = re.compile(r'(Regulation\s*\(EU\).*?OJ\s+L\s*\d+\,\s*\d+\.\d+\.\d+|\bDirective\s*\(\s*EU\s*\).+OJ)', re.IGNORECASE | re.DOTALL)
RE_PAGE_NUMBER = re.compile(r'^\d*/\d+$')  

# ---------- Utility helpers ----------
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        dict_blocks = page.get_text("dict")  # keep full metadata
        pages.append({"page_num": i + 1, "text": text, "dict": dict_blocks})
    return pages, doc.page_count

def guess_headers_footers(pages):
    """Return two sets: probable headers and probable footers, by frequency across pages.
       We ignore pure page-number lines like '1/144' for header/footer detection so they won't be removed.
    """
    top_counter = collections.Counter()
    bottom_counter = collections.Counter()
    n_pages = len(pages)
    for p in pages:
        lines = [ln.strip() for ln in p["text"].splitlines() if ln.strip()]
        # Collect top lines
        for ln in lines[:TOP_SCAN_LINES]:
            if re.match(r'^\d+\s*/\s*\d+$', ln):  # skip pure page-number as header
                continue
            top_counter[ln] += 1
        # Collect bottom lines
        for ln in lines[-BOTTOM_SCAN_LINES:]:
            if re.match(r'^\d+\s*/\s*\d+$', ln):
                continue
            bottom_counter[ln] += 1
    headers = {ln for ln, c in top_counter.items() if c >= n_pages * HF_FREQ_THRESHOLD}
    footers = {ln for ln, c in bottom_counter.items() if c >= n_pages * HF_FREQ_THRESHOLD}
    # also filter out empty strings and numeric-only
    headers = {h for h in headers if h and not re.match(r'^[\W_]*\d+[\W_]*$', h)}
    footers = {f for f in footers if f and not re.match(r'^[\W_]*\d+[\W_]*$', f)}
    return headers, footers

def strip_headers_footers(page_text, headers, footers):
    lines = page_text.splitlines()
    out_lines = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out_lines.append("")  # keep blank lines to preserve paragraph breaks
            continue
        if s in headers or s in footers:
            # skip
            continue
        # skip obvious ELI / OJ url footers even if not frequent
        if RE_ELI.search(s):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)

def clean_reference_lines(text):
    """Drop lines that look like 'Regulation (EU) ... (OJ L ...)' or '... OJ L ...' or explicit reference-only lines.
       Also remove standalone '(5)' style lines leftover.
    """
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            lines.append("") 
            continue
        # if line clearly is an OJ/reference -> drop
        if RE_OJ_LINE.search(s) or RE_LONG_REG_REF.search(s):
            continue
        if RE_INLINE_REF_NUMBER_ONLY.match(s):
            continue
        # often references are long lines ending with '.)' - optionally drop if contain 'OJ L'
        if 'OJ L' in s or 'Official Journal' in s:
            continue
        lines.append(ln)
    return "\n".join(lines)

RE_FOOTNOTE = re.compile(r'^\(\d+\)\s')  # lines starting with (number)

def remove_footnotes(text: str) -> str:
    """Remove EU law footnotes like '(8) Regulation ... (OJ L ...)'"""
    cleaned_lines = []
    skip_block = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            cleaned_lines.append("")
            continue
        # Detect start of a footnote
        if RE_FOOTNOTE.match(s) and (
            "Regulation" in s or "Directive" in s or "OJ L" in s or "Decision" in s
        ):
            skip_block = True
            continue
        # Continue skipping until a blank line or until line no longer looks like reference
        if skip_block:
            if not s or s.endswith("."):
                skip_block = False
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def normalize_whitespace(s):
    # collapse multiple blank lines and trim
    s = re.sub(r'\r\n', '\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = s.strip()
    return s

def roman_upper(s):
    if s is None:
        return None
    return s.strip().upper()

def is_bold_line(line_text, page_dict):
    """Return True if the entire line is bold in the PDF."""
    line_text = line_text.strip()
    if not line_text:
        return False

    for block in page_dict["blocks"]:
        for l in block.get("lines", []):
            span_texts = " ".join([s["text"].strip() for s in l.get("spans", []) if s["text"].strip()])
            if span_texts.strip() == line_text:
                # True if all spans in line use a bold font
                return all("Bold" in s["font"] for s in l["spans"] if s["text"].strip())
    return False

def collect_bold_title_lines(start_index, stream, pages):
    """Collect consecutive bold lines from stream[start_index:] and merge them into a single string.
       Returns (title_string, new_index).
    """
    title_lines = []
    j = start_index
    while j < len(stream):
        page_num, candidate = stream[j]
        page_dict = pages[page_num - 1]["dict"]
        if candidate.strip() and is_bold_line(candidate.strip(), page_dict):
            title_lines.append(candidate.strip())
            j += 1
        else:
            break
    if title_lines:
        return " ".join(title_lines), j - 1
    return None, start_index


# ---------- Main parsing function ----------
def parse_pdf_to_chunks(pdf_path, output_json):
    pages, total_pages = extract_pages(pdf_path)
    headers, footers = guess_headers_footers(pages)
    print(f"Detected headers (sample): {list(headers)[:5]}")
    print(f"Detected footers (sample): {list(footers)[:5]}")
    current_annex_section_num = None
    current_annex_section_name = None
    chunks = []

    # State variables
    mode = "recital"   # start in recital/preamble mode until we hit CHAPTER / Article / ANNEX
    current_chapter_num = None
    current_chapter_name = None
    current_section_num = None
    current_section_name = None
    current_article_num = None
    current_article_name = None
    current_annex_num = None
    current_annex_name = None

    # We'll build a stream of lines with (page_num, line_text) so it's easy to maintain page ranges per chunk
    stream = []
    for p in pages:
        cleaned = strip_headers_footers(p["text"], headers, footers)
        cleaned = strip_headers_footers(p["text"], headers, footers)
        cleaned = remove_footnotes(cleaned) 
        # keep page-number tokens if present (do not remove a '1/144' lone line) — they can be used to compute page ranges
        for ln in cleaned.splitlines():
            s = ln.strip()
            if RE_PAGE_NUMBER.match(s):
                continue  # skip page number lines completely
            stream.append((p["page_num"], s))


    # helper to finalize chunk
    def emit_chunk(kind, text_lines, pages_set, meta):
        if not text_lines:
            return
        raw_text = "\n".join(text_lines)
        raw_text = clean_reference_lines(raw_text)
        raw_text = normalize_whitespace(raw_text)
        if not raw_text:
            return
        start = min(pages_set)
        end = max(pages_set)
        if start == end:
            page_range = f"{start}/{total_pages}"
        else:
            page_range = f"{start}-{end}/{total_pages}"
        chunk = {
            "id": meta.get("id") or f"{kind}-{len(chunks)+1}",
            "type": kind,
            "paragraph_number": meta.get("paragraph_number"),
            "page_range": page_range,
            "chapter_number": meta.get("chapter_number"),
            "chapter_name": meta.get("chapter_name"),
            "section_number": meta.get("section_number"),
            "section_name": meta.get("section_name"),
            "article_number": meta.get("article_number"),
            "article_name": meta.get("article_name"),
            "annex_number": meta.get("annex_number"),
            "annex_name": meta.get("annex_name"),
            "text": raw_text
        }
        chunks.append(chunk)

    # Parsing pass: iterate lines, detect structure markers, and build chunks
    buf_lines = []
    buf_pages = set()
    current_recital_num = None
    current_article_buf_meta = {}
    skipping_block = False   # when we are skipping a whole chapter/annex

    i = 0
    while i < len(stream):
        page_num, ln = stream[i]
        stripped = ln.strip()

        # ignore purely empty lines
        if stripped == "":
            # keep a blank-line marker in buffers to preserve paragraph separation
            if buf_lines and buf_lines[-1] != "":
                buf_lines.append("")
                buf_pages.add(page_num)
            i += 1
            continue

        # detect structural markers first
        # ANNEX
        m_ann = RE_ANNEX.match(stripped)
        if m_ann:
            # finalize any prior buffer
            if mode == "recital" and current_recital_num is not None:
                # finalize current recital chunk
                emit_chunk("recital", buf_lines, buf_pages, {
                    "paragraph_number": int(current_recital_num),
                    "chapter_number": None,
                    "chapter_name": None,
                    "section_number": None,
                    "article_number": None
                })
            else:
                # if inside article buffer, emit it
                if buf_lines and (current_article_buf_meta.get("article_number") is not None):
                    emit_chunk("article", buf_lines, buf_pages, current_article_buf_meta)
            # reset
            buf_lines = []
            buf_pages = set()

            current_annex_num = roman_upper(m_ann.group(1))
            current_annex_name = m_ann.group(2).strip() if m_ann.group(2) else None
            
            # reset annex section state
            current_annex_section_num = None
            current_annex_section_name = None

            if not current_annex_name:
                title, new_i = collect_bold_title_lines(i + 1, stream, pages)
                if title:
                    current_annex_name = title
                    i = new_i                    

            mode = "annex"
            skipping_block = current_annex_num in ANNEXES_TO_DROP
            # skip the ANNEX heading line itself (or optionally include as header inside annex chunk)
            i += 1
            continue

        # CHAPTER
        m_ch = RE_CHAPTER.match(stripped)
        if m_ch:
            # finish in-flight buffers
            if mode == "recital" and current_recital_num is not None:
                emit_chunk("recital", buf_lines, buf_pages, {
                    "paragraph_number": int(current_recital_num),
                    "chapter_number": None,
                    "chapter_name": None,
                    "section_number": None,
                    "article_number": None
                })
            elif buf_lines and (current_article_buf_meta.get("article_number") is not None):
                emit_chunk("article", buf_lines, buf_pages, current_article_buf_meta)

            buf_lines = []
            buf_pages = set()

            current_chapter_num = roman_upper(m_ch.group(1))
            current_chapter_name = m_ch.group(2).strip() if m_ch.group(2) else None
            
            if not current_chapter_name:
                title, new_i = collect_bold_title_lines(i + 1, stream, pages)
                if title:
                    current_chapter_name = title
                    i = new_i
            
            current_section_num = None
            current_section_name = None

            mode = "chapter"
            skipping_block = current_chapter_num in CHAPTERS_TO_DROP
            i += 1
            continue


        # SECTION
        m_sec = RE_SECTION.match(stripped)
        if m_sec:
            # finalize any running article paragraphs
            if buf_lines and (current_article_buf_meta.get("article_number") is not None):
                emit_chunk("article", buf_lines, buf_pages, current_article_buf_meta)
                buf_lines = []
                buf_pages = set()

            if mode == "annex":
                # Annex section
                current_annex_section_num = m_sec.group(1)
                try:
                    current_annex_num = int(m_sec.group(1))
                except ValueError:
                    current_annex_num = None  # keep as string if not int
                current_annex_section_name = m_sec.group(2).strip() if m_sec.group(2) else None
                if not current_annex_name:
                    title, new_i = collect_bold_title_lines(i + 1, stream, pages)
                    if title:
                        current_annex_name = title
                        i = new_i

            else:
                # Normal chapter section
                current_section_num = m_sec.group(1)
                try:
                    current_section_num = int(m_sec.group(1))
                except ValueError:
                    current_section_num = current_section_num  # keep as string if not int
                current_section_name = m_sec.group(2).strip() if m_sec.group(2) else None

                if not current_section_name:
                    title, new_i = collect_bold_title_lines(i + 1, stream, pages)
                    if title:
                        current_section_name = title
                        i = new_i


            mode = "section"
            skipping_block = False
            i += 1
            continue



        # ARTICLE
        m_art = RE_ARTICLE.match(stripped)
        if m_art and mode != "annex":
            # finalize previous article if any
            if buf_lines and (current_article_buf_meta.get("article_number") is not None):
                emit_chunk("article", buf_lines, buf_pages, current_article_buf_meta)
            # finalize any current recital (if still in recital mode)
            if mode == "recital" and current_recital_num is not None and buf_lines:
                emit_chunk("recital", buf_lines, buf_pages, {
                    "paragraph_number": int(current_recital_num),
                    "chapter_number": None,
                    "chapter_name": None,
                    "section_number": None,
                    "article_number": None
                })
            # reset buffer for new article
            buf_lines = []
            buf_pages = set()
            current_article_num = int(m_art.group(1))
            current_article_name = m_art.group(2).strip() if m_art.group(2) else None

            if not current_article_name:
                title, new_i = collect_bold_title_lines(i + 1, stream, pages)
                if title:
                    current_article_name = title
                    i = new_i


            current_article_buf_meta = {
                "id": f"article-{current_article_num}",
                "paragraph_number": None,
                "chapter_number": current_chapter_num,
                "chapter_name": current_chapter_name,
                "section_number": current_section_num,
                "section_name": current_section_name,
                "article_number": current_article_num,
                "article_name": current_article_name
            }
            mode = "article"
            skipping_block = False
            i += 1
            continue


        # RECITAL (only while in preamble before first chapter/article/annex)
        m_rec = RE_RECITAL.match(stripped)
        if mode in ("recital",) and m_rec:
            # If we had a prior recital accumulated -> emit it
            if current_recital_num is not None and buf_lines:
                # check if we must drop that recital specifically
                if current_recital_num not in RECITALS_TO_DROP:
                    emit_chunk("recital", buf_lines, buf_pages, {
                        "paragraph_number": int(current_recital_num),
                        "chapter_number": None, "chapter_name": None
                    })
                # else: drop it (e.g., recital 180)
            # start new recital
            current_recital_num = m_rec.group(1)
            # if this new recital should be skipped, set skip flag
            skipping_block = current_recital_num in RECITALS_TO_DROP
            # start new buffer with the remainder of line (if any)
            initial = m_rec.group(2) or ""
            buf_lines = [initial.strip()] if initial.strip() else []
            buf_pages = {page_num}
            i += 1
            continue

        # If currently skipping a whole block (e.g., Chapter XIII or Annex I/X or Recital 180), do not accumulate
        if skipping_block:
            i += 1
            continue

        # Normal content line: append to current buffer appropriate for mode
        # If we are in article mode, we want to split into paragraphs by paragraph numbers or blank line
        if mode == "article":
            # Detect top-level numbered paragraphs (1., 2., 3.) or (1), (2)
            pm_num = RE_ARTICLE_PARAGRAPH.match(stripped)
            pm_paren = RE_ARTICLE_PAREN_NUM.match(stripped)
            if pm_num or pm_paren:
                # finalize previous paragraph buffer if present
                if buf_lines:
                    emit_chunk("article", buf_lines, buf_pages, current_article_buf_meta)
                # start new buffer
                after = stripped[(pm_num.end() if pm_num else pm_paren.end()):].strip()
                buf_lines = [after] if after else []
                buf_pages = {page_num}
                # assign paragraph_number
                para_no = pm_num.group(1) if pm_num else pm_paren.group(1)
                try:
                    para_no = int(para_no)
                except ValueError:
                    para_no = para_no  # keep as string if not int
                current_article_buf_meta["paragraph_number"] = para_no
                current_article_buf_meta["paragraph_number"] = para_no
                current_article_buf_meta["id"] = f"article-{current_article_num}-para-{para_no}"
                i += 1
                continue
            else:
                # subparagraphs like (a), (b), (i), (ii) → just aggregate
                buf_lines.append(ln)
                buf_pages.add(page_num)
                i += 1
                continue


        elif mode == "annex":
            # Annex splitting: often numbered lists or paragraph markers too. We treat lines prefixed with numbers or "(1)" as new paragraphs.
            pm = RE_ANNEX_PARAGRAPH_NUMBER.match(stripped)
            if pm:
                para_no = pm.group(1) or pm.group(2)
                try:
                    para_no = int(para_no)
                except ValueError:
                    para_no = para_no
                if buf_lines:
                    emit_chunk("annex", buf_lines, buf_pages, {
                        "id": f"annex-{current_annex_num}-para-{para_no}",
                        "annex_number": current_annex_num,
                        "annex_name": current_annex_name,
                        "section_number": current_annex_section_num,
                        "section_name": current_annex_section_name,
                        "paragraph_number": para_no
                    })
                # start new buffer
                after = stripped[pm.end():].strip()
                buf_lines = [after] if after else []
                buf_pages = {page_num}
                i += 1
                continue

            else:
                buf_lines.append(ln)
                buf_pages.add(page_num)
                i += 1
                continue

        else:
            # in other modes (recital after initial preamble might have been closed),
            # just accumulate lines until a structural token appears
            buf_lines.append(ln)
            buf_pages.add(page_num)
            i += 1
            continue

    # End of stream: finalize any buffer left
    if mode == "recital" and current_recital_num is not None:
        if current_recital_num not in RECITALS_TO_DROP:
            emit_chunk("recital", buf_lines, buf_pages, {"paragraph_number": int(current_recital_num)})
    elif mode == "article" and current_article_buf_meta.get("article_number") is not None and buf_lines:
        emit_chunk("article", buf_lines, buf_pages, current_article_buf_meta)
    elif mode == "annex" and current_annex_num is not None:
        emit_chunk("annex", buf_lines, buf_pages, {
            "annex_number": current_annex_num,
            "annex_name": current_annex_name
        })

    # Postprocessing: filter out annexes in skip list (if any slipped through) and remove any empty text
    final_chunks = []
    for c in chunks:
        # drop unwanted annexes/chapters/recitals if present
        if c["type"] == "recital" and c.get("paragraph_number") is not None and str(c["paragraph_number"]) in RECITALS_TO_DROP:
            continue
        if c["type"] == "annex" and c.get("annex_number") and c["annex_number"] in ANNEXES_TO_DROP:
            continue
        if c["chapter_number"] in CHAPTERS_TO_DROP:
            continue
        # Remove reference-like trailing sentences (extra safety)
        c["text"] = clean_reference_lines(c["text"])
        c["text"] = normalize_whitespace(c["text"])
        final_chunks.append(c)

    # Write JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"source_file": pdf_path, "total_pages": total_pages, "chunks": final_chunks}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(final_chunks)} chunks to {output_json}")
    return final_chunks

# ---------- Run ----------
if __name__ == "__main__":
    chunks = parse_pdf_to_chunks(PDF_PATH, OUTPUT_JSON)
