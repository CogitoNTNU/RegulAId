# split_ai_act.py
# pip install pymupdf
import sys
import re
from pathlib import Path
import fitz  # PyMuPDF

# ---------- paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PDF  = SCRIPT_DIR / "AI-act.pdf"
OUTPUT_PDF = SCRIPT_DIR / "AIACT_no_hdr_keep_pn.pdf"

if not INPUT_PDF.exists():
    print(f"ERROR: input PDF not found -> {INPUT_PDF}")
    sys.exit(1)

# ---------- settings ----------
TOP_ZONE_PT = 160          # only lines in this top band are considered header candidates
FOOTER_PT = 90             # footer band to clean while keeping page number
PAD_BELOW_WHEREAS = 4      # pad below "Whereas" when trimming

# page-number line like "123/144" (or "Page 123/144")
PN_RE = re.compile(r"^\s*(?:Page\s*)?\d+\s*/\s*\d+\s*$", re.IGNORECASE)

# header-looking lines we want to remove (unless they contain ANNEX)
HEADER_LINE_MATCHERS = [
    re.compile(r"^\s*EN\s*$", re.IGNORECASE),
    re.compile(r"^\s*OJ\s*L\b.*\d{1,2}\.\d{1,2}\.\d{4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*Official Journal.*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$"),  # standalone date line
]
SKIP_IF_CONTAINS = re.compile(r"\bANNEX\b", re.IGNORECASE)  # never remove lines with ANNEX

# footer strings to remove (we keep only the page number)
EXTRA_FOOTER_STRINGS = ["ELI: http", "ELI: https", "Official Journal", "OJ L"]

def find_page_number_rect(page: fitz.Page, band: fitz.Rect):
    """Return bbox of page number inside 'band', or None."""
    words = page.get_text("words")
    by_line = {}
    for x0, y0, x1, y1, w, block, line, wn in words:
        cy = (y0 + y1) / 2
        if band.y0 <= cy <= band.y1:
            by_line.setdefault((block, line), []).append((x0, y0, x1, y1, w))
    for _, items in by_line.items():
        items.sort(key=lambda t: t[0])
        text = "".join(t[4] for t in items).strip()
        if PN_RE.match(text):
            x0 = min(t[0] for t in items); y0 = min(t[1] for t in items)
            x1 = max(t[2] for t in items); y1 = max(t[3] for t in items)
            return fitz.Rect(x0, y0, x1, y1)
    return None

def find_first_whereas(doc: fitz.Document):
    """Return (page_index, bbox) of the first 'whereas' (case-insensitive), else (None, None)."""
    for i, page in enumerate(doc):
        for x0, y0, x1, y1, w, block, line, wn in page.get_text("words"):
            if re.sub(r"[:：]+$", "", w).lower() == "whereas":
                return i, fitz.Rect(x0, y0, x1, y1)
    return None, None

def redact_header_lines(page: fitz.Page, top_zone_pt: float):
    """Remove only header-looking lines in the top zone; keep any line containing 'ANNEX'."""
    r = page.rect
    top_band = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + top_zone_pt)

    # 1) Text lines in the top band
    words = page.get_text("words")  # x0,y0,x1,y1,text,block,line,word_no
    lines = {}
    for x0, y0, x1, y1, w, b, ln, wn in words:
        cy = (y0 + y1) / 2
        if top_band.y0 <= cy <= top_band.y1:
            lines.setdefault((b, ln), []).append((x0, y0, x1, y1, w))
    for _, items in lines.items():
        items.sort(key=lambda t: t[0])
        txt = "".join(t[4] for t in items).strip()
        if SKIP_IF_CONTAINS.search(txt):  # preserve ANNEX lines
            continue
        if any(rx.match(txt) for rx in HEADER_LINE_MATCHERS):
            x0 = min(t[0] for t in items); y0 = min(t[1] for t in items)
            x1 = max(t[2] for t in items); y1 = max(t[3] for t in items)
            page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(1, 1, 1))

    # 2) Horizontal rule near the top (thin, long line)
    try:
        drawings = page.get_drawings()
        page_width = r.width
        for d in drawings:
            for p in d.get("items", []):
                if p[0] == "l":  # line segment
                    x0, y0, x1, y1 = p[1]
                    if abs(y1 - y0) < 1.0 and r.y0 <= y0 <= r.y0 + top_zone_pt:
                        if abs(x1 - x0) > page_width * 0.5:
                            page.add_redact_annot(fitz.Rect(min(x0, x1), y0 - 2, max(x0, x1), y0 + 2), fill=(1, 1, 1))
    except Exception:
        pass  # some PDFs may not expose drawings

with fitz.open(INPUT_PDF) as doc:
    # A) Keep only content BELOW the first "Whereas" (remove the word too)
    start_page_idx, whereas_bbox = find_first_whereas(doc)
    if start_page_idx is not None:
        if start_page_idx > 0:
            doc.delete_pages(0, start_page_idx - 1)
        page = doc[0]
        r = page.rect
        cut_rect = fitz.Rect(r.x0, r.y0, r.x1, min(r.y1, whereas_bbox.y1 + PAD_BELOW_WHEREAS))
        if cut_rect.height > 0.5:
            page.add_redact_annot(cut_rect, fill=(1, 1, 1))
            page.apply_redactions()
    else:
        print("WARN: 'Whereas' not found. Keeping full document.")

    # B) Header cleanup (preserve ANNEX) + footer cleanup (keep page number)
    for page in doc:
        # Header: surgically remove header-looking lines / rule
        redact_header_lines(page, TOP_ZONE_PT)

        # Footer: clear everything except the page number
        r = page.rect
        footer_band = fitz.Rect(r.x0, r.y1 - FOOTER_PT, r.x1, r.y1)

        for term in EXTRA_FOOTER_STRINGS:
            for inst in page.search_for(term, clip=footer_band):
                page.add_redact_annot(inst, fill=(1, 1, 1))

        pn_rect = find_page_number_rect(page, footer_band)
        if pn_rect is None:
            page.add_redact_annot(footer_band, fill=(1, 1, 1))
        else:
            left  = fitz.Rect(footer_band.x0, footer_band.y0, pn_rect.x0, footer_band.y1)
            right = fitz.Rect(pn_rect.x1, footer_band.y0, footer_band.x1, footer_band.y1)
            if left.width > 0.5:
                page.add_redact_annot(left, fill=(1, 1, 1))
            if right.width > 0.5:
                page.add_redact_annot(right, fill=(1, 1, 1))
            pad = 2
            above = fitz.Rect(footer_band.x0, footer_band.y0, footer_band.x1, pn_rect.y0 - pad)
            below = fitz.Rect(footer_band.x0, pn_rect.y1 + pad, footer_band.x1, footer_band.y1)
            if above.height > 0.5:
                page.add_redact_annot(above, fill=(1, 1, 1))
            if below.height > 0.5:
                page.add_redact_annot(below, fill=(1, 1, 1))

        page.apply_redactions()

    doc.save(OUTPUT_PDF)

print(f"Saved cleaned PDF -> {OUTPUT_PDF}")
