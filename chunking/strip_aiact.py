# split_ai_act.py
# pip install pymupdf
import sys
import re
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF

# ---------- paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PDF  = SCRIPT_DIR / "AI-act.pdf"
OUTPUT_PDF = SCRIPT_DIR / "AIACT_new.pdf"

if not INPUT_PDF.exists():
    print(f"ERROR: input PDF not found -> {INPUT_PDF}")
    sys.exit(1)

# ---------- settings ----------
TOP_ZONE_PT = 160          # only lines in this top band are considered header candidates
FOOTER_PT = 90             # footer band to clean while keeping page number
PAD_BELOW_WHEREAS = 4      # pad below "Whereas" when trimming

BOTTOM_RATIO = 0.45        # bottom % of the page to scan for small-font lines
SMALL_PT_MAX = 9.5         # remove spans <= this size (≈0.8em of 11–12pt text)

# page-number line like "123/144" (or "Page 123/144")
PN_RE = re.compile(r"^\s*(?:Page\s*)?\d+(?:\s*/\s*\d+)?\s*$", re.IGNORECASE)

# header-looking lines we want to remove (unless they contain ANNEX)
HEADER_LINE_MATCHERS = [
    re.compile(r"^\s*EN\s*$", re.IGNORECASE),
    re.compile(r"^\s*OJ\s*L\b.*\d{1,2}\.\d{1,2}\.\d{4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*Official Journal.*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$"),  # standalone date line
]
SKIP_IF_CONTAINS = re.compile(r"\bANNEX\b", re.IGNORECASE)  # never remove lines with ANNEX

# ---------- helpers ----------
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

def redact_small_font_footer_lines_absolute(
    page: fitz.Page,
    pn_rect: Optional[fitz.Rect],
    small_pt_max: float = SMALL_PT_MAX,
    bottom_ratio: float = BOTTOM_RATIO,
    vertical_gap_threshold: float = 25.0,  # max distance to next line to keep paragraph number
):
    """
    Remove footer reference numbers like (7), (⁸), (¹¹) that stand alone,
    but keep paragraph numbers like (10) that start a section (even if on their own line).
    """
    r = page.rect
    y_thresh = r.y1 - (r.height * bottom_ratio)
    d = page.get_text("dict")

    lines = [ln for b in d.get("blocks", []) if b.get("type", 0) == 0 for ln in b.get("lines", [])]
    line_count = len(lines)

    for i, ln in enumerate(lines):
        spans = ln.get("spans", [])
        if not spans:
            continue

        lb = ln.get("bbox", [0, 0, 0, 0])
        cy = (lb[1] + lb[3]) / 2.0
        line_rect = fitz.Rect(*lb)
        full_text = "".join(sp.get("text", "") for sp in spans).strip()

        # Skip page number area
        if pn_rect is not None and line_rect.intersects(pn_rect):
            continue

        # --- 1) Detect if line is a "(number)" ---
        if re.fullmatch(r"\(\s*[⁰¹²³⁴⁵⁶⁷⁸⁹0-9]{1,3}\s*\)", full_text):
            # Get next line's bbox if available
            next_ln = lines[i + 1] if i + 1 < line_count else None
            next_y0 = next_ln["bbox"][1] if next_ln else None

            # Check vertical distance to next line
            if next_y0 is not None:
                vertical_gap = next_y0 - lb[3]
            else:
                vertical_gap = float("inf")

            # If this line is *close* to the next line, assume paragraph number → keep it
            if vertical_gap < vertical_gap_threshold:
                continue  # keep

            # If far away and near the bottom → remove it (reference)
            if cy > y_thresh:
                page.add_redact_annot(line_rect, fill=(1, 1, 1))
                continue

        # --- 2) Remove other small-font footer text ---
        max_size = max(float(sp.get("size", 0.0)) for sp in spans)
        if max_size <= small_pt_max and cy > y_thresh:
            page.add_redact_annot(line_rect, fill=(1, 1, 1))

# ---------- main ----------
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
        pn_rect = find_page_number_rect(page, footer_band)

        # Remove small-font footnotes/sources but keep paragraph numbers
        redact_small_font_footer_lines_absolute(
            page, pn_rect=pn_rect, small_pt_max=SMALL_PT_MAX, bottom_ratio=BOTTOM_RATIO
        )

        # Keep only the page number
        if pn_rect is None:
            page.add_redact_annot(footer_band, fill=(1, 1, 1))
        else:
            left  = fitz.Rect(footer_band.x0, footer_band.y0, pn_rect.x0, footer_band.y1)
            right = fitz.Rect(pn_rect.x1, footer_band.y0, footer_band.x1, footer_band.y1)
            pad = 2
            above = fitz.Rect(footer_band.x0, footer_band.y0, footer_band.x1, pn_rect.y0 - pad)
            below = fitz.Rect(footer_band.x0, pn_rect.y1 + pad, footer_band.x1, footer_band.y1)
            for rect in [left, right, above, below]:
                if rect.get_area() > 0.5:
                    page.add_redact_annot(rect, fill=(1, 1, 1))

        page.apply_redactions()

    doc.save(OUTPUT_PDF)

print(f" Saved cleaned PDF -> {OUTPUT_PDF}")