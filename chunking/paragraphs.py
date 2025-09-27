import re
import json
from typing import List, Dict, Optional
import os

# 1. Get the absolute directory of the current script (paragraphs.py)
# '.../RegulAId/chunking'
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level (from 'chunking' to 'RegulAId')
parent_dir = os.path.dirname(script_dir)

# 3. Construct the full, absolute path to the target file
# This joins '.../RegulAId' with 'data/processed/AIACT.md'
file_path = os.path.join(parent_dir, 'data', 'processed', 'AIACT.md')

# ---------- 1) Cleaning helpers ----------
def strip_until_first_paragraph(raw: str) -> str:
    """Start the document where the first paragraph bullet '- (' appears.
       Fallback to 'Whereas' if no bullet found."""
    m = re.search(r'\n\s*-\s*\(', raw)
    if m:
        return raw[m.start():].strip()
    m2 = re.search(r'\bWhereas\b[:]?','' if raw is None else raw, flags=re.I)
    if m2:
        return raw[m2.start():].strip()
    return raw

def remove_noise_lines(text: str) -> str:
    """Remove markdown headings, HTML comment tags, standalone numbered footnotes,
       common header/footer tokens (ELI:, OJ, standalone dates), and short ALL-CAPS headers."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out.append("")  # preserve paragraph breaks
            continue
        # HTML comment
        if s.startswith('<!--') and s.endswith('-->'):
            continue
        # markdown headings
        if re.match(r'^\s*#{1,6}\s*', ln):
            continue
        # stand-alone footnotes like "( 1 ) ..." or "(1) ..."
        if re.match(r'^\(\s*\d+\s*\)\s*.*', s):
            continue
        # ELI / obvious OJ references or short dates
        if 'ELI:' in s or re.search(r'\bOJ\b', s) or re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', s):
            continue
        # page numbers like "12/144" or "x/144" as isolated or part-of-line
        if re.search(r'\b\d+\s*/\s*\d+\b', s) or re.search(r'^\(?x/\d+\)?$', s, re.I):
            continue
        # short almost-all-caps header line (heuristic)
        letters = re.sub(r'[^A-Za-z]+', '', s)
        if letters and len(s) < 70 and sum(1 for c in letters if c.isupper())/len(letters) > 0.6:
            continue
        out.append(ln)
    return "\n".join(out).strip()

# ---------- 2a) Chunking recitals (bullets) ----------
def chunk_recitals(text: str) -> List[Dict]:
    """Find bullets like '- (1) ...' and create per-paragraph chunks."""
    patt = re.compile(r'^\s*-\s*\((\d+)\)\s*(.*?)\s*(?=(?:^\s*-\s*\(\d+\)\s*)|\Z)',
                      re.S | re.M)
    chunks = []
    for m in patt.finditer(text):
        num = int(m.group(1))
        if num > 178:  # stop after recital 180
            break
        body = m.group(2).strip()
        body = re.sub(r'\s+', ' ', body)  # normalize whitespace
        meta = {
            "paragraph_number": num,
            "subparagraph": None,
            "subsubparagraph": None,
            "page": None,
            "type": "recital",
            "chapter_number": None,
            "chapter_name": None,
            "section_number": None,
            "section_name": None,
            "article_number": None,
            "article_name": None,
            "annex_number": None,
            "annex_name": None
        }
        chunks.append({
            "id": f"recital-{num}",
            "text": body,
            "metadata": meta
        })
    return chunks

# ---------- 2b) Chunking chapters & articles ----------
def chunk_chapters_and_articles(text: str) -> List[Dict]:
    """
    Chunk chapters, sections, and articles.
    - Walk the document sequentially and detect CHAPTER / SECTION / Article headers.
    - For each Article, extract the article body (from header end to next header).
    - If the article body contains numbered paragraphs (1., 2., ...), split into one chunk per numbered paragraph.
    - If the article body DOES NOT contain numbered paragraphs, treat the whole article body (including (a),(b),...) as a single paragraph chunk.
    """
    chunks: List[Dict] = []

    # Combined header regex (multiline) - FIXED for AI Act format
    header_re = re.compile(
        r'^\s*##\s*CHAPTER\s+(?P<chap_num>[IVXLC]+)\s*$\s*^\s*##\s*(?P<chap_name>.*)$|'  # CHAPTER header
        r'^\s*##\s*SECTION\s+(?P<sec_num>\d+)\s*$\s*^\s*##\s*(?P<sec_name>.*)$|'  # SECTION header  
        r'^\s*##\s*Article\s+(?P<art_num>\d+)\s*$\s*^\s*##\s*(?P<art_name>.*)$',  # ARTICLE header
        flags=re.M | re.I
    )

    # Also try a simpler pattern for articles that might be on single lines
    simple_article_re = re.compile(
        r'^\s*##\s*Article\s+(?P<art_num>\d+)\s*$\s*^\s*##\s*(?P<art_name>[^\n]+)$',
        flags=re.M | re.I
    )

    current_chapter = None
    current_chapter_name = None
    current_section = None
    current_section_name = None

    # First, let's find all the article headers using a simpler approach
    article_pattern = re.compile(
        r'^\s*##\s*Article\s+(\d+)\s*$.*?^\s*##\s*(.*?)\s*$',
        flags=re.M | re.I | re.DOTALL
    )
    
    # Also find chapter headers
    chapter_pattern = re.compile(
        r'^\s*##\s*CHAPTER\s+([IVXLC]+)\s*$.*?^\s*##\s*(.*?)\s*$',
        flags=re.M | re.I | re.DOTALL
    )
    
    # Section headers
    section_pattern = re.compile(
        r'^\s*##\s*SECTION\s+(\d+)\s*$.*?^\s*##\s*(.*?)\s*$', 
        flags=re.M | re.I | re.DOTALL
    )

    # Find chapters first
    chapter_matches = list(chapter_pattern.finditer(text))
    for chap_match in chapter_matches:
        chap_num = chap_match.group(1)
        chap_name = chap_match.group(2).strip()
        
        # Skip Chapter XIII completely
        if chap_num == "XIII":
            continue
            
        current_chapter = chap_num
        current_chapter_name = chap_name
        chunks.append({
            "id": f"chapter-{chap_num}",
            "text": chap_name,
            "metadata": {
                "paragraph_number": None,
                "subparagraph": None,
                "subsubparagraph": None,
                "page": None,
                "type": "chapter",
                "chapter_number": chap_num,
                "chapter_name": chap_name,
                "section_number": None,
                "section_name": None,
                "article_number": None,
                "article_name": None,
                "annex_number": None,
                "annex_name": None
            }
        })

    # Find sections
    section_matches = list(section_pattern.finditer(text))
    for sec_match in section_matches:
        sec_num = sec_match.group(1)
        sec_name = sec_match.group(2).strip()
        current_section = sec_num
        current_section_name = sec_name
        chunks.append({
            "id": f"section-{sec_num}",
            "text": sec_name,
            "metadata": {
                "paragraph_number": None,
                "subparagraph": None,
                "subsubparagraph": None,
                "page": None,
                "type": "section",
                "chapter_number": current_chapter,
                "chapter_name": current_chapter_name,
                "section_number": sec_num,
                "section_name": sec_name,
                "article_number": None,
                "article_name": None,
                "annex_number": None,
                "annex_name": None
            }
        })

    # Find articles - use a more robust pattern
    # Articles are typically formatted as:
    # ## Article  1
    # ## Subject matter
    article_header_pattern = re.compile(
        r'^\s*##\s*Article\s+(\d+)\s*$(?:\s*^\s*##\s*([^\n]+))?',
        flags=re.M | re.I
    )
    
    # Find all article starting positions
    article_starts = []
    for match in re.finditer(r'^\s*##\s*Article\s+(\d+)', text, flags=re.M | re.I):
        article_starts.append(match.start())
    
    # Add the end of text as the last boundary
    article_starts.append(len(text))
    
    # Process each article
    for i in range(len(article_starts) - 1):
        start_pos = article_starts[i]
        end_pos = article_starts[i + 1]
        article_block = text[start_pos:end_pos]
        
        # Extract article number and name
        art_match = re.search(r'^\s*##\s*Article\s+(\d+)\s*$', article_block, flags=re.M | re.I)
        if not art_match:
            continue
            
        art_num = art_match.group(1)
        
        # Try to find article name (usually on next line after Article header)
        name_match = re.search(r'^\s*##\s*Article\s+\d+\s*$\s*^\s*##\s*([^\n]+)', article_block, flags=re.M | re.I)
        art_name = name_match.group(1).strip() if name_match else f"Article {art_num}"
        
        # Extract article body (everything after the article name line)
        body_start = name_match.end() if name_match else art_match.end()
        art_body = article_block[body_start:].strip()
        
        # Skip if it's just "HAVE ADOPTED THIS REGULATION:" or similar transitional text
        if any(phrase in art_body.upper() for phrase in ["HAVE ADOPTED", "GENERAL PROVISIONS", "CHAPTER"]):
            continue
            
        # Clean up the article body
        art_body = re.sub(r'^\s*`', '', art_body)  # Remove leading backticks
        art_body = re.sub(r'\s+', ' ', art_body).strip()
        
        if not art_body or len(art_body) < 10:  # Skip very short articles
            continue
            
        # Check if article has numbered paragraphs
        para_patt = re.compile(r'(?m)^\s*(\d+)\.\s+(.*?)(?=(?:^\s*\d+\.\s)|\Z)', re.S)
        paras = list(para_patt.finditer(art_body))
        
        if paras:
            # Article has numbered paragraphs -> create chunk per numbered paragraph
            for pm in paras:
                para_num = pm.group(1)
                para_text = re.sub(r'\s+', ' ', pm.group(2)).strip()
                chunks.append({
                    "id": f"article-{art_num}-para-{para_num}",
                    "text": para_text,
                    "metadata": {
                        "paragraph_number": str(para_num),
                        "subparagraph": None,
                        "subsubparagraph": None,
                        "page": None,
                        "type": "article-paragraph",
                        "chapter_number": current_chapter,
                        "chapter_name": current_chapter_name,
                        "section_number": current_section,
                        "section_name": current_section_name,
                        "article_number": art_num,
                        "article_name": art_name,
                        "annex_number": None,
                        "annex_name": None
                    }
                })
        else:
            # No numbered paragraphs -> treat ENTIRE article body as single paragraph chunk
            text_to_use = art_body if art_body else art_name
            text_to_use = text_to_use.strip()
            
            if text_to_use:  # Only add if there's actual content
                chunks.append({
                    "id": f"article-{art_num}-para-1",
                    "text": text_to_use,
                    "metadata": {
                        "paragraph_number": "1",
                        "subparagraph": None,
                        "subsubparagraph": None,
                        "page": None,
                        "type": "article-paragraph",
                        "chapter_number": current_chapter,
                        "chapter_name": current_chapter_name,
                        "section_number": current_section,
                        "section_name": current_section_name,
                        "article_number": art_num,
                        "article_name": art_name,
                        "annex_number": None,
                        "annex_name": None
                    }
                })

    return chunks

# ---------- 3) Putting it together ----------
def strip_and_chunk(raw_text: str) -> List[Dict]:
    step1 = strip_until_first_paragraph(raw_text)
    step2 = remove_noise_lines(step1)

    # split the doc at the first CHAPTER header so recitals only operate on the pre-chapter part
    chap_first = re.search(r'(?m)^\s*CHAPTER\b', step2)
    if chap_first:
        recitals_text = step2[:chap_first.start()]
        rest_text = step2[chap_first.start():]
    else:
        recitals_text = step2
        rest_text = ""
    recitals = chunk_recitals(recitals_text)
    chapters_articles = chunk_chapters_and_articles(rest_text)

    return recitals + chapters_articles




# Now use the absolute path to open the file
with open(file_path, 'r', encoding='utf-8') as f: raw = f.read()
chunks = strip_and_chunk(raw)
print(json.dumps(chunks, indent=2, ensure_ascii=False))


