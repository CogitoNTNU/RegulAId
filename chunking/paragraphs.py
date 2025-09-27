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
    """Start the document where the first paragraph bullet '- (' appears."""
    # Look for the first recital bullet point
    m = re.search(r'\n\s*-\s*\(\s*1\s*\)', raw)
    if m:
        return raw[m.start():].strip()
    
    # If not found, look for any recital pattern
    m = re.search(r'\n\s*-\s*\(\s*\d+\s*\)', raw)
    if m:
        return raw[m.start():].strip()
        
    # Fallback: start from "Whereas" if present
    m2 = re.search(r'\bWhereas\b[:]?', raw, flags=re.I)
    if m2:
        return raw[m2.start():].strip()
    
    # Final fallback: return the original text
    return raw

def remove_noise_lines(text: str) -> str:
    """Remove only obvious noise, be more conservative"""
    out = []
    lines = text.splitlines()
    
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            out.append("")
            continue
            
        # Remove HTML comments
        if s.startswith('<!--') and s.endswith('-->'):
            continue
            
        # Remove markdown headings that are just numbers or very short
        if re.match(r'^\s*#{1,6}\s*\d+\s*$', s):  # Only remove headings that are just numbers
            continue
            
        # Remove stand-alone footnotes like "( 1 ) OJ C 517, 22.12.2021, p. 56."
        if re.match(r'^\(\s*\d+\s*\)\s*OJ\s+[C|L]\s+\d+', s):
            continue
            
        # Remove ELI lines
        if 'ELI:' in s:
            continue
            
        # Keep everything else, including article headers!
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
    """Simplified article chunking that's more robust"""
    chunks: List[Dict] = []
    
    if not text or len(text.strip()) < 100:
        print("Warning: Text too short for article chunking")
        return chunks
    
    # Find all article positions
    article_pattern = re.compile(r'^\s*##\s*Article\s+(\d+)\s*$', re.M | re.I)
    article_matches = list(article_pattern.finditer(text))
    
    print(f"Found {len(article_matches)} article header matches")
    
    if not article_matches:
        # Try alternative pattern
        article_pattern2 = re.compile(r'^\s*Article\s+(\d+)\s*$', re.M | re.I)
        article_matches = list(article_pattern2.finditer(text))
        print(f"Found {len(article_matches)} article header matches with alternative pattern")
    
    # Also find chapters
    chapter_pattern = re.compile(r'^\s*##\s*CHAPTER\s+([IVXLC]+)\s*$', re.M | re.I)
    chapter_matches = list(chapter_pattern.finditer(text))
    
    current_chapter = None
    current_chapter_name = None
    
    # Process chapters first
    for chap_match in chapter_matches:
        chap_num = chap_match.group(1)
        # Skip Chapter XIII
        if chap_num == "XIII":
            continue
            
        # Try to get chapter name (next line after CHAPTER header)
        chap_start = chap_match.end()
        next_line_match = re.search(r'^\s*##\s*(.+?)\s*$', text[chap_start:], re.M)
        chap_name = next_line_match.group(1).strip() if next_line_match else f"Chapter {chap_num}"
        
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
    
    # Process articles
    for i, art_match in enumerate(article_matches):
        art_num = art_match.group(1)
        
        # Find the start of this article's content
        art_start = art_match.end()
        
        # Find the end (start of next article or end of text)
        art_end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)
        
        article_block = text[art_start:art_end]
        
        # Extract article name (usually the next line after "Article X")
        name_match = re.search(r'^\s*##\s*(.+?)\s*$', article_block, re.M)
        art_name = name_match.group(1).strip() if name_match else f"Article {art_num}"
        
        # The actual article content starts after the name line
        content_start = name_match.end() if name_match else 0
        art_body = article_block[content_start:].strip()
        
        # Clean up the body
        art_body = re.sub(r'\s+', ' ', art_body).strip()
        
        if art_body and len(art_body) > 10:
            # For now, treat each article as a single chunk
            chunks.append({
                "id": f"article-{art_num}",
                "text": f"{art_name}: {art_body}",
                "metadata": {
                    "paragraph_number": None,
                    "subparagraph": None,
                    "subsubparagraph": None,
                    "page": None,
                    "type": "article",
                    "chapter_number": current_chapter,
                    "chapter_name": current_chapter_name,
                    "section_number": None,
                    "section_name": None,
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


