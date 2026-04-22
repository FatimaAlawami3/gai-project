import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
OUTPUT_DIR = PROJECT_ROOT / "data"

PDF_FILES = [
    str(PDF_DIR / "Traffic Law.pdf"),
    str(PDF_DIR / "Theoretical Driving Handbook Trainee-Moroor.pdf"),
    str(PDF_DIR / "101 EN.pdf"),
]

OUTPUT_JSON = str(OUTPUT_DIR / "saudi_road_safety_kb.json")

# Chunk size control for long sections
MAX_CHARS_PER_CHUNK = 1800
MIN_CHARS_PER_CHUNK = 300


# =========================
# HELPERS
# =========================
def clean_text(text: str) -> str:
    """Clean extracted PDF text for better chunk quality."""
    if not text:
        return ""

    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\uf0b7", "•")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def read_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text page by page from a PDF."""
    doc = fitz.open(pdf_path)
    pages = []

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        pages.append({
            "page_num": i + 1,   # human-friendly page numbering
            "text": clean_text(text)
        })

    doc.close()
    return pages


def split_long_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Split long section text into smaller semantic chunks.
    Tries paragraphs first, then sentence grouping.
    """
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                # Paragraph still too big: split by sentence groups
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp = ""
                for sent in sentences:
                    if len(temp) + len(sent) + 1 <= max_chars:
                        temp = f"{temp} {sent}".strip()
                    else:
                        if temp:
                            chunks.append(temp)
                        temp = sent
                current = temp

    if current:
        chunks.append(current)

    # Avoid tiny chunks by merging small trailing chunks
    merged = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHARS_PER_CHUNK:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)

    return merged


def infer_keywords(text: str, section_title: str = "", top_n: int = 10) -> List[str]:
    """
    Very lightweight keyword extractor.
    You can later replace this with KeyBERT/spaCy if you want better keywords.
    """
    stopwords = {
        "the", "and", "for", "with", "that", "this", "are", "from", "shall", "must",
        "used", "into", "when", "their", "there", "have", "has", "been", "will",
        "road", "roads", "vehicle", "vehicles", "driver", "drivers", "traffic",
        "chapter", "article", "section", "unit", "page", "general", "including",
        "through", "which", "such", "than", "them", "they", "then", "also", "only",
        "under", "more", "than", "not", "all", "any", "may", "can", "these", "those"
    }

    combined = f"{section_title} {text}".lower()
    words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", combined)

    freq = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:top_n]]


def classify_document(file_name: str) -> str:
    file_name_lower = file_name.lower()
    if "traffic law" in file_name_lower:
        return "traffic_law"
    if "moroor" in file_name_lower or "handbook" in file_name_lower:
        return "moroor_handbook"
    if "101 en" in file_name_lower or "shc" in file_name_lower:
        return "shc_standard"
    return "generic"


def source_priority(doc_type: str) -> int:
    # Lower number = higher authority priority in retrieval logic
    mapping = {
        "traffic_law": 1,
        "moroor_handbook": 2,
        "shc_standard": 3,
        "generic": 4
    }
    return mapping.get(doc_type, 4)


# =========================
# SECTION DETECTORS
# =========================
def detect_traffic_law_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect sections in Traffic Law by Article headings.
    Example pattern: 'Article 36'
    """
    sections = []
    current = None

    article_pattern = re.compile(r"^\s*Article\s+(\d+)\b", re.IGNORECASE)

    for page in pages:
        page_num = page["page_num"]
        lines = page["text"].splitlines()

        for line in lines:
            if article_pattern.match(line):
                if current:
                    sections.append(current)

                art_num = article_pattern.match(line).group(1)
                current = {
                    "section_type": "article",
                    "section_number": art_num,
                    "section_title": f"Article {art_num}",
                    "page_start": page_num,
                    "page_end": page_num,
                    "text": line + "\n"
                }
            else:
                if current:
                    current["text"] += line + "\n"

        if current:
            current["page_end"] = page_num

    if current:
        sections.append(current)

    return postprocess_sections(sections)


def detect_moroor_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect Moroor structure using:
    - Unit headings
    - numbered topic headings
    """
    sections = []
    current_unit = "Unknown Unit"
    current = None

    unit_pattern = re.compile(r"^\s*Unit\s+\w+|^\s*Third uni|^\s*Fourth Unit|^\s*Fifth Unit|^\s*Unit Six", re.IGNORECASE)
    topic_pattern = re.compile(r"^\s*(\d+(\.\d+)*)\.\s+(.+)$")

    for page in pages:
        page_num = page["page_num"]
        lines = page["text"].splitlines()

        for line in lines:
            stripped = line.strip()

            if unit_pattern.match(stripped):
                current_unit = stripped
                continue

            topic_match = topic_pattern.match(stripped)
            if topic_match:
                if current:
                    sections.append(current)

                sec_num = topic_match.group(1)
                sec_title = topic_match.group(3).strip()

                current = {
                    "section_type": "topic",
                    "unit_or_chapter": current_unit,
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "page_start": page_num,
                    "page_end": page_num,
                    "text": stripped + "\n"
                }
            else:
                if current:
                    current["text"] += stripped + "\n"

        if current:
            current["page_end"] = page_num

    if current:
        sections.append(current)

    return postprocess_sections(sections)


def detect_shc_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect SHC sections using decimal numbering:
    1. Introduction
    1.1 Summary of Chapters
    2.1 Scope and Applicability
    3.2.21 SHC 602 – Manual on Uniform Traffic Control Devices
    """
    sections = []
    current = None
    current_chapter = None

    sec_pattern = re.compile(r"^\s*(\d+(\.\d+)+|\d+)\.?\s+(.+)$")

    for page in pages:
        page_num = page["page_num"]
        lines = page["text"].splitlines()

        for line in lines:
            stripped = line.strip()
            match = sec_pattern.match(stripped)

            if match:
                sec_num = match.group(1)
                sec_title = match.group(3).strip()

                # Ignore obvious table of contents rows
                if len(sec_title) < 2:
                    continue

                # chapter-level section tracking
                if re.fullmatch(r"\d+", sec_num):
                    current_chapter = f"{sec_num}. {sec_title}"

                if current:
                    sections.append(current)

                current = {
                    "section_type": "section",
                    "unit_or_chapter": current_chapter,
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "page_start": page_num,
                    "page_end": page_num,
                    "text": stripped + "\n"
                }
            else:
                if current:
                    current["text"] += stripped + "\n"

        if current:
            current["page_end"] = page_num

    if current:
        sections.append(current)

    return postprocess_sections(sections)


def detect_generic_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback: page-based chunking if structure isn't detected well."""
    sections = []
    for page in pages:
        sections.append({
            "section_type": "page_block",
            "section_number": str(page["page_num"]),
            "section_title": f"Page {page['page_num']}",
            "page_start": page["page_num"],
            "page_end": page["page_num"],
            "text": page["text"]
        })
    return postprocess_sections(sections)


def postprocess_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean up sections:
    - remove empty or extremely noisy blocks
    """
    cleaned = []
    for sec in sections:
        sec["text"] = clean_text(sec["text"])
        if len(sec["text"]) >= 40:
            cleaned.append(sec)
    return cleaned


# =========================
# CATEGORY MAPPING
# =========================
def infer_category(section_title: str, text: str, doc_type: str) -> str:
    combined = f"{section_title} {text}".lower()

    rules = [
        ("driving_license_rules", ["license", "driving license", "motorcycle license"]),
        ("traffic_violations_and_fines", ["violation", "fine", "penalty", "offense"]),
        ("traffic_points_system", ["points", "withdrawal of a driver", "license suspension"]),
        ("speed_limits_and_speed_safety", ["speed", "stopping distance", "braking distance", "reaction distance"]),
        ("traffic_accidents_and_legal_liability", ["accident", "liability", "injury", "death"]),
        ("road_signs", ["sign", "warning signs", "regulatory", "guide signs"]),
        ("road_markings_and_routing_elements", ["markings", "routing", "reflectors", "traffic routing elements"]),
        ("right_of_way_and_intersections", ["priority", "intersection", "roundabout", "crossings"]),
        ("pedestrian_rules", ["pedestrian", "sidewalk", "crosswalk"]),
        ("lane_changes_and_turning", ["lane", "changing lanes", "turning", "reversing"]),
        ("overtaking_rules", ["overtake", "overtaking"]),
        ("stopping_and_parking_rules", ["stopping", "parking", "waiting"]),
        ("vehicle_lighting_and_visibility", ["lights", "headlights", "visibility"]),
        ("driver_behavior_and_obligations", ["behavior", "obligations", "cooperative behavior"]),
        ("road_parts_and_road_types", ["road parts", "freeway", "service road", "lane", "roadway"]),
        ("traffic_law_definitions", ["definition", "definitions", "article 2"]),
        ("work_zone_signs", ["work zone", "temporary work area"]),
        ("highway_safety_concepts", ["road safety", "traffic engineering", "highway code"])
    ]

    for category, signals in rules:
        if any(s in combined for s in signals):
            return category

    if doc_type == "traffic_law":
        return "traffic_law_definitions"
    if doc_type == "moroor_handbook":
        return "driver_behavior_and_obligations"
    if doc_type == "shc_standard":
        return "highway_safety_concepts"

    return "general_road_safety"


# =========================
# BUILD KB
# =========================
def make_document_metadata(pdf_path: str) -> Dict[str, Any]:
    name = Path(pdf_path).name
    doc_type = classify_document(name)

    if doc_type == "traffic_law":
        return {
            "document_id": "DOC_TRAFFIC_LAW",
            "source_id": "SRC_TRAFFIC_LAW",
            "title": "Traffic Law",
            "document_type": "law",
            "authority": "Saudi Traffic Law / Royal Decree",
            "source_file": name,
            "source_priority": source_priority(doc_type)
        }

    if doc_type == "moroor_handbook":
        return {
            "document_id": "DOC_MOROOR",
            "source_id": "SRC_MOROOR_HANDBOOK",
            "title": "Theoretical Driving Handbook Trainee",
            "document_type": "handbook",
            "authority": "General Traffic Department (Moroor)",
            "source_file": name,
            "source_priority": source_priority(doc_type)
        }

    if doc_type == "shc_standard":
        return {
            "document_id": "DOC_SHC_101",
            "source_id": "SRC_SHC_101",
            "title": "Saudi Highway Code (SHC 101 – General)",
            "document_type": "standard",
            "authority": "Roads General Authority",
            "source_file": name,
            "source_priority": source_priority(doc_type)
        }

    return {
        "document_id": f"DOC_{slugify(Path(pdf_path).stem).upper()}",
        "source_id": f"SRC_{slugify(Path(pdf_path).stem).upper()}",
        "title": Path(pdf_path).stem,
        "document_type": "generic",
        "authority": "Unknown",
        "source_file": name,
        "source_priority": source_priority(doc_type)
    }


def detect_sections_by_type(doc_type: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if doc_type == "traffic_law":
        return detect_traffic_law_sections(pages)
    if doc_type == "moroor_handbook":
        return detect_moroor_sections(pages)
    if doc_type == "shc_standard":
        return detect_shc_sections(pages)
    return detect_generic_sections(pages)


def build_chunks(doc_meta: Dict[str, Any], sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    doc_type = classify_document(doc_meta["source_file"])

    for sec in sections:
        text_parts = split_long_text(sec["text"])

        for idx, part in enumerate(text_parts, start=1):
            category = infer_category(sec.get("section_title", ""), part, doc_type)
            keywords = infer_keywords(part, sec.get("section_title", ""))

            section_number = sec.get("section_number", "unknown")
            base_id = f"{doc_meta['document_id']}_{slugify(str(section_number))}_{idx:02d}"

            chunk = {
                "chunk_id": base_id,
                "document_id": doc_meta["document_id"],
                "source_id": doc_meta["source_id"],
                "document_title": doc_meta["title"],
                "document_type": doc_meta["document_type"],
                "authority": doc_meta["authority"],
                "source_file": doc_meta["source_file"],
                "source_priority": doc_meta["source_priority"],

                "section_type": sec.get("section_type"),
                "section_number": str(section_number),
                "section_title": sec.get("section_title"),
                "unit_or_chapter": sec.get("unit_or_chapter"),

                "page_start": sec.get("page_start"),
                "page_end": sec.get("page_end"),

                "category": category,
                "topic": slugify(sec.get("section_title", "")),
                "keywords": keywords,

                "content_type": infer_content_type(doc_type),
                "language": "en",
                "country": "Saudi Arabia",

                "text": part,
                "text_cleaned": clean_text(part),

                "citation": {
                    "source_file": doc_meta["source_file"],
                    "page_reference": (
                        f"p.{sec.get('page_start')}"
                        if sec.get("page_start") == sec.get("page_end")
                        else f"pp.{sec.get('page_start')}-{sec.get('page_end')}"
                    ),
                    "official_reference": build_official_reference(doc_type, sec)
                },

                "qa_hints": build_qa_hints(sec.get("section_title", ""), category),
                "retrieval_priority": retrieval_priority(doc_type, category)
            }

            chunks.append(chunk)

    return chunks


def infer_content_type(doc_type: str) -> str:
    if doc_type == "traffic_law":
        return "law"
    if doc_type == "moroor_handbook":
        return "guidance"
    if doc_type == "shc_standard":
        return "standard"
    return "reference"


def build_official_reference(doc_type: str, sec: Dict[str, Any]) -> str:
    if doc_type == "traffic_law":
        return f"Article {sec.get('section_number')}"
    if doc_type == "moroor_handbook":
        return f"{sec.get('unit_or_chapter', '')} - {sec.get('section_number', '')}"
    if doc_type == "shc_standard":
        return f"Section {sec.get('section_number')}"
    return sec.get("section_title", "")


def build_qa_hints(section_title: str, category: str) -> List[str]:
    return [
        f"What does {section_title} mean?",
        f"Explain {section_title} in simple words.",
        f"What are the rules about {category.replace('_', ' ')}?"
    ]


def retrieval_priority(doc_type: str, category: str) -> int:
    base = {
        "traffic_law": 10,
        "moroor_handbook": 8,
        "shc_standard": 7,
        "generic": 5
    }.get(doc_type, 5)

    high_value_categories = {
        "traffic_violations_and_fines",
        "traffic_points_system",
        "traffic_accidents_and_legal_liability",
        "driving_license_rules",
        "right_of_way_and_intersections",
        "road_signs",
        "speed_limits_and_speed_safety"
    }

    if category in high_value_categories:
        base += 1

    return min(base, 10)


def build_knowledge_base(pdf_files: List[str]) -> Dict[str, Any]:
    kb = {
        "metadata": {
            "knowledge_base_name": "saudi_road_safety_rag_kb",
            "version": "1.0",
            "embedding_ready": True,
            "chunking_strategy": "document-aware semantic section chunking",
            "created_for": "AI-powered road safety chatbot",
            "country": "Saudi Arabia",
            "language": "en"
        },
        "documents": [],
        "chunks": []
    }

    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"Skipping missing file: {pdf_path}")
            continue

        print(f"Processing: {pdf_path}")
        doc_meta = make_document_metadata(pdf_path)
        pages = read_pdf_pages(pdf_path)
        sections = detect_sections_by_type(
            classify_document(doc_meta["source_file"]),
            pages
        )
        chunks = build_chunks(doc_meta, sections)

        kb["documents"].append(doc_meta)
        kb["chunks"].extend(chunks)

        print(f"  -> Sections detected: {len(sections)}")
        print(f"  -> Chunks created: {len(chunks)}")

    return kb


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    kb = build_knowledge_base(PDF_FILES)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print(f"\nDone. JSON saved to:\n{OUTPUT_JSON}")
