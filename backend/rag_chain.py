import json
from pathlib import Path
from typing import Any
import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from intent_router import detect_intent, generic_answer, road_topics, suggested_questions
from prompts import ARABIC_RE, RAG_PROMPT, get_prompt_inputs
from rag_config import Settings, load_google_credentials


def _scalar_metadata(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def load_kb_documents(kb_path: Path) -> list[Document]:
    with kb_path.open("r", encoding="utf-8") as file:
        kb = json.load(file)

    documents: list[Document] = []
    for chunk in kb.get("chunks", []):
        citation = chunk.get("citation") or {}
        metadata_keys = [
            "chunk_id",
            "document_id",
            "document_title",
            "document_type",
            "authority",
            "source_file",
            "source_priority",
            "section_type",
            "section_number",
            "section_title",
            "unit_or_chapter",
            "page_start",
            "page_end",
            "category",
            "topic",
            "retrieval_priority",
        ]
        metadata = {
            key: _scalar_metadata(chunk.get(key))
            for key in metadata_keys
            if key in chunk
        }
        metadata.update(
            {
                "citation_source_file": citation.get("source_file"),
                "citation_page_reference": citation.get("page_reference"),
                "official_reference": citation.get("official_reference"),
            }
        )

        text = chunk.get("text_cleaned") or chunk.get("text") or ""
        if text.strip():
            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def create_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    credentials = load_google_credentials(settings) if settings.use_vertexai else None
    kwargs: dict[str, Any] = {
        "model": settings.embedding_model,
        "output_dimensionality": settings.embedding_dimensions,
    }

    if settings.use_vertexai:
        kwargs.update(
            {
                "vertexai": True,
                "project": settings.google_cloud_project,
                "location": settings.google_cloud_location,
                "credentials": credentials,
            }
        )
    else:
        if not settings.api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY in .env, or configure Vertex AI service-account credentials."
            )
        kwargs["api_key"] = settings.api_key

    return GoogleGenerativeAIEmbeddings(**kwargs)


def create_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    credentials = load_google_credentials(settings) if settings.use_vertexai else None
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "thinking_level": settings.thinking_level,
    }

    if settings.use_vertexai:
        kwargs.update(
            {
                "vertexai": True,
                "project": settings.google_cloud_project,
                "location": settings.google_cloud_location,
                "credentials": credentials,
            }
        )
    else:
        if not settings.api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY in .env, or configure Vertex AI service-account credentials."
            )
        kwargs["api_key"] = settings.api_key

    return ChatGoogleGenerativeAI(**kwargs)


def build_vector_store(settings: Settings) -> FAISS:
    documents = load_kb_documents(settings.knowledge_base_path)
    if not documents:
        raise RuntimeError(f"No chunks were found in {settings.knowledge_base_path}")

    embeddings = create_embeddings(settings)
    vector_store = FAISS.from_documents(documents, embeddings)
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(settings.vector_store_path))
    return vector_store


def load_vector_store(settings: Settings) -> FAISS:
    embeddings = create_embeddings(settings)
    index_file = settings.vector_store_path / "index.faiss"
    pkl_file = settings.vector_store_path / "index.pkl"
    if not index_file.exists() or not pkl_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at {settings.vector_store_path}. "
            "Run: python build_vector_store.py"
        )

    return FAISS.load_local(
        str(settings.vector_store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def format_context(docs_with_scores: list[tuple[Document, float]]) -> str:
    blocks = []
    for rank, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata
        citation = build_citation(meta)
        arabic_citation = build_arabic_citation(meta)
        blocks.append(
            "\n".join(
                [
                    f"[Context {rank}]",
                    f"chunk_id: {meta.get('chunk_id')}",
                    f"title: {meta.get('document_title')}",
                    f"section: {meta.get('section_title')}",
                    f"category: {meta.get('category')}",
                    f"citation: {citation}",
                    f"arabic_citation: {arabic_citation}",
                    f"retrieval_distance: {score:.4f}",
                    "text:",
                    doc.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_citation(meta: dict[str, Any]) -> str:
    return ", ".join(
        part
        for part in [
            str(meta.get("source_file") or ""),
            str(meta.get("citation_page_reference") or ""),
            str(meta.get("official_reference") or ""),
        ]
        if part and part != "None"
    )


def build_arabic_citation(meta: dict[str, Any]) -> str:
    parts = []
    if meta.get("source_file"):
        parts.append(f"الملف: {meta.get('source_file')}")
    if meta.get("citation_page_reference"):
        parts.append(f"الصفحات: {meta.get('citation_page_reference')}")
    if meta.get("official_reference"):
        parts.append(f"المرجع: {meta.get('official_reference')}")
    return "، ".join(parts)


def format_source(doc: Document, score: float, language: str) -> dict[str, Any]:
    meta = doc.metadata
    citation = build_citation(meta)
    return {
        "chunk_id": meta.get("chunk_id"),
        "document_title": meta.get("document_title"),
        "section_title": meta.get("section_title"),
        "category": meta.get("category"),
        "source_file": meta.get("source_file"),
        "page_reference": meta.get("citation_page_reference"),
        "official_reference": meta.get("official_reference"),
        "citation": citation,
        "display_citation": build_arabic_citation(meta)
        if language == "ar"
        else citation,
        "retrieval_distance": score,
        "preview": doc.page_content[:320],
    }


def source_key(source: dict[str, Any]) -> str:
    return "|".join(
        normalize_text(str(source.get(key) or "")).lower()
        for key in [
            "display_citation",
            "citation",
            "section_title",
            "source_file",
            "page_reference",
            "official_reference",
        ]
    )


def dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for source in sources:
        key = source_key(source)
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _answer_mentions_source(answer: str, source: dict[str, Any]) -> bool:
    normalized_answer = normalize_text(answer).lower()
    citation = normalize_text(str(source.get("citation") or "")).lower()
    display_citation = normalize_text(str(source.get("display_citation") or "")).lower()
    source_file = normalize_text(str(source.get("source_file") or "")).lower()
    page_reference = normalize_text(str(source.get("page_reference") or "")).lower()
    official_reference = normalize_text(str(source.get("official_reference") or "")).lower()

    if citation and citation in normalized_answer:
        return True
    if display_citation and display_citation in normalized_answer:
        return True

    file_is_present = bool(source_file and source_file in normalized_answer)
    detail_is_present = bool(
        (page_reference and page_reference in normalized_answer)
        or (official_reference and official_reference in normalized_answer)
    )
    has_detail = bool(page_reference or official_reference)
    return file_is_present and (detail_is_present or not has_detail)


def select_answer_sources(
    docs_with_scores: list[tuple[Document, float]],
    answer: str,
    language: str,
    question: str = "",
) -> list[dict[str, Any]]:
    seen = set()
    source_items = []
    for doc, score in docs_with_scores:
        source = format_source(doc, float(score), language)
        key = source_key(source)
        if key in seen:
            continue
        seen.add(key)
        source_items.append((source, doc, float(score)))

    sources = [source for source, _, _ in source_items]
    mentioned_sources = [
        source for source in sources if _answer_mentions_source(answer, source)
    ]

    question_terms = _expanded_source_terms(question)
    answer_terms = _expanded_source_terms(answer)
    ranked = []
    for source, doc, score in source_items:
        doc_text = _doc_combined_text(doc)
        question_overlap = sum(1 for term in question_terms if term in doc_text)
        answer_overlap = sum(1 for term in answer_terms if term in doc_text)
        ranked.append((question_overlap, answer_overlap, score, source))

    if question_terms:
        relevant_sources = [
            source
            for question_overlap, answer_overlap, _, source in sorted(
                ranked, key=lambda item: (-item[0], -item[1], item[2])
            )
            if question_overlap > 0 and answer_overlap > 0
        ]
    else:
        max_overlap = max((overlap for _, overlap, _, _ in ranked), default=0)
        minimum_overlap = 1 if max_overlap < 4 else max(2, round(max_overlap * 0.4))
        relevant_sources = [
            source
            for _, answer_overlap, _, source in sorted(
                ranked, key=lambda item: (-item[1], item[2])
            )
            if answer_overlap >= minimum_overlap
        ]

    combined_sources = []
    seen_combined = set()
    for source in [*mentioned_sources, *relevant_sources]:
        key = source_key(source)
        if key in seen_combined:
            continue
        if question_terms:
            source_doc = next(
                (
                    doc
                    for item_source, doc, _ in source_items
                    if source_key(item_source) == key
                ),
                None,
            )
            if source_doc is not None:
                doc_text = _doc_combined_text(source_doc)
                if not any(term in doc_text for term in question_terms):
                    continue
        seen_combined.add(key)
        combined_sources.append(source)

    if combined_sources:
        return combined_sources[:3]
    return sources[:1]


def message_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()
    return str(content)


def stream_chunk_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


SOURCE_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:[*_`#>\-\s]*)?"
    r"(?:sources|references|المصادر|المراجع)"
    r"(?:[*_`\s]*)?:?\s*(?:\n|$)",
    flags=re.IGNORECASE,
)

SOURCE_NOTE_RE = re.compile(
    r"(?:^|\n)\s*(?:This answer is based on the uploaded Saudi traffic sources\.?|"
    r"هذه الإجابة مبنية على مصادر المرور السعودية المرفوعة\.?|"
    r"تعتمد هذه الإجابة على مصادر المرور السعودية المرفوعة\.?)\s*",
    flags=re.IGNORECASE,
)

ARABIC_SECTION_LABEL_REPLACEMENTS = [
    (r"(?mi)^\s*Key Points\s*:\s*$", "النقاط الرئيسية:"),
    (r"(?mi)^\s*Penalties\s*:\s*$", "العقوبات:"),
    (r"(?mi)^\s*Legal Responsibility\s*:\s*$", "المسؤولية القانونية:"),
    (r"(?mi)^\s*Exceptions\s*:\s*$", "الاستثناءات:"),
    (r"(?mi)^\s*Steps\s*:\s*$", "الخطوات:"),
]


def strip_generated_sources(answer: str) -> str:
    match = SOURCE_SECTION_RE.search(answer)
    if not match:
        return answer.strip()
    return answer[: match.start()].strip()


def clean_answer_formatting(answer: str) -> str:
    text = answer.replace("\r\n", "\n")
    text = SOURCE_NOTE_RE.sub("\n", text)
    text = re.sub(r"[ \t]+\*\s+(?=(?:\*\*)|[A-Za-z0-9\u0600-\u06FF])", "\n- ", text)
    text = re.sub(r"(?m)^\s*\*\s+", "- ", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    if ARABIC_RE.search(text):
        for pattern, replacement in ARABIC_SECTION_LABEL_REPLACEMENTS:
            text = re.sub(pattern, replacement, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_chat_history(chat_history: list[dict[str, str]] | None) -> str:
    if not chat_history:
        return "No recent conversation."

    lines = []
    for message in chat_history[-6:]:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if len(content) > 600:
            content = content[:600].rstrip() + "..."
        role_label = "User" if role == "user" else "Assistant"
        lines.append(f"{role_label}: {content}")

    return "\n".join(lines) if lines else "No recent conversation."


FOLLOWUP_TOPIC_KEYWORDS = {
    "roundabout": [
        r"\broundabout\b",
        r"\btraffic circle\b",
        r"دوار",
    ],
    "parking": [
        r"\bparking\b",
        r"\bstopping\b",
        r"\bwaiting\b",
        r"وقوف",
        r"مواقف",
        r"انتظار",
    ],
    "accident": [
        r"\baccident\b",
        r"\bcollision\b",
        r"\bcrash\b",
        r"حادث",
        r"حوادث",
        r"تصادم",
        r"صدم",
    ],
    "pedestrian crossing": [
        r"\bpedestrian\b",
        r"\bcrosswalk\b",
        r"مشا",
        r"عبور",
    ],
    "lane changing": [
        r"\blane\b",
        r"\bchanging lanes\b",
        r"مسار",
    ],
    "unlicensed driver": [
        r"\bunlicensed\b",
        r"\bwithout (?:a )?driving licen[sc]e\b",
        r"\bnot holding (?:a )?driving licen[sc]e\b",
        r"\blicen[sc]e\b.*\bdrive\b",
        r"\bdrive\b.*\blicen[sc]e\b",
        r"رخصة",
        r"بدون رخصة",
        r"غير مرخص",
    ],
    "phone use": [
        r"\bphone\b",
        r"\bmobile\b",
        r"\bcell\s*phone\b",
        r"\bhands[- ]free\b",
        r"\bdevice\b",
        r"\bdistract(ed|ion)?\b",
        r"هاتف",
        r"جوال",
        r"الهاتف",
        r"الجوال",
        r"بدون يد",
        r"بدون استخدام اليد",
        r"تشتيت",
        r"انشغال",
    ],
}


FOLLOWUP_ASPECT_KEYWORDS = {
    "priority right-of-way": [
        r"\bpriority\b",
        r"\bright[- ]of[- ]way\b",
        r"\byield\b",
        r"\bgive way\b",
        r"أولوية",
        r"اولوي",
    ],
    "signals indicators": [
        r"\bsignal\b",
        r"\bindicator\b",
        r"إشارة",
        r"اشارة",
    ],
    "speed safe distance": [
        r"\bspeed\b",
        r"\bdistance\b",
        r"سرعة",
        r"مسافة",
    ],
    "risk danger safety": [
        r"\bdanger\b",
        r"\bdangerous\b",
        r"\brisk\b",
        r"\brisky\b",
        r"\bunsafe\b",
        r"\bsafety\b",
        r"\bhigh[- ]risk\b",
        r"خطر",
        r"خطير",
        r"مخاطر",
        r"سلامة",
        r"غير آمن",
    ],
    "penalty violation fine": [
        r"\bpenalty\b",
        r"\bviolation\b",
        r"\bfine\b",
        r"\bconsequence\b",
        r"\bconsequences\b",
        r"\bwhat happens\b",
        r"\bresult\b",
        r"مخالفة",
        r"غرامة",
        r"عقوبة",
        r"ماذا يحدث",
        r"ما النتيجة",
    ],
    "damage only": [
        r"\bdamage\b",
        r"\bdamaged\b",
        r"\bvehicle damage\b",
        r"\bonly vehicles\b",
        r"\bonly damage\b",
        r"\bnajm\b",
        r"أضرار",
        r"اضرار",
        r"تلف",
        r"نجم",
    ],
    "injuries emergency": [
        r"\binjur(y|ies)\b",
        r"\binjured\b",
        r"\bambulance\b",
        r"\bred crescent\b",
        r"إصابة",
        r"اصابة",
        r"إصابات",
        r"اصابات",
        r"الهلال الأحمر",
    ],
    "fault liability responsibility": [
        r"\bat fault\b",
        r"\bmy fault\b",
        r"\bthe fault\b",
        r"\bfault\b",
        r"\bliable\b",
        r"\bliability\b",
        r"\bresponsible\b",
        r"\bresponsibility\b",
        r"\bin the wrong\b",
        r"\bwas the wrong\b",
        r"\bif i was wrong\b",
        r"\bif i were wrong\b",
        r"المخطئ",
        r"المتسبب",
        r"المتسببة",
        r"تسببت",
        r"متسبب",
        r"غلطان",
        r"غلطانة",
        r"غلط",
        r"أنا السبب",
        r"انا السبب",
        r"كنت السبب",
        r"إذا كنت السبب",
        r"اذا كنت السبب",
        r"لو أنا السبب",
        r"لو انا السبب",
        r"أنا المتسبب",
        r"انا المتسبب",
        r"إذا أنا المتسبب",
        r"اذا انا المتسبب",
        r"لو أنا المتسبب",
        r"لو انا المتسبب",
        r"على خطأ",
        r"الخطأ علي",
        r"الخطأ عليّ",
        r"الخطأ مني",
        r"الغلطة مني",
        r"الحق علي",
        r"الحق عليّ",
        r"مسؤول",
        r"المسؤولية",
    ],
}


SOURCE_ROUTE_BY_INTENT = {
    "definition": "legal_definition",
    "permission_rule": "law_rule",
    "penalty_consequence": "law_penalty",
    "procedure": "handbook_procedure",
    "comparison": "balanced_comparison",
    "followup": "contextual_followup",
    "clarification": "clarification",
    "general_road_safety": "balanced",
}


RELATED_SUGGESTIONS = {
    "en": {
        "definition": [
            "What rule applies to this term?",
            "Is there a penalty related to this definition?",
            "Can you give an example from driving?",
        ],
        "permission_rule": [
            "What permission is required?",
            "What happens if this rule is violated?",
            "Who is responsible in this case?",
        ],
        "penalty_consequence": [
            "Who is legally responsible in this case?",
            "Does the penalty change if the violation is repeated?",
            "Is there a related safety procedure?",
        ],
        "procedure": [
            "What if there are injuries?",
            "What if only vehicles are damaged?",
            "What should I avoid doing?",
        ],
        "comparison": [
            "Which rule should I follow in practice?",
            "What is the legal difference?",
            "Which source explains this best?",
        ],
        "followup": [
            "What is the penalty for this?",
            "What should I do next?",
            "Who has priority or responsibility?",
        ],
        "clarification": [
            "What if there are injuries?",
            "What if only vehicle damage occurs?",
            "Can the vehicle be moved after an accident?",
        ],
        "general_road_safety": [
            "What is the related penalty?",
            "What should the driver do step by step?",
            "Which source covers this rule?",
        ],
    },
    "ar": {
        "definition": [
            "ما القاعدة المرتبطة بهذا المصطلح؟",
            "هل توجد عقوبة مرتبطة بهذا التعريف؟",
            "هل يمكنك إعطاء مثال من القيادة؟",
        ],
        "permission_rule": [
            "ما التصريح المطلوب؟",
            "ماذا يحدث عند مخالفة هذه القاعدة؟",
            "من المسؤول في هذه الحالة؟",
        ],
        "penalty_consequence": [
            "من المسؤول قانونياً في هذه الحالة؟",
            "هل تختلف العقوبة عند تكرار المخالفة؟",
            "هل توجد إجراءات سلامة مرتبطة؟",
        ],
        "procedure": [
            "ماذا لو كانت هناك إصابات؟",
            "ماذا لو كان الضرر في المركبات فقط؟",
            "ما الذي يجب تجنبه؟",
        ],
        "comparison": [
            "أي قاعدة يجب اتباعها عملياً؟",
            "ما الفرق القانوني؟",
            "أي مصدر يوضح ذلك؟",
        ],
        "followup": [
            "ما العقوبة في هذه الحالة؟",
            "ما الخطوة التالية؟",
            "من له الأولوية أو المسؤولية؟",
        ],
        "clarification": [
            "ماذا لو كانت هناك إصابات؟",
            "ماذا لو كان الضرر في المركبات فقط؟",
            "هل يمكن تحريك المركبة بعد الحادث؟",
        ],
        "general_road_safety": [
            "ما العقوبة المرتبطة بذلك؟",
            "ما الخطوات التي يجب على السائق اتباعها؟",
            "أي مصدر يغطي هذه القاعدة؟",
        ],
    },
}


TOPIC_SUGGESTIONS = {
    "en": {
        "accident": [
            "What should the driver do if the accident causes injuries?",
            "When should Najm be contacted for the accident?",
            "Can the vehicles be moved before the accident is reported?",
        ],
        "roundabout": [
            "Who has priority when entering the roundabout?",
            "Which lane should the driver choose in the roundabout?",
            "When should the driver signal before exiting the roundabout?",
        ],
        "parking": [
            "Where is parking prohibited?",
            "When is stopping allowed instead of parking?",
            "What is the penalty for illegal parking?",
        ],
        "vehicle_color": [
            "Is prior permission required before changing a vehicle's color?",
            "What is the penalty for changing a vehicle's color without approval?",
            "Does the rule apply to the vehicle owner, the repair shop, or both?",
        ],
        "phone_use": [
            "Is using a phone while driving allowed?",
            "When is a hands-free device required while driving?",
            "What is the consequence of using a phone while driving?",
        ],
        "unlicensed_driver": [
            "What is the penalty for allowing an unlicensed person to drive?",
            "Who is legally responsible if an unlicensed driver causes an accident?",
            "Does the rule apply to the owner, the designated driver, or both?",
        ],
        "speed": [
            "What should the driver do when road or weather conditions reduce visibility?",
            "How should the driver adjust speed near hazards or intersections?",
            "What is the penalty for speeding?",
        ],
        "road_signs": [
            "What must the driver do at a stop sign?",
            "What is the difference between a warning sign and a regulatory sign?",
            "What is the penalty for ignoring a traffic signal?",
        ],
        "lane": [
            "When is changing lanes allowed?",
            "When is overtaking prohibited?",
            "Who has priority when two lanes merge?",
        ],
        "pedestrian": [
            "When must the driver give way to pedestrians?",
            "What should the driver do near a pedestrian crossing?",
            "What is the penalty for failing to give way to pedestrians?",
        ],
    },
    "ar": {
        "accident": [
            "ماذا يجب على السائق فعله إذا نتجت إصابات عن الحادث؟",
            "متى يجب التواصل مع نجم بخصوص الحادث؟",
            "هل يمكن تحريك المركبات قبل الإبلاغ عن الحادث؟",
        ],
        "roundabout": [
            "من له الأولوية عند دخول الدوار؟",
            "أي مسار يجب على السائق اختياره في الدوار؟",
            "متى يجب على السائق استخدام الإشارة قبل الخروج من الدوار؟",
        ],
        "parking": [
            "أين يمنع الوقوف؟",
            "متى يسمح بالتوقف بدلاً من الوقوف؟",
            "ما عقوبة الوقوف المخالف؟",
        ],
        "vehicle_color": [
            "هل يلزم الحصول على موافقة قبل تغيير لون المركبة؟",
            "ما عقوبة تغيير لون المركبة بدون موافقة؟",
            "هل تطبق القاعدة على مالك المركبة أم على الورشة أم على الاثنين؟",
        ],
        "phone_use": [
            "هل يسمح باستخدام الهاتف أثناء القيادة؟",
            "متى يشترط استخدام الهاتف عبر نظام بدون استخدام اليد؟",
            "ما نتيجة استخدام الهاتف أثناء القيادة؟",
        ],
        "unlicensed_driver": [
            "ما عقوبة السماح لشخص غير مرخص له بقيادة المركبة؟",
            "من المسؤول قانونياً إذا تسبب سائق غير مرخص له في حادث؟",
            "هل تطبق القاعدة على المالك أم السائق المعين أم على الاثنين؟",
        ],
        "speed": [
            "ماذا يجب على السائق فعله عندما تقل الرؤية بسبب الطريق أو الطقس؟",
            "كيف يجب على السائق تعديل السرعة قرب المخاطر أو التقاطعات؟",
            "ما عقوبة تجاوز السرعة؟",
        ],
        "road_signs": [
            "ماذا يجب على السائق فعله عند علامة قف؟",
            "ما الفرق بين العلامة التحذيرية والعلامة التنظيمية؟",
            "ما عقوبة تجاهل الإشارة المرورية؟",
        ],
        "lane": [
            "متى يسمح بتغيير المسار؟",
            "متى يمنع التجاوز؟",
            "من له الأولوية عند اندماج مسارين؟",
        ],
        "pedestrian": [
            "متى يجب على السائق إعطاء الأولوية للمشاة؟",
            "ماذا يجب على السائق فعله قرب ممر المشاة؟",
            "ما عقوبة عدم إعطاء الأولوية للمشاة؟",
        ],
    },
}


CLARIFYING_SUGGESTIONS = {
    "en": {
        "procedure": [
            "Do you mean step-by-step instructions after a traffic accident?",
            "Do you mean step-by-step instructions when approaching a roundabout?",
            "Do you mean step-by-step instructions for parking or stopping?",
        ],
        "penalty_consequence": [
            "Do you mean the penalty for not reporting an accident?",
            "Do you mean the penalty for a roundabout violation?",
            "Do you mean the penalty for allowing an unlicensed person to drive?",
        ],
        "permission_rule": [
            "Do you mean whether a roundabout action is allowed?",
            "Do you mean whether parking is allowed in a certain place?",
            "Do you mean whether changing a vehicle's color is allowed?",
            "Do you mean whether using a phone while driving is allowed?",
        ],
        "clarification": [
            "Do you mean a roundabout rule, an accident procedure, or a parking rule?",
            "Do you mean a penalty, a legal rule, or step-by-step instructions?",
            "Do you want the rule for accidents, roundabouts, or parking?",
        ],
    },
    "ar": {
        "procedure": [
            "هل تقصد خطوات ما بعد وقوع حادث مروري؟",
            "هل تقصد خطوات القيادة عند الاقتراب من الدوار؟",
            "هل تقصد خطوات الوقوف أو التوقف؟",
        ],
        "penalty_consequence": [
            "هل تقصد عقوبة عدم الإبلاغ عن الحادث؟",
            "هل تقصد عقوبة مخالفة في الدوار؟",
            "هل تقصد عقوبة السماح لشخص غير مرخص له بالقيادة؟",
        ],
        "permission_rule": [
            "هل تقصد ما إذا كان إجراء معين في الدوار مسموحاً؟",
            "هل تقصد ما إذا كان الوقوف مسموحاً في مكان معين؟",
            "هل تقصد ما إذا كان تغيير لون المركبة مسموحاً؟",
            "هل تقصد ما إذا كان استخدام الهاتف أثناء القيادة مسموحاً؟",
        ],
        "clarification": [
            "هل تقصد قاعدة تخص الدوار أم إجراء بعد الحادث أم قاعدة للوقوف؟",
            "هل تقصد عقوبة أم قاعدة نظامية أم خطوات عملية؟",
            "هل تريد الحكم المتعلق بالحوادث أم الدوارات أم الوقوف؟",
        ],
    },
}


def _primary_topic_for_question(
    question: str,
    language: str,
    followup_topic: str | None = None,
) -> str | None:
    if followup_topic:
        return followup_topic

    topics = road_topics(question, language)
    if not topics:
        return None

    preferred_order = [
        "accident",
        "roundabout",
        "parking",
        "vehicle_color",
        "driving_license",
        "phone_use",
        "unlicensed_driver",
        "speed",
        "road_signs",
        "lane",
        "pedestrian",
    ]
    for topic in preferred_order:
        if topic in topics:
            return topic
    return sorted(topics)[0]


def smart_suggested_questions(
    language: str,
    answer_intent: str,
    question: str = "",
    followup_topic: str | None = None,
    needs_clarification: bool = False,
) -> list[str]:
    return suggested_questions(language)


def build_clarification_answer(
    language: str,
    answer_intent: str | None = None,
    followup_topic: str | None = None,
    followup_aspect: str | None = None,
) -> str:
    if language == "ar":
        if answer_intent == "procedure" and not followup_topic:
            return (
                "ما الحالة التي تريد الخطوات لها بالضبط؟ "
                "مثلاً: عند الاقتراب من الدوار، أو بعد وقوع حادث، أو عند الوقوف والتوقف."
            )
        if followup_topic == "accident" and followup_aspect == "penalty violation fine":
            return (
                "هل تقصد عقوبة ماذا بالضبط في موضوع الحادث؟ "
                "مثلاً: حادث بأضرار فقط، أو حادث مع إصابات، أو عدم الإبلاغ عن الحادث؟"
            )
        if followup_topic == "roundabout" and followup_aspect == "penalty violation fine":
            return (
                "هل تقصد عقوبة أي مخالفة في الدوار بالضبط؟ "
                "مثلاً: عدم إعطاء الأولوية، أو اختيار المسار الخاطئ، أو شيء آخر؟"
            )
        if followup_aspect == "penalty violation fine":
            return (
                "هل يمكنك توضيح المخالفة أو الحالة المقصودة بالضبط؟ "
                "العقوبة تختلف حسب الفعل المحدد."
            )
        return "هل يمكنك توضيح ما الذي تقصده بالضبط حتى أجيبك بشكل صحيح؟"

    if answer_intent == "procedure" and not followup_topic:
        return (
            "Which situation do you want step-by-step instructions for exactly? "
            "For example: approaching a roundabout, after a traffic accident, or parking and stopping."
        )
    if followup_topic == "accident" and followup_aspect == "penalty violation fine":
        return (
            "Which accident-related penalty do you mean exactly? "
            "For example: a damage-only accident, an accident with injuries, or failing to report the accident?"
        )
    if followup_topic == "roundabout" and followup_aspect == "penalty violation fine":
        return (
            "Which roundabout violation do you mean exactly? "
            "For example: not giving priority, using the wrong lane, or something else?"
        )
    if followup_aspect == "penalty violation fine":
        return (
            "Which specific violation or situation do you mean exactly? "
            "The penalty can differ depending on the exact action."
        )
    return "Can you clarify exactly what you mean so I can answer correctly?"


def should_ask_clarification_before_rag(
    question: str,
    language: str,
    is_followup: bool,
    answer_intent: str,
    followup_topic: str | None,
    followup_aspect: str | None,
) -> bool:
    normalized = " ".join(question.strip().lower().split())
    current_topics = road_topics(question, language)
    current_roles = _matching_roles(question)
    is_vehicle_modification_question = (
        (
            re.search(r"\b(car|vehicle|automobile)\b", normalized)
            and re.search(r"\b(modif|alter|change|paint|colo(u)?r|shape)\b", normalized)
        )
        or (
            re.search(r"سيار|مركب", question)
            and re.search(r"لون|طلاء|صبغ|شكل|تعديل|تغيير", question)
        )
    )
    is_allowing_other_driver_question = (
        (
            re.search(r"\b(car|vehicle|automobile)\b", normalized)
            and re.search(r"\b(let|allow|another person|someone else|others?)\b", normalized)
            and re.search(r"\bdrive\b", normalized)
        )
        or (
            re.search(r"سيار|مركب", question)
            and re.search(r"السماح|يسمح|دع|شخص آخر|شخص اخر|غيري|غيره", question)
            and re.search(r"قيادة|يقود|يسوق", question)
        )
    )
    has_specific_context = any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in [
            r"\bwhen\b",
            r"\bafter\b",
            r"\bat\b",
            r"\bnear\b",
            r"\bif\b",
            r"\bduring\b",
            r"\bfor\b",
            r"عند",
            r"بعد",
            r"اذا",
            r"إذا",
            r"أثناء",
            r"اثناء",
            r"قرب",
            r"لدى",
        ]
    )

    if answer_intent == "procedure" and not current_topics and not has_specific_context:
        return True

    if answer_intent == "permission_rule" and is_vehicle_modification_question:
        return False

    if answer_intent == "permission_rule" and is_allowing_other_driver_question:
        return False

    if (
        answer_intent == "permission_rule"
        and not current_topics
        and not current_roles
        and not has_specific_context
    ):
        return True

    if not is_followup:
        return False

    if followup_aspect == "penalty violation fine" and not current_topics and not current_roles:
        if followup_topic in {"unlicensed driver", "vehicle color"}:
            return False
        return True

    vague_followup_patterns = {
        "en": [
            r"^\s*what about (that|this|it)\s*[?.!]*\s*$",
            r"^\s*and (that|this|it)\s*[?.!]*\s*$",
            r"^\s*what happens then\s*[?.!]*\s*$",
        ],
        "ar": [
            r"^\s*ماذا عن (ذلك|هذا|هذي)\s*[؟.!]*\s*$",
            r"^\s*وماذا عن (ذلك|هذا|هذي)\s*[؟.!]*\s*$",
            r"^\s*وماذا بعدها\s*[؟.!]*\s*$",
        ],
    }

    if not current_topics and not current_roles and any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in vague_followup_patterns.get(language, [])
    ):
        return True

    return False


def clarification_result(
    language: str,
    routing: dict[str, Any],
    question: str,
    answer_intent: str,
    followup_topic: str | None,
    followup_aspect: str | None,
    rewritten_query: str | None = None,
    model: str = "",
    embedding_model: str = "",
    embedding_dimensions: int = 0,
) -> dict[str, Any]:
    return {
        "answer": build_clarification_answer(
            language,
            answer_intent=answer_intent,
            followup_topic=followup_topic,
            followup_aspect=followup_aspect,
        ),
        "sources": [],
        "is_fallback": False,
        "language": language,
        "intent": routing["intent"],
        "intent_detail": routing.get("detail", "default"),
        "answer_intent": "clarification",
        "source_route": "clarification",
        "needs_clarification": True,
        "rewritten_query": rewritten_query
        if rewritten_query and normalize_text(rewritten_query) != normalize_text(question)
        else None,
        "followup_topic": followup_topic,
        "followup_aspect": followup_aspect,
        "used_rag": False,
        "suggested_questions": smart_suggested_questions(
            language,
            answer_intent,
            question=question,
            followup_topic=followup_topic,
            needs_clarification=True,
        ),
        "model": model,
        "embedding_model": embedding_model,
        "embedding_dimensions": embedding_dimensions,
    }


def _find_keyword_group(text: str, groups: dict[str, list[str]]) -> str | None:
    normalized = text.lower()
    for label, patterns in groups.items():
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
            return label
    return None


def extract_followup_topic(chat_history: list[dict[str, str]] | None) -> str | None:
    if not chat_history:
        return None
    recent_user_messages = [
        str(message.get("content", "")).strip()
        for message in reversed(chat_history[-6:])
        if str(message.get("role", "")).strip().lower() == "user"
    ]
    for content in recent_user_messages:
        if not content:
            continue
        topic = _find_keyword_group(content, FOLLOWUP_TOPIC_KEYWORDS)
        if topic:
            return topic

    recent_text = " ".join(
        str(message.get("content", "")) for message in chat_history[-4:]
    )
    return _find_keyword_group(recent_text, FOLLOWUP_TOPIC_KEYWORDS)


def extract_followup_aspect(question: str) -> str | None:
    return _find_keyword_group(question, FOLLOWUP_ASPECT_KEYWORDS)


def rewrite_followup_question(
    question: str, chat_history: list[dict[str, str]] | None
) -> str:
    topic = extract_followup_topic(chat_history)
    aspect = extract_followup_aspect(question)
    if topic == "roundabout" and aspect == "priority right-of-way":
        return (
            "Not giving priority to the vehicles inside the roundabout before "
            "the vehicles outside it. Roundabout priority right-of-way. "
            f"Current question: {question}"
        )
    if topic == "unlicensed driver" and aspect == "penalty violation fine":
        return (
            "Allowing a person not holding a driving license to drive a vehicle. "
            "Owner, designated driver, or possessor allows an unlicensed person to drive. "
            "Fine not less than 1,000 riyals and not more than 2,000 riyals. "
            "If this results in a traffic accident, both persons are jointly liable. "
            f"Current question: {question}"
        )
    if topic == "accident" and aspect == "damage only":
        return (
            "Traffic accident with vehicle damage only and no injuries. "
            "Contact Najm. Download the Najm application, take photographs of the accident, "
            "move the vehicle only when allowed, determine the location, enter the data in the app, "
            "confirm phone number, and submit the report. "
            f"Current question: {question}"
        )
    if topic == "accident" and aspect == "fault liability responsibility":
        return (
            "Traffic accident where the driver is at fault or responsible for causing the accident. "
            "Explain the legal consequences, detention period if there are injuries or death, "
            "handover to public prosecution when required, release conditions, "
            "investigation process, and any exception such as force majeure. "
            f"Current question: {question}"
        )
    if topic == "accident" and aspect == "injuries emergency":
        return (
            "Traffic accident resulting in injuries. "
            "Call Traffic at 911 or 993 and call the Red Crescent at 997. "
            f"Current question: {question}"
        )
    if topic == "pedestrian crossing" and aspect == "risk danger safety":
        return (
            "Pedestrian crossing safety and danger when pedestrians cross outside designated areas. "
            "Pedestrians move at about 4 to 5 km/h while vehicles move much faster. "
            "This difference in speed creates high-risk situations for road users. "
            "Drivers must slow down, be prepared to stop, and must not endanger pedestrians while they are crossing. "
            f"Current question: {question}"
        )
    if topic and aspect:
        return f"In the context of {topic}, explain {aspect}. Current question: {question}"
    if topic:
        return f"In the context of {topic}, answer this follow-up question: {question}"
    return question


def last_user_topic(chat_history: list[dict[str, str]] | None) -> str | None:
    if not chat_history:
        return None
    for message in reversed(chat_history):
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        content = str(message.get("content", "")).strip()
        if content:
            return content
    return None


def rewrite_vague_question(
    question: str,
    chat_history: list[dict[str, str]] | None,
    answer_intent: str,
) -> str:
    normalized = " ".join(question.lower().split())
    vague_patterns = [
        r"\bthat\b",
        r"\bit\b",
        r"\bdo that\b",
        r"\bthis\b",
        r"ذلك",
        r"هذا",
        r"هذي",
    ]
    if not any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in vague_patterns):
        return question

    previous_topic = last_user_topic(chat_history)
    if not previous_topic:
        return question
    return (
        f"{question}\n"
        f"Rewrite context: the user's previous road-safety topic was: {previous_topic}\n"
        f"Answer intent: {answer_intent}."
    )


TOPIC_RETRIEVAL_HINTS = {
    "accident": (
        "Traffic accident procedures. Accidents with injuries and accidents with vehicle damage only. "
        "Call Traffic at 911 or 993. Call the Red Crescent at 997. Contact Najm for damage-only accidents. "
        "Moving vehicles after an accident only when allowed. Accident scene reporting and legal responsibility."
    ),
    "roundabout": (
        "Roundabout road junction rules. Give way to vehicles already inside the roundabout. "
        "Choose the correct lane or track before entering. Use turn signals before exiting. "
        "Roundabout priority right-of-way and approaching the roundabout."
    ),
    "parking": (
        "Parking stopping and waiting rules. Where parking is prohibited. When stopping is allowed instead of parking. "
        "Illegal parking violations and penalties."
    ),
    "vehicle_color": (
        "Vehicle altered by changing its color without prior permission from the competent authority. "
        "Changing the shape or color of a vehicle by repair shop owners and workers without prior valid permission. "
        "Consequences fine closure repair shop Article 25 Article 64."
    ),
    "driving_license": (
        "Driving license requirements and categories. Private driving license public driving license motorcycle driving license and temporary driving license. "
        "Minimum age to obtain each type of driving license. License validity renewal issuing conditions training hours driving school and driving test."
    ),
    "phone_use": (
        "Using mobile phones while driving. Using mobile phones without a hands-free device. "
        "Driver distraction while driving. Traffic violations points system. "
        "Driver obligations and safe driving behavior."
    ),
    "unlicensed_driver": (
        "Article 77 allowing any person not holding a driving license to drive the vehicle. "
        "Vehicle owner designated driver possessor allows unlicensed person to drive. "
        "Fine not less than 1,000 riyals and not more than 2,000 riyals. "
        "Traffic accident both persons jointly liable subject to competent court."
    ),
    "speed": (
        "Speed limits and safe driving speed. Adjusting speed to surrounding road and weather conditions. "
        "Stopping distance braking distance reaction distance and speeding penalties."
    ),
    "road_signs": (
        "Traffic signs road signs and traffic signals. Warning signs regulatory signs guide signs and driver obligations."
    ),
    "lane": (
        "Lane changing turning reversing and overtaking rules. Mirror checks signaling lane discipline and merge priority."
    ),
    "pedestrian": (
        "Pedestrian crossing and crosswalk rules. Giving way to pedestrians and driver duties near pedestrian areas. "
        "When approaching pedestrian zones drivers must check both sides of the road for pedestrians, slow down, and be prepared to stop. "
        "Drivers must not obstruct pedestrians who have started crossing and must stop in front of a pedestrian crossing when pedestrians want to cross."
    ),
}


INTENT_RETRIEVAL_HINTS = {
    "definition": "Focus on the main rule or definition for this topic.",
    "permission_rule": "Focus on whether it is allowed or prohibited, and who the rule applies to.",
    "penalty_consequence": "Focus on the violation, fine, penalty points, legal consequence, and responsible party.",
    "procedure": "Focus on practical step-by-step instructions and what the driver should do.",
    "comparison": "Focus on the directly comparable rules or procedures for this topic.",
    "clarification": "Focus on the broad governing rule and the safest directly supported explanation.",
    "followup": "Focus on the most relevant continuation of the same topic.",
    "general_road_safety": "Focus on the most directly relevant rule or guidance for this topic.",
}


def expand_retrieval_query(
    question: str,
    answer_intent: str = "general_road_safety",
) -> str:
    normalized = question.lower()
    driving_license_requirement_terms = (
        re.search(r"\blicen[sc]e\b", normalized)
        and re.search(
            r"\b(obtain|get|issue|issuing|renew|renewal|valid|validity|minimum age|at least|years|private|public|motorcycle|temporary|requirements?)\b",
            normalized,
        )
    )
    arabic_driving_license_requirement_terms = (
        re.search(r"رخصة|رخصه", question)
        and re.search(
            r"الحصول|استخراج|إصدار|اصدار|تجديد|صلاحية|مدة|الحد الأدنى|الادنى|عمر|سن|خاصة|عمومية|مؤقتة|شروط",
            question,
        )
    )

    if driving_license_requirement_terms or arabic_driving_license_requirement_terms:
        return (
            f"{question}\n"
            "Driving license requirements and categories. Private driving license public driving license motorcycle driving license and temporary driving license. "
            "Minimum age to obtain each type of driving license. Reaching the age of eighteen years for private and motorcycle driving licenses. "
            "Reaching the age of twenty years for public and construction vehicle driving licenses. Temporary driving license may be issued from age seventeen under specified controls. "
            "Training hours driving school and passing the driving test."
        )

    vehicle_color_terms = (
        re.search(r"\b(car|vehicle|automobile)\b", normalized)
        and re.search(r"\b(modif|alter|change|paint|colo(u)?r|shape)\b", normalized)
    )
    arabic_vehicle_color_terms = (
        (
            re.search(r"سيار|مركب", question)
            and re.search(r"لون|طلاء|صبغ", question)
        )
        or re.search(r"(?:سيار|مركب).*(?:تعديل|تغيير|شكل)", question)
        or re.search(r"(?:تعديل|تغيير|شكل).*(?:سيار|مركب)", question)
    )

    if vehicle_color_terms or arabic_vehicle_color_terms:
        return (
            f"{question}\n"
            "Vehicle altered by changing its color without prior permission from the competent authority. "
            "Changing the shape or color of a vehicle by repair shop owners and workers without prior valid permission. "
            "Consequences fine closure repair shop Article 25 Article 64."
        )

    unlicensed_driver_terms = (
        re.search(r"\b(unlicensed|not holding|without (?:a )?driving licen[sc]e)\b", normalized)
        or (
            re.search(r"\blicen[sc]e\b", normalized)
            and re.search(r"\b(allow|allows|let|friend|person|someone|drive|driving|vehicle|car)\b", normalized)
        )
    )
    arabic_unlicensed_driver_terms = (
        re.search(r"بدون رخصة|بدون رخصه|غير مرخص|لا يحمل رخصة|لا يحمل رخصه|ليس لديه رخصة|ليس لديه رخصه", question)
        or (
            re.search(r"رخصة|رخصه|مرخص", question)
            and re.search(r"السماح|يسمح|اسمح|دع|يدع|سمحت|شخص|صديق", question)
            and re.search(r"قيادة|يقود|يسوق|سيار|مركب", question)
        )
    )

    if unlicensed_driver_terms or arabic_unlicensed_driver_terms:
        return (
            f"{question}\n"
            "Article 77 allowing any person not holding a driving license to drive the vehicle. "
            "Vehicle owner designated driver possessor allows unlicensed person to drive. "
            "Fine not less than 1,000 riyals and not more than 2,000 riyals. "
            "Traffic accident both persons jointly liable subject to competent court."
        )

    phone_use_terms = (
        re.search(r"\b(phone|mobile|cell\s*phone|hands[- ]free|device)\b", normalized)
        and re.search(r"\b(driv(e|ing)|use|using|while)\b", normalized)
    ) or re.search(r"\bdistract(ed|ion)?\b", normalized)
    arabic_phone_use_terms = (
        re.search(r"هاتف|جوال", question)
        and re.search(r"قياد|سائق|أثناء", question)
    ) or re.search(r"بدون يد|بدون استخدام اليد|تشتيت|انشغال", question)

    if phone_use_terms or arabic_phone_use_terms:
        return (
            f"{question}\n"
            "Using mobile phones while driving. Using mobile phones without a hands-free device. "
            "Driver distraction while driving. Traffic violations points system. "
            "Driver obligations and safe driving behavior."
        )

    pedestrian_improper_crossing_terms = (
        re.search(r"\bpedestrian\b", normalized)
        and (
            re.search(r"\bnot using\b", normalized)
            or re.search(r"\bnot using a crossing\b", normalized)
            or re.search(r"\bnot using (?:a )?crosswalk\b", normalized)
            or re.search(r"\boutside\b.*\bcross", normalized)
            or re.search(r"\bwithout\b.*\bcross", normalized)
        )
    ) or (
        re.search(r"مشا|ممر", question)
        and re.search(r"بدون|خارج|ليس|غير", question)
        and re.search(r"عبور|مخصص", question)
    )

    if pedestrian_improper_crossing_terms:
        return (
            f"{question}\n"
            "Pedestrians crossing roads other than the places designated for them. "
            "Pedestrians may only use designated areas as specified by the Regulations. "
            "Driver duties near pedestrians include slowing down, being prepared to stop, and not endangering pedestrians while they are crossing."
        )

    language = "ar" if ARABIC_RE.search(question) else "en"
    topic = _primary_topic_for_question(question, language)
    topic_hint = TOPIC_RETRIEVAL_HINTS.get(topic or "")
    intent_hint = INTENT_RETRIEVAL_HINTS.get(answer_intent, "")
    if topic_hint:
        hint_parts = [topic_hint]
        if intent_hint:
            hint_parts.append(intent_hint)
        return f"{question}\n{' '.join(hint_parts)}"

    return question


def build_retrieval_query(
    question: str,
    chat_history: list[dict[str, str]] | None,
    is_followup: bool = False,
    answer_intent: str = "general_road_safety",
) -> str:
    explicit_question = rewrite_vague_question(question, chat_history, answer_intent)
    expanded_question = expand_retrieval_query(explicit_question, answer_intent)
    if not chat_history:
        return expanded_question
    if is_followup:
        rewritten = expand_retrieval_query(
            rewrite_followup_question(explicit_question, chat_history),
            answer_intent,
        )
        return f"{rewritten}\nRelevant prior context:\n{format_chat_history(chat_history[-4:])}"
    return f"{format_chat_history(chat_history[-4:])}\nCurrent question: {expanded_question}"


def _doc_combined_text(doc: Document) -> str:
    meta = doc.metadata
    return " ".join(
        str(value or "")
        for value in [
            meta.get("section_title"),
            meta.get("category"),
            meta.get("topic"),
            doc.page_content,
        ]
    ).lower()


QUERY_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "must",
    "should",
    "that",
    "the",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "allowed",
    "answer",
    "assistant",
    "authority",
    "based",
    "competent",
    "context",
    "current",
    "exceeding",
    "following",
    "general",
    "prior",
    "question",
    "relevant",
    "rule",
    "specific",
    "user",
    "valid",
    "without",
    "على",
    "عن",
    "في",
    "من",
    "ما",
    "ماذا",
    "متى",
    "كيف",
    "الذي",
    "التي",
    "يجب",
    "عند",
}


SOURCE_SELECTION_STOPWORDS = QUERY_STOPWORDS | {
    "car",
    "cars",
    "driver",
    "drivers",
    "road",
    "roads",
    "safety",
    "traffic",
    "vehicle",
    "vehicles",
    "سائق",
    "السائق",
    "سيارة",
    "سيار",
    "طريق",
    "طرق",
    "مرور",
    "مركبة",
    "مركب",
}


SOURCE_TERM_EXPANSIONS = {
    "accident": {"accident", "collision", "liability"},
    "color": {"color", "colour", "changing", "altered"},
    "colour": {"color", "colour", "changing", "altered"},
    "consequence": {"fine", "penalty", "punished", "closure", "violation"},
    "consequences": {"fine", "penalty", "punished", "closure", "violation"},
    "fine": {"fine", "penalty", "punished", "violation"},
    "modification": {"modification", "altered", "altering", "changing"},
    "modifications": {"modification", "altered", "altering", "changing"},
    "modifying": {"modification", "altered", "altering", "changing"},
    "modify": {"modification", "altered", "altering", "changing"},
    "phone": {"phone", "mobile", "hands-free", "device"},
    "mobile": {"phone", "mobile", "hands-free", "device"},
    "hands-free": {"phone", "mobile", "hands-free", "device"},
    "parking": {"parking", "stopping", "waiting"},
    "penalty": {"fine", "penalty", "punished", "violation"},
    "roundabout": {"roundabout", "priority", "track"},
    "speed": {"speed", "distance", "braking"},
    "الهاتف": {"الهاتف", "جوال", "بدون", "يد"},
    "جوال": {"الهاتف", "جوال", "بدون", "يد"},
    "لون": {"لون", "تغيير", "تعديل"},
    "تغيير": {"لون", "تغيير", "تعديل"},
    "تعديل": {"لون", "تغيير", "تعديل"},
    "عقوبة": {"عقوبة", "غرامة", "مخالفة"},
    "غرامة": {"عقوبة", "غرامة", "مخالفة"},
}


def _expanded_source_terms(text: str) -> set[str]:
    terms = {
        term
        for term in _query_terms(text)
        if term not in SOURCE_SELECTION_STOPWORDS
    }
    expanded = set(terms)
    for term in terms:
        expanded.update(SOURCE_TERM_EXPANSIONS.get(term, set()))
    return expanded


ROLE_KEYWORDS = {
    "driver": [r"\bdriver\b", r"\bdriving\b", r"سائق", r"السائق"],
    "owner": [r"\bowner\b", r"\bvehicle owner\b", r"مالك", r"صاحب المركبة"],
    "repair_shop": [
        r"\brepair shop\b",
        r"\bworkshop\b",
        r"\bgarage\b",
        r"\bmechanic\b",
        r"ورشة",
        r"إصلاح",
        r"اصلاح",
    ],
}


def _query_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}|[\u0600-\u06FF]{3,}", text.lower()):
        if token not in QUERY_STOPWORDS:
            terms.add(token)
    return terms


def _matching_roles(text: str) -> set[str]:
    normalized = text.lower()
    return {
        label
        for label, patterns in ROLE_KEYWORDS.items()
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)
    }


def rerank_retrieved_docs(
    docs_with_scores: list[tuple[Document, float]],
    retrieval_query: str,
    k: int,
    role_query: str | None = None,
) -> list[tuple[Document, float]]:
    if not docs_with_scores:
        return []

    query_roles = _matching_roles(role_query or retrieval_query)
    candidates = docs_with_scores
    if query_roles:
        role_matches = [
            (doc, score)
            for doc, score in docs_with_scores
            if _matching_roles(_doc_combined_text(doc)) & query_roles
        ]
        if role_matches:
            candidates = role_matches

    terms = _query_terms(retrieval_query)
    if not terms:
        return candidates[:k]

    ranked = []
    for doc, score in candidates:
        doc_text = _doc_combined_text(doc)
        overlap = sum(1 for term in terms if term in doc_text)
        ranked.append((overlap, score, doc))

    if not any(overlap for overlap, _, _ in ranked):
        return candidates[:k]

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(doc, score) for overlap, score, doc in ranked[:k]]


def _source_route_score(doc: Document, answer_intent: str) -> int:
    meta = doc.metadata
    doc_type = str(meta.get("document_type") or "").lower()
    source_file = str(meta.get("source_file") or "").lower()
    category = str(meta.get("category") or "").lower()
    section_title = str(meta.get("section_title") or "").lower()
    official_reference = str(meta.get("official_reference") or "").lower()
    text = _doc_combined_text(doc)

    is_law = doc_type == "law" or "traffic law" in source_file
    is_handbook = doc_type == "handbook" or "moroor" in source_file or "handbook" in source_file
    is_standard = doc_type == "standard"
    score = 0

    if answer_intent == "definition":
        if is_law:
            score += 8
        if "definition" in category or "definition" in section_title:
            score += 8
        if "article 2" in official_reference or "article 2" in section_title:
            score += 5
    elif answer_intent in {"permission_rule", "penalty_consequence"}:
        if is_law:
            score += 10
        if "violation" in category or "fine" in category or "penalt" in text:
            score += 6
        if "article" in official_reference or "article" in section_title:
            score += 3
        if is_handbook and answer_intent == "penalty_consequence":
            score -= 4
    elif answer_intent == "procedure":
        if is_handbook:
            score += 10
        if any(term in category for term in ["driver_behavior", "accident", "road_signs"]):
            score += 4
        if is_law:
            score -= 2
    elif answer_intent == "clarification":
        if is_handbook:
            score += 4
        if is_law:
            score += 2
    elif answer_intent == "comparison":
        if is_law or is_handbook:
            score += 4
        if is_standard:
            score += 1
    else:
        if is_law or is_handbook:
            score += 3

    return score


def route_retrieved_docs(
    docs_with_scores: list[tuple[Document, float]],
    answer_intent: str,
    k: int,
) -> list[tuple[Document, float]]:
    if not docs_with_scores:
        return []

    ranked = [
        (_source_route_score(doc, answer_intent), score, doc)
        for doc, score in docs_with_scores
    ]
    if not any(route_score > 0 for route_score, _, _ in ranked):
        return docs_with_scores[:k]

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(doc, score) for route_score, score, doc in ranked[:k]]


def filter_focused_docs(
    docs_with_scores: list[tuple[Document, float]],
    retrieval_query: str,
) -> list[tuple[Document, float]]:
    normalized_query = retrieval_query.lower()
    is_vehicle_color_query = (
        re.search(r"\b(car|vehicle|automobile)\b", normalized_query)
        and re.search(r"\b(modif|alter|change|paint|colo(u)?r|shape)\b", normalized_query)
    ) or (
        (
            re.search(r"سيار|مركب", retrieval_query)
            and re.search(r"لون|طلاء|صبغ", retrieval_query)
        )
        or re.search(r"(?:سيار|مركب).*(?:تعديل|تغيير|شكل)", retrieval_query)
        or re.search(r"(?:تعديل|تغيير|شكل).*(?:سيار|مركب)", retrieval_query)
    )

    if not is_vehicle_color_query:
        is_unlicensed_driver_query = (
            re.search(
                r"\b(unlicensed|not holding|without (?:a )?driving licen[sc]e|article 77)\b",
                normalized_query,
            )
            or (
                re.search(r"\blicen[sc]e\b", normalized_query)
                and re.search(r"\b(allow|allows|let|friend|person|someone|drive|driving|vehicle|car)\b", normalized_query)
            )
            or (
                re.search(r"رخصة|مرخص", retrieval_query)
                and re.search(r"قيادة|يقود|يسوق|شخص|صديق|سيار|مركب", retrieval_query)
            )
        )
        if is_unlicensed_driver_query:
            focused = []
            for doc, score in docs_with_scores:
                text = _doc_combined_text(doc)
                official_reference = str(doc.metadata.get("official_reference") or "").lower()
                is_direct_article = official_reference == "article 77"
                has_unlicensed_rule = (
                    "not holding" in text
                    and "driving" in text
                    and "license" in text
                    and "fine" in text
                )
                if is_direct_article or has_unlicensed_rule:
                    focused.append((doc, score))

            return focused or docs_with_scores

        is_phone_use_query = (
            re.search(r"\b(phone|mobile|cell\s*phone|hands[- ]free|device)\b", normalized_query)
            and re.search(r"\b(driv(e|ing)|use|using|while)\b", normalized_query)
        ) or re.search(r"\bdistract(ed|ion)?\b", normalized_query) or (
            re.search(r"هاتف|جوال", retrieval_query)
            and re.search(r"قياد|سائق|أثناء", retrieval_query)
        ) or re.search(r"بدون يد|بدون استخدام اليد|تشتيت|انشغال", retrieval_query)

        if not is_phone_use_query:
            is_pedestrian_query = (
                re.search(r"\bpedestrian\b", normalized_query)
                or re.search(r"\bcrosswalk\b|\bcrossing\b", normalized_query)
                or (
                    re.search(r"مشا|ممر", retrieval_query)
                    and re.search(r"عبور|يعبر|يعبرون|طريق", retrieval_query)
                )
            )

            if not is_pedestrian_query:
                return docs_with_scores

            is_improper_crossing_query = (
                re.search(r"\bnot using\b", normalized_query)
                or re.search(r"\bnot using a crossing\b", normalized_query)
                or re.search(r"\bnot using (?:a )?crosswalk\b", normalized_query)
                or re.search(r"\boutside\b.*\bcross", normalized_query)
                or re.search(r"\bwithout\b.*\bcross", normalized_query)
                or (
                    re.search(r"بدون|خارج|ليس|غير", retrieval_query)
                    and re.search(r"عبور|مخصص", retrieval_query)
                )
            )
            is_pedestrian_danger_query = (
                re.search(r"\bdanger\b|\bdangerous\b|\brisk\b|\brisky\b|\bsafety\b|\bhigh[- ]risk\b", normalized_query)
                or re.search(r"خطر|خطير|مخاطر|سلامة|غير آمن", retrieval_query)
            )

            focused = []
            for doc, score in docs_with_scores:
                text = _doc_combined_text(doc)
                has_pedestrian = "pedestrian" in text or "crosswalk" in text or "crossing" in text
                if is_pedestrian_danger_query:
                    has_danger_context = any(
                        term in text
                        for term in [
                            "high-risk",
                            "danger",
                            "endanger",
                            "pedestrians move at a speed",
                            "vehicles are moving at speeds much higher",
                            "difference in speed may create high-risk situations",
                            "slow down",
                            "prepared to stop",
                        ]
                    )
                    has_designated_area_rule = any(
                        term in text
                        for term in [
                            "designated areas",
                            "other than the places designated",
                            "pedestrians may only use designated areas",
                        ]
                    )
                    if has_pedestrian and (has_danger_context or has_designated_area_rule):
                        focused.append((doc, score))
                    continue
                if is_improper_crossing_query:
                    has_designated_area_rule = any(
                        term in text
                        for term in [
                            "designated areas",
                            "other than the places designated",
                            "pedestrians may only use designated areas",
                        ]
                    )
                    if has_pedestrian and has_designated_area_rule:
                        focused.append((doc, score))
                    continue
                has_driver_action = any(
                    term in text
                    for term in [
                        "slow down",
                        "prepared to stop",
                        "stop in front of a pedestrian crossing",
                        "do not obstruct pedestrians",
                        "give way to pedestrians",
                    ]
                )
                if has_pedestrian and has_driver_action:
                    focused.append((doc, score))

            return focused or docs_with_scores

        focused = []
        for doc, score in docs_with_scores:
            text = _doc_combined_text(doc).lower()
            has_phone_rule = (
                (("mobile phone" in text) or ("phone" in text))
                and "driving" in text
            ) or "hands-free" in text or "distract" in text
            has_violation_points = (
                ("mobile phone" in text or "hands-free" in text)
                and "points" in text
            )
            if has_phone_rule or has_violation_points:
                focused.append((doc, score))

        return focused or docs_with_scores

    focused = []
    for doc, score in docs_with_scores:
        text = _doc_combined_text(doc)
        official_reference = str(doc.metadata.get("official_reference") or "").lower()
        is_direct_article = official_reference in {"article 25", "article 64"}
        has_vehicle_color_rule = (
            "vehicle" in text
            and "color" in text
            and any(term in text for term in ["altered", "changing", "change"])
        )
        if is_direct_article or has_vehicle_color_rule:
            focused.append((doc, score))

    return focused or docs_with_scores


def _roundabout_priority_rank(doc: Document, score: float) -> tuple[int, float]:
    text = _doc_combined_text(doc)
    priority = 0
    if "not giving priority to the vehicles inside the roundabout" in text:
        priority -= 100
    if "inside the roundabout" in text:
        priority -= 50
    if "roundabout" in text:
        priority -= 20
    if "priority" in text or "give way" in text or "right-of-way" in text:
        priority -= 10
    if "highway" in text:
        priority += 15
    return priority, score


def filter_followup_docs(
    docs_with_scores: list[tuple[Document, float]],
    topic: str | None,
    aspect: str | None,
    k: int,
) -> list[tuple[Document, float]]:
    if topic != "roundabout" or aspect != "priority right-of-way":
        return docs_with_scores

    focused = []
    for doc, score in docs_with_scores:
        text = _doc_combined_text(doc)
        has_roundabout = "roundabout" in text
        has_priority = (
            "priority" in text
            or "give way" in text
            or "right-of-way" in text
            or "inside the roundabout" in text
        )
        if has_roundabout and has_priority:
            focused.append((doc, score))

    if not focused:
        return docs_with_scores[:k]

    focused.sort(key=lambda item: _roundabout_priority_rank(item[0], item[1]))
    return focused[: min(k, 3)]


class RoadSafetyRAG:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._vector_store: FAISS | None = None
        self._llm: ChatGoogleGenerativeAI | None = None

    @property
    def vector_store(self) -> FAISS:
        if self._vector_store is None:
            self._vector_store = load_vector_store(self.settings)
        return self._vector_store

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    def _generic_result(self, routing: dict[str, Any], language: str) -> dict[str, Any]:
        answer_intent = str(routing.get("answer_intent") or routing.get("detail", "default"))
        return {
            "answer": generic_answer(
                str(routing["intent"]),
                language,
                str(routing.get("detail", "default")),
            ),
            "sources": [],
            "is_fallback": False,
            "language": language,
            "intent": routing["intent"],
            "intent_detail": routing.get("detail", "default"),
            "answer_intent": answer_intent,
            "source_route": "no_rag",
            "needs_clarification": False,
            "used_rag": False,
            "suggested_questions": suggested_questions(language),
            "model": self.settings.llm_model,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
        }

    def _prepare_rag(
        self,
        question: str,
        top_k: int | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        routing = detect_intent(question, chat_history=chat_history)
        language = str(routing["language"])
        if not routing["use_rag"]:
            return None, self._generic_result(routing, language)

        k = top_k or self.settings.retriever_top_k
        answer_intent = str(routing.get("answer_intent") or routing.get("detail") or "general_road_safety")
        is_followup = answer_intent == "followup" or routing.get("detail") == "followup"
        followup_topic = extract_followup_topic(chat_history) if is_followup else None
        followup_aspect = extract_followup_aspect(question) if is_followup else None
        retrieval_query = build_retrieval_query(
            question,
            chat_history if is_followup else None,
            is_followup=bool(is_followup),
            answer_intent=answer_intent,
        )
        if should_ask_clarification_before_rag(
            question,
            language,
            bool(is_followup),
            answer_intent,
            followup_topic,
            followup_aspect,
        ):
            return None, clarification_result(
                language,
                routing,
                question,
                answer_intent,
                followup_topic,
                followup_aspect,
                rewritten_query=retrieval_query,
                model=self.settings.llm_model,
                embedding_model=self.settings.embedding_model,
                embedding_dimensions=self.settings.embedding_dimensions,
            )
        candidate_docs_with_scores = self.vector_store.similarity_search_with_score(
            retrieval_query, k=max(k * 4, 12)
        )
        candidate_docs_with_scores = filter_focused_docs(
            candidate_docs_with_scores, retrieval_query
        )
        docs_with_scores = filter_followup_docs(
            candidate_docs_with_scores, followup_topic, followup_aspect, k
        )
        docs_with_scores = route_retrieved_docs(docs_with_scores, answer_intent, k)
        docs_with_scores = rerank_retrieved_docs(
            docs_with_scores, retrieval_query, k, role_query=question
        )
        context = format_context(docs_with_scores)
        prompt_inputs = get_prompt_inputs(
            question,
            context,
            chat_history=format_chat_history(chat_history) if is_followup else "No recent conversation.",
            answer_intent=answer_intent,
        )
        messages = RAG_PROMPT.format_messages(**prompt_inputs)
        return {
            "docs_with_scores": docs_with_scores,
            "followup_aspect": followup_aspect,
            "followup_topic": followup_topic,
            "answer_intent": answer_intent,
            "is_followup": is_followup,
            "messages": messages,
            "prompt_inputs": prompt_inputs,
            "question": question,
            "retrieval_query": retrieval_query,
            "source_route": SOURCE_ROUTE_BY_INTENT.get(answer_intent, "balanced"),
            "routing": routing,
        }, None

    def _finish_rag_result(self, prepared: dict[str, Any], answer: str) -> dict[str, Any]:
        prompt_inputs = prepared["prompt_inputs"]
        routing = prepared["routing"]
        clean_answer = clean_answer_formatting(strip_generated_sources(answer))
        fallback_used = normalize_text(clean_answer) == normalize_text(
            prompt_inputs["fallback_message"]
        )

        return {
            "answer": clean_answer,
            "sources": []
            if fallback_used
            else select_answer_sources(
                prepared["docs_with_scores"],
                clean_answer,
                prompt_inputs["language"],
                question=prepared["retrieval_query"],
            ),
            "is_fallback": fallback_used,
            "language": prompt_inputs["language"],
            "intent": routing["intent"],
            "intent_detail": routing.get("detail", "default"),
            "answer_intent": prepared["answer_intent"],
            "source_route": prepared["source_route"],
            "needs_clarification": prepared["answer_intent"] == "clarification",
            "rewritten_query": prepared["retrieval_query"]
            if normalize_text(prepared["retrieval_query"]) != normalize_text(prepared["question"])
            else None,
            "followup_topic": prepared["followup_topic"],
            "followup_aspect": prepared["followup_aspect"],
            "used_rag": True,
            "suggested_questions": smart_suggested_questions(
                prompt_inputs["language"],
                prepared["answer_intent"],
                question=prepared["question"],
                followup_topic=prepared["followup_topic"],
                needs_clarification=fallback_used,
            ),
            "model": self.settings.llm_model,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
        }

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        prepared, generic_result = self._prepare_rag(question, top_k, chat_history)
        if generic_result is not None:
            return generic_result

        assert prepared is not None
        messages = prepared["messages"]
        response = self.llm.invoke(messages)
        answer = message_to_text(response.content)
        return self._finish_rag_result(prepared, answer)

    def ask_stream(
        self,
        question: str,
        top_k: int | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ):
        prepared, generic_result = self._prepare_rag(question, top_k, chat_history)
        if generic_result is not None:
            yield {
                "type": "metadata",
                **{key: value for key, value in generic_result.items() if key != "answer"},
            }
            yield {"type": "chunk", "content": generic_result["answer"]}
            yield {"type": "done", **generic_result}
            return

        assert prepared is not None
        prompt_inputs = prepared["prompt_inputs"]
        routing = prepared["routing"]
        yield {
            "type": "metadata",
            "is_fallback": False,
            "language": prompt_inputs["language"],
            "intent": routing["intent"],
            "intent_detail": routing.get("detail", "default"),
            "answer_intent": prepared["answer_intent"],
            "source_route": prepared["source_route"],
            "needs_clarification": prepared["answer_intent"] == "clarification",
            "rewritten_query": prepared["retrieval_query"]
            if normalize_text(prepared["retrieval_query"]) != normalize_text(prepared["question"])
            else None,
            "followup_topic": prepared["followup_topic"],
            "followup_aspect": prepared["followup_aspect"],
            "used_rag": True,
            "suggested_questions": smart_suggested_questions(
                prompt_inputs["language"],
                prepared["answer_intent"],
                question=prepared["question"],
                followup_topic=prepared["followup_topic"],
                needs_clarification=False,
            ),
            "model": self.settings.llm_model,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
        }

        answer_parts: list[str] = []
        for chunk in self.llm.stream(prepared["messages"]):
            content = stream_chunk_to_text(chunk.content)
            if not content:
                continue
            answer_parts.append(content)
            yield {"type": "chunk", "content": content}

        answer = "".join(answer_parts)
        yield {"type": "done", **self._finish_rag_result(prepared, answer)}
