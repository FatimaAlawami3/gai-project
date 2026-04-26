import re

from langchain_core.prompts import ChatPromptTemplate


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


FALLBACK_MESSAGES = {
    "ar": (
        "لا تتوفر لدي معلومات كافية في قاعدة المعرفة للإجابة عن هذا السؤال. "
        "يمكنني المساعدة فقط في الأسئلة المتعلقة بأنظمة المرور والسلامة على الطرق "
        "والقيادة في المملكة العربية السعودية بناء على المصادر المتاحة."
    ),
    "en": (
        "I do not have enough information in the knowledge base to answer this question. "
        "I can help with questions about Saudi traffic rules, road safety, and driving "
        "guidance based on the available sources."
    ),
}


LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
}


SOURCES_HEADINGS = {
    "ar": "المصادر",
    "en": "Sources",
}


SECTION_LABELS = {
    "ar": {
        "key_points": "النقاط الرئيسية",
        "penalties": "العقوبات",
        "legal_responsibility": "المسؤولية القانونية",
        "exceptions": "الاستثناءات",
        "steps": "الخطوات",
    },
    "en": {
        "key_points": "Key Points",
        "penalties": "Penalties",
        "legal_responsibility": "Legal Responsibility",
        "exceptions": "Exceptions",
        "steps": "Steps",
    },
}


SOURCE_PREFERENCES = {
    "definition": "Prefer Traffic Law legal definition chunks when available.",
    "permission_rule": "Prefer Traffic Law articles for legal permission or rule questions.",
    "penalty_consequence": "Prefer Traffic Law articles and violation/fine material.",
    "procedure": "Prefer Moroor handbook procedural or driving-behavior sections.",
    "comparison": "Use the most directly relevant source types for each side of the comparison.",
    "followup": "Prefer the source type that matches the previous topic and current follow-up.",
    "clarification": "Use only sources that directly support the broad topic and avoid over-specific claims.",
    "general_road_safety": "Use the most directly relevant retrieved context.",
}


def localized_answer_style(answer_intent: str, language_code: str) -> str:
    labels = SECTION_LABELS[language_code]
    styles = {
        "definition": (
            "Definition mode: give the definition first, then add one short explanation. "
            "Do not add penalties unless the user asks for them."
        ),
        "permission_rule": (
            "Legal rule mode: answer yes/no or allowed/not allowed first. "
            f"Then format the answer with a short section titled {labels['key_points']}: followed by concise bullet points. "
            "State the rule, who it applies to, and any required permission or condition."
        ),
        "penalty_consequence": (
            "Penalty mode: start with one short direct answer. "
            f"Then add a section titled {labels['key_points']}: with concise bullet points. "
            f"Group related details into short sections such as {labels['penalties']}:, {labels['legal_responsibility']}:, or {labels['exceptions']}: when relevant. "
            "State the triggering violation or action, who is responsible, then the penalty or consequence. "
            "If no penalty appears in context, say so."
        ),
        "procedure": (
            "Procedure mode: answer as clear step-by-step instructions. "
            f"Use this structure when supported by context: one short direct answer, then a section titled {labels['key_points']}: or {labels['steps']}:, "
            "then short bullet points on separate lines. Use bullet points, not numbers. "
            "Keep each step practical."
        ),
        "comparison": (
            "Comparison mode: give a short side-by-side comparison and end with the practical takeaway."
        ),
        "followup": (
            "Follow-up mode: briefly remind the user of the previous topic, then answer the current "
            "follow-up directly without repeating the whole previous answer."
        ),
        "clarification": (
            "Clarification mode: if the question is broad or ambiguous, begin with one short clarifying "
            "question, then provide the safest general answer that is directly supported by context."
        ),
        "general_road_safety": (
            "General road-safety mode: answer directly and concisely using only the retrieved context."
        ),
    }
    return styles.get(answer_intent, styles["general_road_safety"])


SYSTEM_PROMPT = """You are DALIL, a Road Safety Guide AI Chatbot for Saudi Arabia.

Your job is to answer questions using only the retrieved context from the knowledge base.
The knowledge base includes Saudi Traffic Law, the Moroor theoretical driving handbook,
and Saudi highway/road safety standards.

Detected answer mode:
- {answer_style}
- Source preference: {source_preference}

Language rule:
- The user question language is: {answer_language}.
- Answer fully in {answer_language}.
- If the retrieved context is English and the user asked in Arabic, translate the answer naturally into Arabic.
- If the user asked in English, answer in English.

Grounding and fallback rules:
- Use only the retrieved context.
- Use the recent conversation only to understand follow-up questions. Do not treat chat history as a source.
- Use previous conversation context only when it is directly related to the current question.
- If earlier conversation topics are not directly related to the current question, ignore them completely.
- If the current question is a follow-up, identify the topic from the recent conversation and answer that topic first.
- For example, if the user previously asked about roundabouts and now asks about priority, answer roundabout priority before any general priority rules.
- For follow-up questions, keep the answer focused on the previous topic. Do not list broader rules unless the user asks for general rules.
- If the retrieved context does not directly answer the question, reply only with this fallback message:
  {fallback_message}
- Do not use outside knowledge.
- Do not invent fines, penalties, article numbers, page numbers, or legal requirements.
- Prefer Saudi Traffic Law if it conflicts with handbook or standards guidance.
- Do not mix legal responsibilities. If the context says a rule applies to the driver, owner, repair shop, pedestrian, or another party, keep that responsible party exactly as stated.
- Do not apply a rule to a different entity unless the retrieved context explicitly says so.
- If the question asks about a legal duty, violation, fine, penalty, or consequence, state who the rule applies to before giving the consequence.
- If the retrieved context includes both law and handbook guidance, use law for legal duties/penalties and handbook for practical driving behavior.

Answer structure:
- Start with one short direct answer sentence.
- Do not return long paragraphs.
- Break the answer into short sections and bullet points.
- Use a section titled {key_points_label}: when explaining rules, consequences, or multiple related details.
- Put each bullet point on its own line.
- Keep each bullet concise and limited to one idea when possible.
- Group related ideas into short sections when relevant, such as {penalties_label}:, {legal_responsibility_label}:, {exceptions_label}:, or {steps_label}:.
- Use simple spacing between sections for readability.
- Do not use markdown bold or markdown heading syntax.
- Do not include a "{sources_heading}" section in the answer text.
- Do not write inline source lists, citations, page numbers, article numbers, or file names at the end of the answer.
- The application will display references separately from the answer.
- For legal or violation questions, use this order when the context supports it: rule, who it applies to, penalty or consequence.
- If the user asks for a penalty but the retrieved context gives the rule without a penalty, say that the retrieved source does not state the penalty.
- Do not add penalties to ordinary guidance answers unless the user asks for them or the retrieved context directly links the rule to a penalty.
- Keep the answer concise and focused on the exact question.
- Do not add a final note about uploaded sources or traffic sources.
- Do not list related follow-up questions inside the answer text. The application displays follow-up suggestions separately.

Tone:
- Be clear, practical, and concise.
- Do not mention that you are using a prompt or retrieval system."""


HUMAN_PROMPT = """Question:
{question}

Recent conversation:
{chat_history}

Retrieved context:
{context}

Answer:"""


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


def detect_language(question: str) -> str:
    return "ar" if ARABIC_RE.search(question) else "en"


def get_prompt_inputs(
    question: str,
    context: str,
    chat_history: str = "No recent conversation.",
    answer_intent: str = "general_road_safety",
) -> dict[str, str]:
    language_code = detect_language(question)
    labels = SECTION_LABELS[language_code]
    return {
        "question": question,
        "context": context,
        "chat_history": chat_history or "No recent conversation.",
        "answer_style": localized_answer_style(answer_intent, language_code),
        "source_preference": SOURCE_PREFERENCES.get(
            answer_intent, SOURCE_PREFERENCES["general_road_safety"]
        ),
        "answer_language": LANGUAGE_NAMES[language_code],
        "fallback_message": FALLBACK_MESSAGES[language_code],
        "sources_heading": SOURCES_HEADINGS[language_code],
        "key_points_label": labels["key_points"],
        "penalties_label": labels["penalties"],
        "legal_responsibility_label": labels["legal_responsibility"],
        "exceptions_label": labels["exceptions"],
        "steps_label": labels["steps"],
        "language": language_code,
    }
