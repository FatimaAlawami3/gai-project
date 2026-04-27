# DALIL System Documentation

## 1. System Overview

DALIL is an AI-powered road safety chatbot for Saudi Arabia built as a Retrieval-Augmented Generation (RAG) system over a curated road-safety knowledge base. The project is designed to answer user questions about Saudi traffic law, driving procedures, violations, accidents, roundabouts, licensing, parking, phone use while driving, and related safety guidance while staying grounded in authoritative source material.

This is not a generic chatbot with unconstrained generation. Its architecture deliberately separates:

- authoritative content preparation,
- semantic retrieval,
- intent-aware routing,
- follow-up interpretation,
- grounded response generation,
- and presentation of citations in the UI.

The implemented system combines:

- a FastAPI backend,
- a LangChain-based RAG orchestration layer,
- Vertex AI embedding configurations for controlled model comparison,
- Gemini generation models,
- a local FAISS vector index,
- a React/Vite chat frontend,
- and a curated knowledge base derived from selected Saudi road-safety PDFs.

The system is production-oriented in design even though it is run locally during development. It includes typed API contracts, startup warmup, streaming responses, metadata-rich retrieval, deterministic routing for non-RAG intents, bilingual behavior, structured answer formatting, iterative fixes for ambiguous queries and multi-turn follow-ups, and a profile-based embedding comparison setup for experimental evaluation.

## 2. Objectives and Design Philosophy

The project aims to develop an AI-powered road safety chatbot using a Retrieval-Augmented Generation (RAG) approach. In the implemented system, that objective becomes a strict engineering rule: the model should answer only from retrieved source context and should not invent legal details, penalties, article numbers, or driving rules.

Key design principles implemented in code:

- Source-grounded responses only.
- Prefer Saudi Traffic Law for legal obligations and penalties.
- Prefer handbook guidance for practical driving behavior and procedures.
- Treat chat history as conversational context, not as evidence.
- Detect and short-circuit non-road-safety intents before retrieval.
- Ask for clarification when the user query is too vague to answer safely.
- Preserve bilingual usability in Arabic and English.
- Keep references separate from generated prose so the answer remains readable.

## 3. Knowledge Base and Data Pipeline

### 3.1 Source Corpus Selection

The knowledge base is built from three PDF sources stored in `data/pdfs/`:

- `Traffic Law.pdf`
- `Theoretical Driving Handbook Trainee-Moroor.pdf`
- `101 EN.pdf`

These sources map to three distinct knowledge roles in the system:

- `Traffic Law.pdf`
  - Legal authority layer.
  - Used for articles, duties, violations, penalties, and legal responsibility.
- `Theoretical Driving Handbook Trainee-Moroor.pdf`
  - Procedural and driver-guidance layer.
  - Used for practical driving instructions, safety behavior, and operational guidance.
- `101 EN.pdf`
  - Standards/reference layer.
  - Used for road-safety and highway-code style material.

This selection reflects an intentional hierarchy rather than a simple document dump. The system distinguishes law, handbook guidance, and standards because the answer generator must not mix practical advice and legal obligation incorrectly.

### 3.2 PDF Parsing Pipeline

The ingestion pipeline is implemented in `scripts/GAI_Jason_Convertor.py`.

Parsing is performed with PyMuPDF (`fitz`). Each PDF is read page by page using `page.get_text("text")`, then normalized through a `clean_text()` function that:

- removes soft hyphens,
- converts bullet glyphs,
- replaces non-breaking spaces,
- normalizes horizontal whitespace,
- compresses excessive newlines,
- and trims noisy line spacing.

This cleaning stage is important because downstream chunking and embeddings depend heavily on coherent text boundaries. Without cleanup, PDF extraction artifacts would degrade semantic chunk quality and reduce retrieval precision.

### 3.3 Document-Aware Structural Parsing

The converter does not use one generic chunking rule for all PDFs. It uses document-specific section detectors:

- Traffic Law:
  - detects sections by `Article <number>`
- Moroor handbook:
  - detects units and numbered topic headings
- Saudi Highway Code / SHC:
  - detects decimal-numbered sections such as `1`, `1.1`, `2.1`, `3.2.21`
- Generic fallback:
  - page-based blocks if structure cannot be detected

This is a strong design decision in the project. Instead of blindly chunking by characters alone, the system first reconstructs document structure, then chunks within those semantic sections. That makes retrieval more aligned with how users ask questions such as:

- "What does Article 77 say?"
- "What should a driver do after an accident?"
- "Who has priority in a roundabout?"

### 3.4 Chunking Strategy

The chunking policy is defined directly in the converter:

- maximum chunk size: `1800` characters
- minimum chunk size: `300` characters
- chunking style: semantic section chunking
- overlap: no fixed sliding-window overlap is used

The implementation first keeps an entire detected section if it is small enough. If a section is too long, it is split:

- first by paragraphs,
- then by grouped sentences if a paragraph still exceeds the limit.

Very small trailing chunks are merged back into the previous chunk to avoid embedding low-value fragments.

This means the project uses a structure-aware variable chunking strategy rather than a constant-token sliding window. That is a meaningful implementation choice:

- it preserves article integrity for legal text,
- keeps handbook procedures together,
- and reduces fragmentation of rule-consequence relationships.

### 3.5 Metadata Enrichment

Each chunk is converted into a metadata-rich JSON entry, not just raw text. The converter adds:

- `chunk_id`
- `document_id`
- `source_id`
- `document_title`
- `document_type`
- `authority`
- `source_file`
- `source_priority`
- `section_type`
- `section_number`
- `section_title`
- `unit_or_chapter`
- `page_start`
- `page_end`
- `category`
- `topic`
- `keywords`
- `content_type`
- `language`
- `country`
- `text`
- `text_cleaned`
- `citation`
- `qa_hints`
- `retrieval_priority`

This enrichment is central to the retrieval system. The chatbot does not rely only on vector distance. It also uses these fields for:

- citation formatting,
- source prioritization,
- follow-up document filtering,
- route-specific reranking,
- and final source selection shown in the UI.

### 3.6 Semantic Categorization

During JSON generation, each section is mapped into domain categories such as:

- `driving_license_rules`
- `traffic_violations_and_fines`
- `traffic_points_system`
- `speed_limits_and_speed_safety`
- `traffic_accidents_and_legal_liability`
- `road_signs`
- `right_of_way_and_intersections`
- `stopping_and_parking_rules`
- `traffic_law_definitions`

This categorization is rule-based and derived from content signals in section titles and text. It gives the retrieval layer more structure than plain embeddings alone.

### 3.7 Current Knowledge Base Size

The current generated JSON file is `data/saudi_road_safety_kb.json`.

Actual corpus statistics from the project data:

- documents: `3`
- chunks: `619`

Chunk distribution by document type:

- `standard`: `362`
- `handbook`: `168`
- `law`: `89`

Most frequent categories currently include:

- `road_signs`
- `traffic_violations_and_fines`
- `highway_safety_concepts`
- `driving_license_rules`
- `speed_limits_and_speed_safety`

## 4. Embedding Layer and Vector Database

### 4.1 Embedding Models

The final project uses `gemini-embedding-001` as the primary embedding model, and it now includes a controlled comparison baseline using `text-multilingual-embedding-002`.

The two configured embedding profiles are:

- `gemini-embedding-001`
  - output dimensionality: `3072`
  - role: final selected production-style configuration
- `text-multilingual-embedding-002`
  - output dimensionality: `768`
  - role: multilingual baseline for experimental comparison

These settings are defined in `backend/rag_config.py` and used in `backend/rag_chain.py`.

Embeddings are created through `GoogleGenerativeAIEmbeddings`. The implementation supports two runtime modes:

- Gemini API key mode
- Vertex AI mode with service-account credentials

This dual-path configuration is a practical deployment choice. The system can run with direct API credentials or with Google Cloud / Vertex AI credentials depending on environment setup.

### 4.2 Embedding Construction

The vectorization pipeline is implemented in `load_kb_documents()` and `build_vector_store()`.

The process is:

1. Read `saudi_road_safety_kb.json`
2. Convert each chunk into a LangChain `Document`
3. Store `text_cleaned` as `page_content`
4. Carry selected metadata fields into the document metadata
5. Embed each document with Gemini embeddings
6. Build a FAISS index from the full document set

The system intentionally stores scalarized metadata only. Non-scalar metadata is JSON-serialized through `_scalar_metadata()` so the vector store can safely persist it.

### 4.3 Vector Stores

The project uses local FAISS, and after the comparison extension it maintains separate vector stores per embedding profile:

- backend builder: `backend/build_vector_store.py`
- persisted files:
  - `data/vector_store/index.faiss`
  - `data/vector_store/index.pkl`
  - `data/vector_store_gemini_embedding_001/index.faiss`
  - `data/vector_store_gemini_embedding_001/index.pkl`
  - `data/vector_store_text_multilingual_embedding_002/index.faiss`
  - `data/vector_store_text_multilingual_embedding_002/index.pkl`

Vector store loading is lazy but cached through the `RoadSafetyRAG` instance. On API startup, the backend warms the vector store and the LLM so first-token latency is reduced after server boot.

The experimental setup deliberately isolates vector stores by embedding model so that:

- the original Gemini-based index is preserved,
- the multilingual baseline can be built without overwriting the original index,
- and the frontend can switch between embedding profiles without changing the shared knowledge base or prompt logic.

### 4.4 Similarity Search

The retrieval engine uses:

- FAISS similarity search with score
- top-k expansion before pruning

The actual search call is:

- `similarity_search_with_score(retrieval_query, k=max(k * 4, 12))`

This means the system deliberately over-retrieves first, then applies additional filtering and reranking. The effective architecture is therefore:

- dense retrieval
- focused filtering
- follow-up-aware filtering
- intent-aware routing
- lexical/role-aware reranking

This is more sophisticated than a basic "top-3 nearest chunks" pipeline.

## 5. Retrieval-Augmented Generation Pipeline

### 5.1 End-to-End Flow

The implemented request flow is:

1. User submits a question from the React frontend
2. Frontend sends:
   - `question`
   - `top_k`
   - `chat_history`
   - `embedding_profile`
3. FastAPI receives the request in `/ask` or `/ask/stream`
4. `RoadSafetyRAG._prepare_rag()` runs intent detection
5. If the query is non-RAG:
   - return static/generic answer immediately
6. If the query is RAG:
   - detect answer intent
   - detect follow-up topic/aspect when applicable
   - rewrite or expand the retrieval query
   - optionally trigger clarification instead of retrieval
   - retrieve candidate chunks from FAISS
   - filter and rerank those chunks
   - format context blocks
   - inject context into the system/human prompt
   - call Gemini chat model
   - clean answer formatting
   - select citations separately
   - return structured response metadata

### 5.2 Prompt Engineering

Prompt logic is centralized in `backend/prompts.py`.

The prompt template explicitly instructs the model to:

- answer only from retrieved context,
- use chat history only for follow-up interpretation,
- ignore unrelated earlier conversation,
- prefer law over handbook when legal duties are involved,
- avoid inventing penalties or article numbers,
- avoid mixing responsible parties,
- answer in Arabic or English based on the user question,
- and keep references out of the main answer text.

The prompt is parameterized by:

- answer style
- source preference
- answer language
- fallback message
- source heading label
- retrieved context
- recent conversation summary

This gives the system answer-mode-specific behavior without having to build multiple separate chains.

### 5.3 Grounding Enforcement

Grounding is enforced at multiple levels:

- router-level short-circuiting for unsupported intents,
- prompt-level "use only retrieved context" instruction,
- explicit fallback message when context is insufficient,
- separate citation rendering instead of model-generated citation prose,
- source stripping and formatting cleanup after generation,
- and source selection only from retrieved documents.

The model is also instructed not to produce its own sources section. The backend strips any generated `Sources` or `References` block if it appears, and the frontend renders references independently from structured source objects.

### 5.4 Context Formatting

Retrieved documents are serialized into structured context blocks containing:

- rank
- chunk id
- document title
- section title
- category
- English citation
- Arabic citation
- retrieval distance
- full chunk text

This is a strong design choice: the generator sees not just raw text, but also the chunk's documentary identity and citation metadata. That helps the model reason about source type and traceability while still keeping citations out of the final answer text.

## 6. Intent Detection and Routing System

### 6.1 Router Role

Intent detection is implemented in `backend/intent_router.py`. This module decides whether a question should:

- bypass RAG entirely,
- use static project/system responses,
- or enter the full retrieval pipeline.

This prevents expensive retrieval and generation for simple conversational queries.

### 6.2 Intent Classes in the Implemented System

The current routing logic distinguishes:

- `road_safety`
- `project_info`
- `greeting`
- `thanks`
- `capability`
- `general`

For RAG-routed road-safety queries, the router also assigns a more specific answer intent such as:

- `definition`
- `permission_rule`
- `penalty_consequence`
- `procedure`
- `comparison`
- `clarification`
- `followup`
- `general_road_safety`

### 6.3 Static Meta Capabilities

Project and capability questions do not go through retrieval. They are answered from hardcoded router logic.

The system can answer questions such as:

- what the project is called,
- who is behind the project,
- who supervised it,
- what course it was built for,
- what DALIL can do.

This is implemented through:

- pattern-based detection for project queries,
- detail-level routing such as `team`, `supervisor`, `course`, `project_name`,
- and fixed bilingual answers returned directly from `project_info_answer()` and `GENERIC_ANSWERS`.

This is an important architectural decision: project metadata is not in the vector store. It is part of the application layer.

### 6.4 Why the Router Matters

Without the router, many simple messages would unnecessarily hit the vector database and LLM. Instead, DALIL only enters RAG when it is actually dealing with a road-safety knowledge request.

This reduces:

- latency,
- cost,
- and the chance of irrelevant retrieval.

## 7. Follow-Up Handling and Conversation Memory

### 7.1 Memory Model

The system does not use a database-backed memory or full conversational state store. Conversation memory is lightweight and request-scoped.

The frontend sends recent `chat_history`, and the backend uses it only for follow-up interpretation.

Important implementation detail:

- history limit: `6` messages
- this is not six questions
- it is usually about three user/assistant turns

The frontend builds history from visible chat messages and excludes messages marked `excludeFromHistory`.

### 7.2 Follow-Up Detection

The router determines follow-ups through:

- `history_has_road_safety_context()`
- `looks_like_followup()`
- `history_road_topics()`
- `is_directly_related_followup()`

The logic combines:

- short-query heuristics,
- keyword-based follow-up cues,
- and topic overlap between the current question and recent user messages.

Arabic and English are both supported.

Examples of follow-up style cues handled in code:

- English:
  - `what about`
  - `and`
  - `then`
  - `it`
  - `that`
- Arabic:
  - `ماذا عن`
  - `طيب`
  - `هل`
  - `لو`
  - `ولو`

### 7.3 Topic and Aspect Extraction

The RAG layer extracts:

- follow-up topic
- follow-up aspect

Topic examples include:

- roundabout
- parking
- accident
- driving license
- phone use
- unlicensed driver
- speed / safe distance
- traffic signals / road signs
- pedestrian crossing
- vehicle modification / color

Aspect examples include:

- priority / right-of-way
- penalty / violation / fine
- damage only
- injuries / emergency
- fault / liability / responsibility

This lets the system treat follow-ups as structured retrieval problems instead of vague chat continuations.

### 7.4 Follow-Up Query Rewriting

When a follow-up is detected, the system rewrites the retrieval query using topic-aware templates.

Implemented examples include:

- roundabout priority:
  - expands to text about giving priority to vehicles inside the roundabout
- unlicensed driver penalty:
  - expands with Article 77 semantics and fine range
- accident with damage only:
  - expands with Najm workflow and reporting behavior
- accident fault/liability:
  - expands toward detention, prosecution, investigation, and force majeure
- accident with injuries:
  - expands toward emergency contact procedure

The system also formats recent conversation into a short textual context block and appends it to the retrieval query for follow-ups.

### 7.5 Vague Follow-Up Repair

The project implements a second rewriting layer for vague pronoun-based follow-ups such as:

- `What happens if I do that?`
- `What about that?`
- informal Arabic accident-responsibility follow-ups expressed as short, incomplete, or colloquial continuations of the previous turn

If the current user question contains vague referents such as `that`, `it`, `هذا`, or `ذلك`, the system rewrites it with the last user topic so retrieval is anchored properly.

The same retrieval layer also contains topic-aware expansion for certain direct legal-rule questions. A later development fix added a dedicated phone-use topic so English and Arabic questions about mobile-phone use while driving are expanded toward the relevant handbook and traffic-points material instead of depending on one narrow surface phrasing.

## 8. Clarification Handling

One of the most important advanced features added during development is clarification-first behavior for underspecified questions.

The system now checks whether a question should be clarified before RAG. This is implemented in `should_ask_clarification_before_rag()`.

Cases that can trigger clarification include:

- broad procedural questions with no topic
- vague permission/rule questions with no clear action or role, unless the wording already matches a direct rule path such as vehicle modification or allowing another person to drive the car
- ambiguous penalty follow-ups
- vague anaphoric follow-ups with no recoverable topic

Examples of targeted clarification behavior added during iteration:

- asking for "the penalty" after an accident answer now asks which accident-related penalty is meant
- asking "What should the driver do step by step?" with no clear scenario now triggers a clarification question

This replaced earlier fallback behavior that said there was not enough information, which was less helpful in multi-turn use.

## 9. Retrieval Refinement and Source Selection

### 9.1 Query Expansion

The system expands some high-value query types before retrieval.

Implemented special expansions and topic-aware retrieval hints include:

- vehicle modification / repainting / color change
- allowing an unlicensed person to drive
- driving-license requirements, minimum age, validity, and renewal questions
- speed and safe-driving-distance questions
- traffic-signal and road-sign questions
- pedestrian-crossing safety questions

These expansions inject critical legal phrases and related consequences into the search query to improve retrieval of the correct article or rule.

### 9.2 Multi-Stage Retrieval Control

The retrieval stack is not a single FAISS call. It includes:

- FAISS candidate retrieval
- focused document filtering
- follow-up-aware filtering
- intent-based routing of retrieved docs
- reranking based on lexical overlap and role matching

This is especially important for questions where semantically similar but legally different documents might otherwise be mixed together.

### 9.3 Source Selection for UI References

References shown to the user are selected after answer generation, not before. The backend:

- formats candidate sources,
- deduplicates them,
- checks whether the answer implicitly references them,
- compares answer/question term overlap against retrieved documents,
- and returns a compact list of up to three relevant citations.

This allows the answer itself to stay clean while the UI still presents traceable references.

## 10. Backend Architecture

### 10.1 FastAPI Application

The backend entry point is `backend/main.py`.

Implemented endpoints:

- `GET /health`
- `POST /ask`
- `POST /ask/stream`

`/health` reports:

- knowledge base availability
- vector store availability
- configured LLM model
- embedding model
- embedding dimensions
- whether Vertex AI is being used
- default embedding profile
- available embedding profiles

### 10.2 Typed Request/Response Models

The API uses Pydantic models:

- `AskRequest`
- `ChatMessage`
- `AskResponse`

The response model includes rich metadata beyond the answer itself:

- `answer`
- `sources`
- `is_fallback`
- `language`
- `intent`
- `intent_detail`
- `answer_intent`
- `source_route`
- `needs_clarification`
- `rewritten_query`
- `followup_topic`
- `followup_aspect`
- `used_rag`
- `suggested_questions`
- `embedding_profile`
- model metadata

This makes the backend observable and easier to debug.

### 10.3 Streaming Design

The frontend primarily uses `POST /ask/stream`, which returns NDJSON events.

The event pattern is:

- `metadata`
- `chunk`
- `done`

This gives the UI immediate state visibility before final answer completion. The metadata event already includes intent, routing fields, and the active embedding profile before all tokens arrive.

### 10.4 Backend Components

The main backend modules are:

- `main.py`
  - FastAPI app and API schema
- `rag_chain.py`
  - core RAG orchestration, retrieval, rewriting, reranking, answer packaging
- `intent_router.py`
  - intent detection, generic replies, project info, suggested questions
- `prompts.py`
  - prompt template, answer styles, grounding rules, fallback policy
- `rag_config.py`
  - environment/config loading and credentials resolution
- `build_vector_store.py`
  - offline index build entry point

## 11. Frontend Architecture

### 11.1 Stack

The frontend is implemented with:

- React 19
- React DOM 19
- Vite 7

It is intentionally small and centered around one main file:

- `frontend/src/main.jsx`
- `frontend/src/styles.css`

### 11.2 Frontend Responsibilities

The frontend handles:

- message state,
- history extraction,
- streaming consumption,
- answer cleanup,
- source rendering,
- starter/suggested question chips,
- embedding-profile selection,
- bilingual UI switching,
- and chat reset behavior.

### 11.3 Rendering Strategy

The UI does not trust raw model formatting blindly. It performs client-side formatting cleanup and structural rendering through:

- `cleanAnswerText()`
- `FormattedText`
- `InlineMarkdown`
- `Sources`

This is important because the model may still return varying line layouts. The frontend normalizes them into readable sections and bullet lists.

### 11.4 Suggested Question Behavior

A major implementation change during development was to stop allowing suggested questions to drift based on model output.

The frontend now uses a fixed whitelist of starter questions and ignores dynamic backend variation in practice. The list currently includes:

- traffic accident action
- phone use while driving
- driving without a license
- letting someone else drive the car
- car modification
- roundabout behavior
- who is behind the project

The suggestion language switches between English and Arabic based on the selected UI language.

### 11.5 Language Toggle

The frontend now includes a manual language switcher. When the user toggles Arabic:

- UI copy changes,
- suggestion chips switch to Arabic,
- the chat header brand name changes from `DALIL` to `دليل`.

When toggled back to English, the fixed suggestion set returns to English immediately.

### 11.6 Embedding Model Toggle

The frontend now also includes a second top-level control for embedding comparison. The user can switch between:

- `gemini-embedding-001`
- `text-multilingual-embedding-002`

This control sends the selected `embedding_profile` to the backend with each request. The response bubble also displays a short embedding badge so the active retrieval configuration is visible during evaluation. This turned the production UI into a lightweight experiment console without requiring separate frontend builds.

## 12. Real Folder Structure and Responsibilities

### 12.1 Root Layout

The actual project structure is:

- `backend/`
  - API, routing, prompts, RAG logic, config
- `comparison_versions/`
  - embedding comparison setup, per-model settings, helper scripts, evaluation material
- `frontend/`
  - React/Vite chat interface
- `data/`
  - PDFs, generated JSON KB, FAISS vector store, credentials
- `scripts/`
  - PDF-to-JSON conversion script
- `logs/`
  - local runtime logs
- `README.md`
  - setup and project overview

### 12.2 Data Folder

`data/` contains the entire retrieval substrate:

- `pdfs/`
  - source PDF corpus
- `saudi_road_safety_kb.json`
  - generated structured knowledge base
- `vector_store/`
  - FAISS index files
- `vector_store_gemini_embedding_001/`
  - isolated FAISS index for the Gemini embedding experiment
- `vector_store_text_multilingual_embedding_002/`
  - isolated FAISS index for the multilingual embedding experiment
- `credentials/`
  - local Google credential material for Vertex AI mode

### 12.3 Comparison Folders

`comparison_versions/` was added to support the model-comparison requirement without disturbing the main project configuration.

It contains:

- `comparison_versions/gemini-embedding-001/`
  - `settings.env`
  - `build_vector_store.ps1`
  - `run_api.ps1`
  - `README.md`
- `comparison_versions/text-multilingual-embedding-002/`
  - `settings.env`
  - `build_vector_store.ps1`
  - `run_api.ps1`
  - `README.md`
- `comparison_versions/embedding_comparison_question_sheet.md`
  - structured evaluation prompts and scoring template
- `comparison_versions/embedding_comparison_report_sections.md`
  - report-ready experiment and results text

### 12.4 Public Assets

`frontend/public/` contains branding assets such as:

- `dalil-avatar.png`
- `dalil-logo.png`
- `dalil-symbol.png`

These are used in the visual identity of the chat experience.

## 13. Meta-Capabilities of the Chatbot

DALIL is not only a road-safety QA system. It also has application-level self-description features.

Implemented meta questions include:

- Who is behind this project?
- What is the project name?
- What course was it built for?
- Who supervised it?
- What can DALIL do?

These are not hallucinated by the model and are not retrieved from PDFs. They are handled deterministically in `intent_router.py` through hardcoded bilingual responses.

This hybrid design is important:

- road-safety facts come from the KB through RAG
- system/project facts come from fixed application metadata

## 14. Development Iterations and Improvements

This project evolved substantially during development. Several features were added or corrected through iteration.

### 14.1 Answer Formatting Improvements

Originally, some answers came back as long paragraphs or numbered steps.

We changed the system so that:

- the prompt asks for a short direct answer first,
- then a localized heading such as `Key Points:` in English or `النقاط الرئيسية:` in Arabic,
- then concise bullet points,
- and procedure answers use bullets rather than numbered lists.

We also adjusted the frontend renderer so that if the model still returns numbered lists, they are displayed as bullets.

### 14.2 Clarification Instead of Fallback

Originally, ambiguous follow-up questions sometimes triggered the generic fallback:

- "I do not have enough information..."

This was improved by adding clarification logic for underspecified cases. Now DALIL asks the user to clarify the exact violation, scenario, or procedure when the question is too broad.

### 14.3 Follow-Up Handling Repairs

Follow-up handling required multiple rounds of improvement.

Issues addressed:

- vague English follow-ups not mapping correctly to prior topics
- informal Arabic follow-ups not being recognized reliably
- Arabic accident-responsibility follow-ups being interpreted too narrowly
- pronoun-based follow-ups like `What happens if I do that?` being misrouted
- follow-up topic extraction being contaminated by assistant text instead of user topic

Fixes implemented:

- expanded English fault/liability phrases
- expanded Arabic follow-up detection beyond one exact phrase to cover broader informal cues and fault/responsibility wording
- widened the rule-based Arabic coverage for accident-blame and liability signals so common informal variants are more likely to be linked back to the previous accident topic
- changed follow-up topic extraction to prioritize recent user messages
- improved vague follow-up rewriting using the last user topic
- added a dedicated `phone_use` road topic for English and Arabic phone/mobile-driving questions
- added phone-use retrieval expansion and focused document filtering so valid phone-use questions no longer fall back when the supporting chunks already exist in the knowledge base
- added direct retrieval support for driving-license requirement questions in English and Arabic
- widened signal and speed coverage to include traffic-light wording plus braking/reaction-distance vocabulary
- added pedestrian improper-crossing handling and danger/safety follow-up support
- broadened the vehicle-modification topic so questions like `Is modifying a car allowed?` are handled as direct legal-rule questions instead of unnecessary clarification cases
- tightened Arabic vehicle-modification matching so phrases like `تعديل السرعة` are treated as speed questions rather than color/modification questions
- localized answer section labels so Arabic answers use Arabic headings such as `النقاط الرئيسية:` instead of English labels

These changes broadened Arabic follow-up coverage, but they do not imply exhaustive understanding of every possible informal Arabic phrasing.

### 14.4 Suggested Question Control

Suggested questions also went through several iterations.

Changes made:

- replaced vague or overly ambitious suggestions with a fixed approved list
- added `Who is behind this project?`
- localized suggestions into Arabic and English
- made suggestion language follow the selected UI language
- stopped allowing suggestions to change based on DALIL's answer content

This stabilized the UI and kept the prompts consistent with project scope.

### 14.5 UI Language Improvements

We added a user-facing language toggle and then refined it further so that:

- UI text switches correctly,
- suggestions change with the selected language,
- the DALIL title changes to `دليل` in Arabic mode.

### 14.6 Live Server Behavior

During debugging, one important operational issue appeared:

- backend code changes were sometimes correct in files but not reflected in behavior

The reason was that the running server process was not always using reload mode. Restarting the FastAPI server was necessary in some debugging cycles to validate the real behavior of updated follow-up logic.

### 14.7 Embedding Comparison Infrastructure

To satisfy the course requirement for baseline comparison without destabilizing the main system, we extended the architecture with embedding-profile support.

Changes made:

- added `RAG_ENV_FILE` support so the backend can load per-experiment settings files
- added separate `settings.env` files for Gemini and multilingual experiments
- created separate vector-store destinations for each embedding model
- copied the original Gemini index into its own experiment folder
- built a second FAISS index using `text-multilingual-embedding-002`
- added a frontend embedding-model switch so both configurations can be tested from the same UI
- fixed a real bug in config loading by forcing environment overrides, so two profiles can be loaded correctly in the same Python process

## 15. Evaluation and Testing

### 15.1 Development Validation Style

Testing in this project was pragmatic and feature-driven rather than formal benchmark-driven.

We validated the system through:

- syntax checks,
- frontend production builds,
- targeted API smoke tests,
- live conversational regression testing,
- side-by-side embedding comparison runs,
- and direct inspection of returned routing metadata.

### 15.2 Query Types Explicitly Tested

Examples of tested questions during iteration include:

- What should a driver do when approaching a roundabout?
- What about priority?
- What should I do after a traffic accident?
- What is the related penalty?
- Can I let my friend drive my car without a license?
- What happens if I do that?
- what if I was the wrong?
- Is it allowed to use a phone while driving?
- هل يسمح باستخدام الهاتف أثناء القيادة؟
- informal Arabic accident-responsibility follow-up variants written in short or colloquial form
- Is modifying a car allowed?
- Who is behind this project?
- وش العقوبة؟
- What is the weather today?
- Who are you?
- هل يسمح باستخدام الهاتف أثناء القيادة؟
- ماذا يجب علي فعله بعد وقوع حادث مروري؟

### 15.3 Edge Cases Addressed

The system was specifically hardened against:

- ambiguous penalty follow-ups
- overly broad procedural questions
- English/Arabic language switching mismatches
- suggestion chip drift
- assistant-generated inline source sections
- follow-ups with pronouns instead of explicit topic names
- informal Arabic accident-responsibility phrasing across multiple variants
- configuration leakage between embedding profiles in the same Python process

### 15.4 Verification Signals

The backend exposes useful observability fields that help validate correctness:

- `intent`
- `intent_detail`
- `answer_intent`
- `used_rag`
- `needs_clarification`
- `followup_topic`
- `followup_aspect`
- `rewritten_query`

These fields were used during debugging to confirm whether the system:

- used RAG or bypassed it,
- identified a follow-up correctly,
- requested clarification correctly,
- rewrote the query as intended,
- and actually used the selected embedding profile.

### 15.5 Embedding Comparison Results

After the comparison infrastructure was added, we ran a controlled side-by-side evaluation between:

- `gemini-embedding-001`
- `text-multilingual-embedding-002`

Both models were evaluated under identical conditions:

- same PDFs
- same knowledge-base JSON
- same prompts
- same LLM
- same retrieval parameter (`top_k = 3` during the comparison run)

Observed outcome:

- both embedding configurations answered the evaluation set successfully without falling back
- both preserved the improved follow-up behavior for English and Arabic multi-turn cases
- `gemini-embedding-001` produced more complete accident-related and Arabic answers
- `text-multilingual-embedding-002` remained competitive on roundabout and unlicensed-driver questions, but several answers were shorter and less detailed

These comparison results refer to the scored experiment snapshot. Later routing and retrieval fixes broadened the live system further in areas such as driving-license questions, traffic signals, speed/safe-distance wording, pedestrian-crossing follow-ups, and some Arabic disambiguation cases.

This led to the final selection of `gemini-embedding-001` as the preferred production embedding model, with `text-multilingual-embedding-002` retained as the experimental baseline.

## 16. Current Technical Stack

### Backend

- FastAPI `0.128.0`
- Uvicorn `0.40.0`
- LangChain `1.2.15`
- langchain-community `0.4.1`
- langchain-google-genai `4.2.2`
- faiss-cpu `1.13.2`
- python-dotenv `1.2.2`
- pydantic-settings `2.13.1`
- google-auth `2.49.2`

### Frontend

- React `19.2.1`
- React DOM `19.2.1`
- Vite `7.3.0`
- `@vitejs/plugin-react`

## 17. System Summary

DALIL is a bilingual, metadata-aware, intent-routed RAG chatbot tailored to Saudi road safety. The implemented system goes beyond naive question answering by combining document-aware ingestion, structured chunk metadata, Gemini embeddings, FAISS retrieval, route-sensitive prompting, deterministic application-level routing, clarification logic, and follow-up rewriting.

Its most important engineering characteristics are:

- authority-aware use of law versus guidance,
- profile-based embedding comparison with isolated vector stores,
- prevention of unsupported free-form answers,
- explicit handling of ambiguity,
- repair of multi-turn context failures,
- and a frontend that stabilizes answer readability and bilingual usability.

In its current state, the project is a well-structured local AI application with clear subsystem boundaries, real retrieval grounding, and a development history that reflects iterative hardening toward a reliable road-safety assistant rather than a generic chatbot.
