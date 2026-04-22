# AI-Powered Road Safety Chatbot for Saudi Arabia: Research Paper-Style Technical Documentation

## I. Introduction

Generative Artificial Intelligence has transformed the design of interactive information systems by enabling natural language understanding, contextual reasoning, and fluent response generation over large knowledge spaces. In particular, large language models (LLMs) have made it possible to build conversational agents capable of synthesizing domain-specific knowledge into accessible responses. However, general-purpose LLMs are not intrinsically reliable for regulated or safety-sensitive domains because they may produce ungrounded or hallucinated content when asked about legal rules, procedural obligations, or operational constraints. This limitation is especially critical in road safety contexts, where inaccurate advice may mislead drivers about legal duties, accident procedures, licensing obligations, or penalty consequences.

To address this problem, the DALIL project was designed as an AI-powered road safety chatbot for Saudi Arabia using a Retrieval-Augmented Generation (RAG) architecture. Rather than relying on unconstrained parametric generation, the system grounds its responses in a curated corpus of Saudi traffic and driving references, then uses an LLM only after retrieving relevant evidence. The resulting system acts as a domain-specific conversational assistant capable of answering questions about Saudi Traffic Law, the Moroor theoretical driving handbook, road behavior, roundabouts, traffic accidents, parking, and related safety procedures.

The motivation for the project emerged from the practical difficulty of accessing and interpreting formal road-safety information. Saudi traffic law documents, driver handbooks, and standards contain valuable but structurally heterogeneous information. For many end users, especially those seeking quick clarification during learning or compliance scenarios, directly navigating these documents is inefficient. DALIL was therefore conceived as a bilingual conversational interface that translates authoritative road-safety material into accessible, grounded, and context-aware answers in both English and Arabic.

The project was built around three primary documentary sources:

- `Traffic Law.pdf`, representing the legal and regulatory layer.
- `Theoretical Driving Handbook Trainee-Moroor.pdf`, representing practical and educational driver guidance.
- `101 EN.pdf`, representing broader road-safety and standards-oriented reference material.

These sources were deliberately selected to cover complementary functions. The legal source provides enforceable rules, article-based duties, and consequences. The Moroor handbook provides procedural interpretation and practical driver guidance. The standards-oriented source broadens coverage for signage, highway concepts, and structural road-safety practices. This documentary composition was necessary because a law-only corpus would have been too rigid for procedure-oriented questions, while a handbook-only corpus would have been too weak for penalties and legal liability.

The implemented goals of the project were:

- to ingest and structurally transform official road-safety documents into a machine-retrievable knowledge base;
- to index that knowledge base semantically using embeddings and local vector search;
- to route queries intelligently depending on whether they are road-safety questions, follow-up questions, or system/meta questions;
- to support multi-turn conversational interaction without treating previous chat turns as factual evidence;
- to prevent hallucinated penalties, article numbers, or legal duties;
- to provide readable, structured answers with separately rendered citations;
- and to support bilingual operation through Arabic/English language detection and interface adaptation.

From an engineering perspective, DALIL is a hybrid deterministic-generative system. Static project facts and application capabilities are handled through explicit router logic, while domain questions are answered through retrieval and grounded generation. This separation gives the final system a more reliable operational profile than a purely generative chatbot.

## II. Methodology

### A. System Architecture

The implemented DALIL architecture follows a retrieval-augmented conversational pipeline in which user input is first interpreted at the application layer, then optionally passed through a semantic retrieval and grounded generation sequence. At a high level, the operational flow can be described as:

User -> Query Submission -> Intent Detection -> Retrieval Query Construction -> Embedding-Oriented Candidate Search -> Document Filtering and Reranking -> Prompt Context Injection -> LLM Response Generation -> Post-Processing -> Structured API Response -> Frontend Rendering.

In the simplified framing requested by the course, the core RAG path is:

User -> Query -> Retrieval -> LLM -> Response.

In the actual implementation, however, the system is more controlled. The backend does not send every query directly into the vector search stage. Instead, it first determines whether the message belongs to:

- a road-safety query that should use RAG,
- a follow-up question that should use contextual retrieval,
- or a generic/system/meta intent that should bypass RAG entirely.

This distinction is essential because many messages do not require retrieval. Greetings, thanks, capability questions, and project-information questions are handled deterministically, which improves latency and reduces the risk of irrelevant retrieval.

The architecture is divided into the following major subsystems:

- **Presentation Layer**
  - React/Vite chat interface in `frontend/`
  - handles input capture, streaming display, language toggle, answer formatting, suggestion chips, and source rendering
- **API Layer**
  - FastAPI server in `backend/main.py`
  - exposes health, standard ask, and streaming ask endpoints
- **Intent and Application Logic Layer**
  - `backend/intent_router.py`
  - determines whether a query is road-safety, follow-up, project-info, greeting, thanks, capability, or general unsupported chatter
- **Retrieval and Orchestration Layer**
  - `backend/rag_chain.py`
  - performs vector search, query rewriting, follow-up interpretation, filtering, reranking, prompt construction, and result packaging
- **Prompt Control Layer**
  - `backend/prompts.py`
  - defines grounding rules, answer styles, fallback policy, and language enforcement
- **Configuration Layer**
  - `backend/rag_config.py`
  - resolves environment variables, credentials, model settings, and file paths
- **Offline Knowledge Preparation Layer**
  - `scripts/GAI_Jason_Convertor.py`
  - converts PDFs into structured JSON chunks with metadata
  - `backend/build_vector_store.py`
  - generates the FAISS vector index from the knowledge base

Architecturally, DALIL is not a monolithic "chat with an LLM" application. It is a layered inference stack with deterministic routing before generation and post-generation sanitation after inference. This layered organization is one of the most important characteristics of the final implementation.

### B. Data Pipeline

#### 1. Source Selection

The data pipeline begins from a curated set of three PDF documents stored under `data/pdfs/`:

- `Traffic Law.pdf`
- `Theoretical Driving Handbook Trainee-Moroor.pdf`
- `101 EN.pdf`

These documents are not treated as interchangeable. Each one fulfills a distinct role:

- `Traffic Law.pdf`
  - statutory rules, legal obligations, violations, penalties, and article-based references
- `Theoretical Driving Handbook Trainee-Moroor.pdf`
  - educational and procedural driving guidance
- `101 EN.pdf`
  - standards-oriented road-safety and signage reference material

This source composition gave the system both legal precision and practical usability. That was necessary because the chatbot needed to answer both "what is legally allowed?" and "what should the driver do?" questions.

#### 2. PDF Parsing and Cleaning

The parsing workflow is implemented in `scripts/GAI_Jason_Convertor.py`. Parsing is performed using PyMuPDF (`fitz`), with each PDF processed page by page via `page.get_text("text")`.

The raw extracted text is normalized through `clean_text()`, which performs:

- removal of soft hyphen artifacts;
- normalization of PDF bullet symbols;
- replacement of non-breaking spaces;
- compaction of repeated spaces and tabs;
- reduction of excessive blank lines;
- trimming of stray whitespace around line breaks.

These cleanup operations are essential because PDF extraction noise degrades both chunk quality and embedding quality.

#### 3. Structural Segmentation

Instead of applying a generic splitter to every document, the system uses document-specific detectors:

- **Traffic Law detector**
  - identifies article boundaries such as `Article <number>`
- **Moroor handbook detector**
  - tracks unit headings and numbered topic headings
- **SHC / standards detector**
  - captures decimal section structures such as `1`, `1.1`, `2.1`, and `3.2.21`
- **generic detector**
  - falls back to page-level segmentation if structure is not reliably inferable

This was a deliberate design decision. The corpus is structurally heterogeneous, so uniform splitting would have weakened legal and procedural coherence.

#### 4. Conversion to JSON

After structural segmentation, the processed content is serialized into `data/saudi_road_safety_kb.json`. This file contains:

- a `metadata` block describing the knowledge base;
- a `documents` list containing source-level metadata;
- a `chunks` list containing each retrieval unit and its associated metadata.

This JSON file is the canonical machine-readable representation of the corpus and serves as the input to the vector-store build process.

#### 5. Chunking Strategy

The implemented chunking configuration is explicitly defined in the converter:

- maximum characters per chunk: `1800`
- minimum characters per chunk: `300`
- chunking style: semantic section chunking
- overlap: no fixed sliding overlap was implemented

The chunking logic proceeds hierarchically:

- if a section is already short enough, it is kept intact;
- if too long, it is split by paragraphs;
- if a paragraph still exceeds the threshold, it is split into grouped sentences;
- if a resulting fragment is too small, it is merged back into the previous chunk.

This means DALIL uses variable-length semantic chunking rather than a fixed token-window splitter. The approach preserves legal article integrity, keeps handbook procedures coherent, and avoids producing low-value microscopic chunks.

The final generated knowledge base contains:

- `3` source documents
- `619` chunks

#### 6. Metadata Design

Each chunk stores rich metadata, including:

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

This metadata is actively consumed later for reranking, source assembly, category-sensitive routing, and reference display.

#### 7. Content Categorization

The converter assigns rule-based semantic categories such as:

- `traffic_violations_and_fines`
- `traffic_accidents_and_legal_liability`
- `driving_license_rules`
- `road_signs`
- `speed_limits_and_speed_safety`
- `right_of_way_and_intersections`
- `traffic_points_system`
- `stopping_and_parking_rules`

This category layer turns the JSON corpus into a semantically structured retrieval substrate rather than a plain text archive.

### C. Embedding & Vector Database

#### 1. Embedding Model

The project uses Google Gemini embeddings:

- embedding model: `gemini-embedding-001`
- embedding dimensionality: `3072`

These settings are defined in `backend/rag_config.py` and instantiated in `backend/rag_chain.py` through `GoogleGenerativeAIEmbeddings`.

The system supports two authentication modes:

- direct Gemini API key mode
- Vertex AI mode with Google service-account credentials

This dual-path configuration makes the architecture operationally portable between local API use and Google Cloud-backed execution.

#### 2. Vector Representation

Each cleaned chunk is converted into a dense vector representation. The vector captures semantic proximity rather than exact lexical overlap, enabling the system to retrieve relevant material even when user wording differs from source wording.

Operationally, the embedding flow is:

1. load structured JSON chunks;
2. transform each chunk into a LangChain `Document`;
3. preserve retrieval-relevant metadata alongside the text;
4. generate a 3072-dimensional dense embedding for each document;
5. store the resulting vectors in a FAISS index.

#### 3. Storage and Indexing

The vector database is local FAISS, generated via `backend/build_vector_store.py` and stored under:

- `data/vector_store/index.faiss`
- `data/vector_store/index.pkl`

The system uses `FAISS.from_documents(...)` during build time and `FAISS.load_local(...)` during runtime. The vector store is loaded once and cached inside the `RoadSafetyRAG` instance. On backend startup, the system warms both the vector store and the LLM client to reduce cold-start latency.

#### 4. Similarity Search and Top-k Retrieval

The DALIL system uses top-k semantic retrieval through `similarity_search_with_score(...)`. The implementation does not define a hand-written cosine similarity routine. Instead, FAISS returns similarity/distance scores through the LangChain FAISS interface, and the backend then performs additional filtering and reranking.

The search strategy is deliberately **over-retrieve first, prune later**:

- if the final desired `k` is small, the system first retrieves a larger candidate pool using `max(k * 4, 12)`
- the candidate set is then refined through focused filters and reranking

The operational retrieval stack is therefore:

- dense semantic candidate generation
- focused filtering
- follow-up-aware filtering
- intent-aware routing
- lexical and role-aware reranking

This proved more effective than a naive top-3 nearest-neighbor configuration for legal and procedure-sensitive queries.

### D. Retrieval-Augmented Generation (RAG Pipeline)

#### 1. End-to-End Retrieval Flow

Once a query is classified as a road-safety question requiring retrieval, the backend executes the following sequence:

1. detect answer intent and determine whether the query is a follow-up;
2. extract follow-up topic and aspect if relevant;
3. rewrite or expand the query for better retrieval specificity;
4. check whether clarification is needed before retrieval;
5. retrieve candidate chunks from FAISS;
6. apply focused filtering and route-aware filtering;
7. rerank the retrieved candidates;
8. serialize the selected context into prompt-ready blocks;
9. inject context and recent conversation into the prompt;
10. invoke the Gemini chat model;
11. clean the generated output;
12. attach citations separately.

This sequence is implemented primarily in `RoadSafetyRAG._prepare_rag()` and `RoadSafetyRAG._finish_rag_result()` inside `backend/rag_chain.py`.

#### 2. Query Preparation

The raw user query is often not used directly. Before retrieval, the system may transform it through:

- vague follow-up rewriting;
- topic-aware follow-up rewriting;
- special-case query expansion;
- chat-history injection for contextual disambiguation.

Implemented special expansions include:

- vehicle color modification;
- allowing an unlicensed person to drive.

These expansions were introduced because direct semantic search over the original wording was sometimes insufficient to retrieve the most legally relevant chunks.

#### 3. Context Injection

Retrieved documents are formatted into explicit context blocks containing:

- retrieval rank;
- chunk identifier;
- document title;
- section title;
- category;
- English citation;
- Arabic citation;
- retrieval distance score;
- raw chunk text.

The prompt therefore receives not only content but also documentary identity and source descriptors.

#### 4. Prompt Engineering

Prompt behavior is defined centrally in `backend/prompts.py`. The prompt template contains:

- a system prompt that encodes grounding, language, answer-structure, and source-governance rules;
- a human prompt containing:
  - the current question,
  - recent conversation,
  - retrieved context.

The system prompt is parameterized by answer mode. Implemented answer styles include:

- `definition`
- `permission_rule`
- `penalty_consequence`
- `procedure`
- `comparison`
- `followup`
- `clarification`
- `general_road_safety`

Each answer style modifies both expected structure and source preference. Penalty questions emphasize rule, responsible party, and consequence. Procedure questions emphasize practical stepwise guidance. Follow-up mode emphasizes continuity with the previous topic without repeating the entire previous answer.

#### 5. Grounding and Hallucination Prevention

Hallucination prevention is enforced through several coordinated mechanisms:

- prompt-level grounding that instructs the model to use only retrieved context;
- conversation governance that allows chat history only for interpreting follow-up intent;
- a controlled fallback policy when retrieved context is insufficient;
- source stripping so the model does not produce its own references section;
- answer sanitation and formatting cleanup;
- explicit legal-role integrity rules that prevent mixing driver, owner, repair-shop, or other parties unless the context supports it.

DALIL is therefore not only retrieval-augmented; it is grounding-constrained.

### E. Intent Detection & Routing

#### 1. Intent Detection Mechanism

Intent detection is implemented in `backend/intent_router.py` using pattern-based analysis plus lightweight conversational context checks. This design was selected because the application needed predictable and inspectable control over routing behavior.

The router first detects the user language, then evaluates the normalized input against several intent classes before retrieval occurs.

#### 2. Supported Intent Types

The implemented system distinguishes these top-level intents:

- `road_safety`
- `project_info`
- `greeting`
- `thanks`
- `capability`
- `general`

For `road_safety` questions, the router also assigns a more specific answer intent:

- `definition`
- `permission_rule`
- `penalty_consequence`
- `procedure`
- `comparison`
- `clarification`
- `followup`
- `general_road_safety`

#### 3. Routing Logic

The routing decision process can be summarized as:

1. if the query matches project-information patterns:
   - route to static project-information response
   - do not use RAG
2. else if the query is a greeting:
   - return a generic greeting
   - do not use RAG
3. else if the query expresses thanks:
   - return a generic thanks response
   - do not use RAG
4. else if the query asks about chatbot capabilities:
   - return a capability response
   - do not use RAG
5. else if the query is a directly related follow-up to recent road-safety context:
   - route to `road_safety` with detail `followup`
   - use RAG
6. else if the query matches road-safety topic patterns:
   - route to `road_safety`
   - use RAG
7. else:
   - route to general fallback behavior without RAG

This logic sharply distinguishes domain questions from application/meta questions and prevents unnecessary vector search.

#### 4. Why Routing Was Necessary

Routing became one of the most important design decisions in the final system. Without it:

- greetings and off-topic messages would unnecessarily consume retrieval and generation resources;
- project questions such as "Who is behind this project?" would be forced through the vector store;
- vague follow-ups would be harder to interpret correctly.

The router therefore functions as the control center of the DALIL backend.

### F. Follow-Up Handling & Conversation Memory

#### 1. Multi-Turn Conversation Model

DALIL does not use a database-backed memory store. Instead, conversation memory is handled through lightweight recent-message history sent from the frontend. The frontend extracts a limited window of visible messages and includes them in each API request as `chat_history`.

The implemented history limit is:

- `6` recent messages

This means the effective context window is usually about three question-answer turns rather than six user questions.

#### 2. Context Preservation Strategy

The backend uses recent history only to interpret the current user question. It does **not** treat chat history as documentary evidence. This separation between conversational context and factual source grounding was an explicit rule in the prompt and a central design principle during development.

#### 3. Follow-Up Recognition

The system uses several helper functions in `intent_router.py` to identify follow-ups:

- `history_has_road_safety_context()`
- `looks_like_followup()`
- `history_road_topics()`
- `is_directly_related_followup()`

The logic combines:

- short-form query heuristics;
- pronoun/anaphora cues;
- question brevity;
- topic overlap between current and recent user messages.

It supports both English and Arabic follow-up patterns.

#### 4. Topic and Aspect Extraction

Once a follow-up is recognized, the backend extracts:

- a **follow-up topic**
- a **follow-up aspect**

Implemented follow-up topics include:

- roundabout
- parking
- accident
- unlicensed driver
- vehicle color

Implemented aspects include:

- priority/right-of-way
- penalty/violation/fine
- damage-only accident behavior
- injuries/emergency behavior
- fault/liability/responsibility

This decomposition allowed the system to distinguish, for example, a roundabout-priority follow-up from a roundabout-penalty follow-up.

#### 5. Follow-Up Query Rewriting

The system includes explicit rewrite templates for high-value follow-up situations. Examples include:

- roundabout priority rewrites;
- accident injury rewrites;
- damage-only accident rewrites;
- accident fault/liability rewrites;
- unlicensed-driver penalty rewrites.

Additionally, vague follow-ups such as:

- `What happens if I do that?`
- `What about that?`
- `ولو كنت غلطان؟`

are transformed using the last relevant user topic.

#### 6. Improvements Made During Development

Follow-up handling required substantial refinement. The major improvements we made include:

- preventing vague accident-related penalty questions from falling back too early;
- making ambiguous questions ask for clarification instead of saying there is no answer;
- expanding Arabic follow-up detection with short colloquial forms such as `لو` and `ولو`;
- adding Arabic fault/liability terms like `غلطان`, `غلطانة`, and `غلط`;
- prioritizing recent **user** topics instead of assistant messages during topic extraction;
- fixing cases where follow-ups were incorrectly pulled toward the wrong previous topic.

These improvements materially increased multi-turn coherence and reduced misrouting.

### G. Backend & Implementation

#### 1. Backend Framework

The backend is implemented with FastAPI in `backend/main.py`. The application exposes three endpoints:

- `GET /health`
- `POST /ask`
- `POST /ask/stream`

The health route reports:

- knowledge base availability;
- vector store availability;
- configured LLM model;
- embedding model;
- embedding dimensions;
- whether Vertex AI mode is active.

#### 2. Request/Response Flow

The request schema is defined through Pydantic models:

- `AskRequest`
- `ChatMessage`
- `AskResponse`

Requests contain:

- `question`
- optional `top_k`
- `chat_history`

Responses contain not only the answer but also rich control metadata:

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
- model descriptors

This response schema significantly improved observability and debugging fidelity.

#### 3. LangChain Pipeline Components

The LangChain-oriented retrieval and generation logic resides in `backend/rag_chain.py`.

The principal components are:

- **retriever**
  - FAISS vector store loaded through LangChain
- **LLM**
  - `ChatGoogleGenerativeAI` configured with `gemini-3.1-pro-preview`
- **prompt layer**
  - `RAG_PROMPT` from `backend/prompts.py`
- **memory/context adapter**
  - lightweight chat-history formatter and follow-up rewrite logic
- **router integration**
  - `detect_intent()` and related helper functions from `intent_router.py`

Although the system does not use a separate LangChain memory module, it does implement application-level contextual memory through formatted chat history and follow-up query transformation.

#### 4. Streaming Implementation

The frontend primarily uses `POST /ask/stream`, which emits NDJSON events in three phases:

- `metadata`
- `chunk`
- `done`

This allows the interface to know, before completion, whether the response is:

- a RAG answer,
- a generic routed answer,
- a clarification,
- or a fallback.

#### 5. Actual Folder Structure and Component Roles

The root project structure is:

- `backend/`
  - FastAPI API
  - intent router
  - prompt configuration
  - RAG orchestration
  - environment-driven settings
- `frontend/`
  - React/Vite user interface
  - streaming consumer
  - bilingual UI controls
  - answer and source rendering
- `data/`
  - original PDF sources
  - generated JSON knowledge base
  - FAISS vector store
  - credential files
- `scripts/`
  - PDF preprocessing and JSON generation
- `logs/`
  - local runtime logs
- `README.md`
  - setup and usage instructions
- `SYSTEM_DOCUMENTATION.md`
  - earlier technical system overview

Within the frontend:

- `frontend/src/main.jsx`
  - application logic, message state, history management, streaming fetch, formatting, UI language switching
- `frontend/src/styles.css`
  - chat interface styling and bilingual layout styling
- `frontend/public/`
  - DALIL branding assets such as avatar and logo files

Within the backend:

- `backend/main.py`
  - API entry point
- `backend/rag_chain.py`
  - retrieval, rewrite, reranking, generation, result packaging
- `backend/intent_router.py`
  - intent detection, meta/system replies, suggested questions
- `backend/prompts.py`
  - prompt design and output policies
- `backend/rag_config.py`
  - configuration and credentials
- `backend/build_vector_store.py`
  - offline vector-store builder
- `backend/requirements.txt`
  - dependency specification

### H. Meta Capabilities

DALIL can answer not only road-safety questions but also application-level meta questions such as:

- "Who built this system?"
- "What data do you use?"
- "How do you work?"
- "What is the project name?"
- "Who supervised the project?"

These capabilities were implemented in two different ways depending on content type.

#### 1. Predefined Meta Logic

Project identity and team questions are handled in `intent_router.py` using predefined bilingual responses. This includes:

- project name;
- team information;
- supervisor information;
- course affiliation;
- general project overview.

These answers are intentionally not retrieved from the vector database because they are application facts rather than corpus facts.

#### 2. System Behavior Explanation

Questions about DALIL’s operational scope are supported through capability responses and prompt-governance logic. The chatbot can explain that it helps with:

- Saudi road safety;
- traffic rules;
- roundabouts;
- violations;
- parking;
- accidents;
- safe driving practices.

In practice, "how DALIL works" is embodied in:

- deterministic intent routing;
- retrieval-grounded answer generation;
- source-separated rendering;
- and bilingual output control.

## III. Experimental Design

Because DALIL is an applied RAG system rather than a train-from-scratch generative model, the experimental design emphasizes behavioral validation, retrieval correctness, routing correctness, and conversation robustness rather than supervised training metrics.

### 1. Testing Strategy

The project relied on iterative testing during development, including:

- backend syntax checks;
- frontend production builds through `npm run build`;
- targeted backend smoke tests;
- manual API validation via `/ask` and `/ask/stream`;
- end-to-end conversational verification through the frontend interface.

This testing strategy was chosen because the main risks were not training collapse or gradient instability, but:

- retrieval mismatch;
- query-routing mistakes;
- ambiguous follow-up failures;
- bilingual UI inconsistency;
- incorrect answer formatting.

### 2. Example Query Set

Representative evaluation queries included:

- "What should a driver do when approaching a roundabout?"
- "What about priority?"
- "What should I do after a traffic accident?"
- "What is the related penalty?"
- "Can I let my friend drive my car without a license?"
- "What happens if I do that?"
- "Is modifying a car allowed?"
- "Who is behind this project?"
- "ولو كنت غلطان؟"

These test cases targeted specific subsystems:

- procedural retrieval;
- follow-up interpretation;
- ambiguity handling;
- legal consequence extraction;
- meta/system routing;
- Arabic follow-up understanding.

### 3. Edge Cases

The system was explicitly tested against:

- vague follow-ups with pronouns;
- ambiguous penalty questions lacking a concrete violation;
- broad procedural questions without scenario anchoring;
- Arabic conversational shorthand;
- answer formatting drift into long paragraphs;
- suggested-question mismatch with user language;
- suggestion drift unrelated to the user’s topic.

### 4. Evaluation Method

Evaluation was primarily performed through:

- response correctness against retrieved source meaning;
- routing correctness through returned metadata fields;
- multi-turn coherence in conversational scenarios;
- stability of references and formatting;
- live regression verification after code changes.

The backend’s structured response metadata played a central role in evaluation. Fields such as:

- `intent`
- `intent_detail`
- `answer_intent`
- `used_rag`
- `needs_clarification`
- `followup_topic`
- `followup_aspect`
- `rewritten_query`

allowed the system to be assessed not only by answer surface form but by its internal decision path.

## IV. Results and Analysis

### 1. System Performance

The final DALIL system achieved a strong level of functional reliability for the implemented scope. It supports:

- road-safety question answering over a curated Saudi corpus;
- bilingual interaction in English and Arabic;
- follow-up interpretation across multiple turns;
- deterministic handling of meta/system questions;
- structured citation rendering;
- streaming response generation.

The use of vector retrieval plus multi-stage filtering improved answer precision compared with direct free-form generation. The system also became substantially more robust after follow-up-specific query rewrites were introduced.

### 2. Accuracy and Reliability

Reliability in DALIL comes primarily from architecture rather than from model fine-tuning. The system reduces error by:

- grounding answers in retrieved documents;
- restricting the LLM through prompt policy;
- preventing unrelated chat history from acting as evidence;
- explicitly preferring law for legal consequences and duties;
- asking for clarification when the question is underspecified.

This architecture improved performance in scenarios where a naive chatbot might hallucinate or overgeneralize, especially in:

- accident-related consequence questions;
- unlicensed-driver legal questions;
- roundabout follow-ups;
- mixed-language conversational use.

### 3. Strengths

The principal strengths of the implemented system are:

- strong source governance;
- rich metadata design;
- hybrid deterministic-plus-generative architecture;
- multi-stage retrieval rather than single-pass nearest-neighbor use;
- explicit follow-up handling;
- bilingual interface and bilingual answer behavior;
- readable structured output design with separate citations.

Another major strength is the separation between:

- application facts,
- conversational context,
- and documentary evidence.

This separation improves interpretability and reduces failure modes.

### 4. Limitations

Despite its strengths, the project has real limitations:

- the knowledge base is limited to three source documents;
- conversation memory uses only a short recent-message window;
- intent detection is rule-based rather than learned;
- suggested questions are fixed rather than generated adaptively;
- the system does not include formal quantitative benchmarking against baseline QA systems;
- no ablation studies were implemented for retrieval-stage components.

In addition, the vector-search pipeline uses FAISS distance-based retrieval through LangChain without a custom similarity calibration layer, which means retrieval behavior is effective but not deeply analytically tuned.

## V. Discussion

The DALIL project demonstrates that a domain-specific conversational assistant can be made meaningfully more reliable by combining structured preprocessing, retrieval grounding, deterministic routing, and careful interface control. A major lesson from building the system was that generation quality alone is not enough. The most difficult problems arose not from language fluency but from control:

- deciding when to retrieve;
- deciding when not to retrieve;
- deciding when a follow-up was truly related;
- deciding when to ask for clarification rather than answer prematurely.

### 1. Follow-Up Handling Issues

Follow-up behavior was one of the most challenging aspects of the project. Early versions sometimes failed in cases such as:

- accident follow-up questions with vague wording;
- Arabic short follow-ups;
- questions referring to fault or liability using colloquial language;
- cases where the previous assistant answer accidentally influenced topic extraction more than the previous user question.

These issues were gradually resolved through:

- better follow-up detection heuristics;
- more explicit topic and aspect extraction;
- topic-specific rewrite templates;
- user-message prioritization in conversation analysis.

### 2. Routing Mistakes

Another source of complexity was routing error. If a question that should have been treated as a follow-up was sent through a generic path, response quality dropped sharply. Similarly, if a meta question was sent through retrieval, the answer could become noisy or needlessly expensive.

The intent router became increasingly important as the system matured. It was not just a convenience layer; it became the backbone of reliable system behavior.

### 3. Retrieval Improvements

The project also showed that basic semantic retrieval was insufficient on its own for legally precise QA. Improvements such as:

- query expansion;
- focused document filtering;
- reranking;
- role-aware matching

were necessary to improve relevance. This was especially visible in questions involving:

- Article 77;
- vehicle modification;
- accident liability;
- roundabout priority.

### 4. Interface Insights

Significant insights also emerged from frontend behavior:

- answer readability improved substantially when long paragraphs were replaced with direct answers and bullet sections;
- suggested questions needed to be fixed and controlled rather than dynamically drifting;
- language switching required synchronization not just in labels, but also in suggestions and branding elements.

These findings show that a strong RAG system depends as much on presentation-layer discipline as on retrieval quality.

## VI. Conclusion & Future Work

DALIL represents a complete end-to-end AI-powered road safety assistant built around retrieval-augmented generation, deterministic routing, bilingual interaction, and document-grounded response generation. The project’s final contribution is not merely a chatbot interface but a structured conversational AI system that operationalizes Saudi road-safety references in a controlled, explainable, and user-accessible way.

The principal contributions of the implemented system are:

- construction of a structured Saudi road-safety knowledge base from heterogeneous PDFs;
- semantic indexing of that corpus with Gemini embeddings and FAISS;
- development of an intent-aware FastAPI RAG backend;
- implementation of multi-turn follow-up interpretation with topic/aspect rewriting;
- integration of static meta capabilities for project-level questions;
- creation of a bilingual React chat frontend with streaming responses and structured answer rendering.

From a practical standpoint, the project demonstrates that domain-constrained generative AI systems can be significantly strengthened when retrieval, routing, prompting, and UI logic are engineered as a coordinated whole rather than as isolated components.

Future work could extend DALIL in several important directions:

- **better embeddings**
  - evaluate alternative embedding models and retrieval calibration strategies for improved legal/procedural separation
- **more data**
  - expand the corpus with additional Saudi traffic regulations, public safety advisories, signage manuals, and multilingual source material
- **smarter intent detection**
  - replace or augment the rule-based router with a learned intent classifier while preserving deterministic safety controls
- **longer and richer memory**
  - incorporate more structured conversational state management for extended multi-turn sessions
- **formal evaluation**
  - introduce benchmark datasets, retrieval evaluation metrics, and ablation studies for follow-up rewriting and reranking stages
- **adaptive suggestions**
  - evolve suggestion generation from a fixed list into topic-aware but still policy-controlled recommendations

In summary, DALIL has matured into a well-engineered local AI system whose design reflects real implementation choices, iterative debugging, and an increasingly disciplined treatment of grounded generation. It stands as a practical example of how generative AI can be adapted for a safety-sensitive domain without surrendering control to unconstrained model behavior.
