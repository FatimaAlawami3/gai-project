# Report-Ready Experiment and Results Sections

## IV. Experimental Design

To satisfy the course requirement for baseline comparison while preserving the integrity of the original DALIL implementation, we conducted an embedding-level comparative experiment between two retrieval configurations:

- `gemini-embedding-001` with `3072`-dimensional embeddings
- `text-multilingual-embedding-002` with `768`-dimensional embeddings

The comparison was intentionally designed as a controlled ablation in which the embedding model was the only changed variable. All other system components were kept constant, including:

- the same PDF corpus (`Traffic Law.pdf`, `Theoretical Driving Handbook Trainee-Moroor.pdf`, and `101 EN.pdf`)
- the same JSON knowledge base (`saudi_road_safety_kb.json`)
- the same chunking strategy
- the same metadata schema
- the same LangChain orchestration layer
- the same prompt templates and grounding policy
- the same LLM (`gemini-3.1-pro-preview`)
- the same retrieval parameterization (`top_k = 3` during the comparison run)

To prevent contamination of the original DALIL setup, the two embedding configurations were isolated into separate experiment folders under `comparison_versions/`, with independent settings files and separate FAISS indices:

- `comparison_versions/gemini-embedding-001/settings.env`
- `comparison_versions/text-multilingual-embedding-002/settings.env`

The corresponding vector stores were stored independently as:

- `data/vector_store_gemini_embedding_001`
- `data/vector_store_text_multilingual_embedding_002`

This separation ensured that the baseline comparison did not overwrite or distort the original Gemini-based configuration already used in the main project.

For the multilingual baseline, a new FAISS index was built by re-embedding all `619` chunks from the shared knowledge base using `text-multilingual-embedding-002`. The Gemini comparison branch reused the existing Gemini-indexed corpus, copied into its own isolated vector-store directory for fair experimental management.

The frontend was also extended to support interactive embedding-profile switching so that the same user interface could query the Gemini and multilingual retrieval configurations without changing the rest of the application state. This allowed the comparison to be run through one shared interface while preserving identical prompts, history behavior, and formatting logic.

### Evaluation Procedure

A focused qualitative evaluation set was selected to cover the major behaviors of DALIL:

- English procedural questions
- English legal rule questions
- Arabic legal rule questions
- Arabic procedural questions
- English follow-up handling
- Arabic follow-up handling

The evaluation prompts included:

- accident-procedure queries
- roundabout-procedure and roundabout-priority queries
- unlicensed-driver legal questions
- phone-use legal questions in English and Arabic
- informal Arabic accident-liability follow-up questions written in short or colloquial form
- vehicle-modification legal questions
- underspecified Arabic penalty questions used to test clarification logic
- out-of-scope questions
- meta/capability questions

The primary evaluation criteria were:

- response correctness
- response completeness
- follow-up continuity
- citation usefulness
- routing stability where applicable

This evaluation was intentionally aligned with the project’s real risk profile. Since DALIL is a retrieval-grounded chatbot rather than a train-from-scratch generative model, the most important indicators were not loss curves or optimization statistics, but retrieval adequacy, answer faithfulness, and multi-turn robustness.

## V. Results and Analysis

### Comparative Findings

The side-by-side evaluation showed that both embedding configurations were operational and capable of answering the tested road-safety questions without falling back to unsupported-response mode. Across the evaluated prompt set, both configurations successfully handled:

- direct procedural questions
- legal-rule questions
- English follow-up questions
- Arabic follow-up questions

However, the qualitative behavior of the two embeddings was not identical.

#### 1. Gemini Embedding 001

`gemini-embedding-001` consistently produced the more complete answers in the evaluated accident and Arabic safety scenarios. In particular:

- for the English accident query, the Gemini configuration returned a broader answer covering both injury-related and damage-only accident branches, with `3` sources attached;
- for the English phone-use query, the Gemini configuration returned a direct rule answer with the hands-free condition and violation framing;
- for the Arabic phone-use query, the Gemini configuration returned a more developed answer with clearer rule framing and stronger detail coverage;
- for the underspecified Arabic penalty prompt `وش العقوبة؟`, Gemini correctly asked the user to clarify the intended violation instead of answering too broadly.

These outputs suggest that the Gemini embedding index retrieved broader and more contextually complete evidence for the tested safety-critical procedural questions.

#### 2. Text Multilingual Embedding 002

`text-multilingual-embedding-002` remained competitive, especially on:

- roundabout procedure
- roundabout priority follow-up
- unlicensed-driver rule and penalty questions

Its outputs were generally concise and correct, and it handled both English and Arabic retrieval successfully. However, in several cases the multilingual baseline returned shorter and less comprehensive answers. For example:

- on the English accident query, the multilingual configuration emphasized the Najm workflow and accident documentation, but the returned answer was less complete than the Gemini version because it did not foreground the injuries branch with the same clarity;
- on the English phone-use query, the multilingual configuration remained correct but was typically shorter than the Gemini variant;
- on the Arabic phone-use query, the multilingual configuration produced a much shorter answer with limited supporting detail compared with the Gemini variant;
- on the Arabic accident procedure query, the multilingual answer remained correct but was operationally narrower;
- on the scored Arabic liability follow-up wording, the multilingual configuration provided a fuller consequence-oriented explanation than Gemini, although its citation support was weaker for the breadth of that answer;
- on the underspecified Arabic penalty prompt `وش العقوبة؟`, the multilingual configuration answered with a broad penalty summary instead of requesting clarification.

This suggests that the multilingual model preserved broad bilingual capability but was somewhat less effective at retrieving the richest set of supporting chunks for more complex procedural questions.

### Follow-Up Behavior

A particularly important finding is that both embedding configurations successfully preserved the improved follow-up behavior added during development. In the evaluated follow-up set:

- `What about priority?` after a roundabout question was correctly interpreted as a roundabout-priority follow-up by both models;
- `What happens if I do that?` after the unlicensed-driver question was correctly resolved to the legal penalty/liability interpretation by both models;
- informal Arabic accident-responsibility follow-ups after the Arabic accident question were also handled successfully by both configurations, although the multilingual answer was more complete on the scored wording while Gemini remained correctly grounded.

These were evaluated as examples of a broader follow-up category rather than as exact trigger phrases hardcoded into the experiment design.

This result indicates that the follow-up improvements were primarily driven by the routing and query-rewriting layers rather than by the embedding model alone. In other words, embedding choice affected retrieval richness, but the conversational repair logic remained stable across both variants.

### Clarification and Non-RAG Stability

The later evaluation rows also showed that not every tested behavior was equally sensitive to embedding choice.

- On the clarification test, Gemini behaved better because it preserved the intended clarification-first policy for an underspecified Arabic penalty prompt, whereas the multilingual run answered too broadly.
- On the out-of-scope question (`What is the weather today?`), both configurations behaved consistently because the question was outside the road-safety RAG path.
- On meta/capability questions such as `Who is behind this project?` and `Who are you?`, both configurations remained stable because these responses are handled by the router and fixed system metadata rather than by vector retrieval.

This distinction is important for interpreting the experiment fairly: some rows primarily measure retrieval quality, while others measure whether routing, clarification, and system-level control logic remain stable under different embedding profiles.

It is also important to note that the comparison assumes the later retrieval fix for phone-use questions is present. After the backend was updated to model phone use while driving as a dedicated topic with topic-aware expansion, valid English and Arabic phone-use questions stopped dropping into fallback and became suitable comparison items.

### Overall Interpretation

From the observed evaluation set, `gemini-embedding-001` emerged as the stronger final choice for DALIL because it provided:

- more complete accident-related responses
- stronger Arabic detail retention
- broader evidence coverage on safety-critical procedural questions
- more reliable clarification behavior on the underspecified Arabic penalty test

At the same time, `text-multilingual-embedding-002` was a reasonable baseline because:

- it remained correct on the tested legal-rule scenarios
- it handled bilingual retrieval successfully
- it preserved follow-up correctness
- it produced the fuller answer on the scored Arabic liability follow-up prompt
- it offered a meaningful comparison point within the same Vertex AI ecosystem

### Summary Table

| Scenario | Gemini Embedding 001 | Text Multilingual Embedding 002 |
|---|---|---|
| English accident procedure | More complete, covered injuries and damage-only branches | Correct but narrower, emphasized Najm/documentation workflow |
| English roundabout procedure | Strong | Strong |
| English unlicensed-driver rule | Strong | Strong |
| English phone-use rule | Strong | Correct but shorter |
| Arabic phone-use rule | More detailed | Correct but shorter |
| Arabic accident procedure | More complete | Correct but more concise |
| Arabic liability follow-up | Correct but shorter | More complete, but citation support is weaker |
| English follow-up continuity | Strong | Strong |
| Arabic follow-up continuity | Strong | Strong |
| Clarification logic (`وش العقوبة؟`) | Correctly asks for clarification | Answers too broadly instead of clarifying |
| Out-of-scope stability | Stable | Stable |
| Meta/capability stability | Stable | Stable |

### Conclusion of Comparison

Based on the controlled embedding comparison, the project retained `gemini-embedding-001` as the primary embedding model for DALIL. The multilingual baseline demonstrated that the system architecture remains functional under a different bilingual embedding configuration, but the Gemini setup produced more complete and better-supported answers on the tested safety-critical scenarios. This made it the more suitable production configuration for the final version of the chatbot.
