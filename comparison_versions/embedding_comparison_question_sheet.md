# Embedding Comparison Question Sheet

This sheet is designed for a fair comparison between:

- `gemini-embedding-001`
- `text-multilingual-embedding-002`

Keep the following fixed for both runs:

- same DALIL backend code
- same PDFs
- same JSON knowledge base
- same prompts
- same LLM: `gemini-3.1-pro-preview`
- same `top_k`
- same evaluation order

Only change:

- embedding model
- vector store built from that embedding model

This sheet records the original comparison set and its manual scoring workflow. Later router and retrieval fixes broadened the live system in areas such as driving-license questions, traffic signals, speed/safe-distance wording, pedestrian follow-ups, and some Arabic disambiguation cases, so the sheet should be rerun if you want a current-system scorecard rather than the original comparison snapshot.

## Scoring Rubric

Use a `1-5` scale for each criterion:

- `5`: fully correct, complete, and well-grounded
- `4`: correct with minor omissions
- `3`: partially correct or somewhat incomplete
- `2`: major omissions or weak grounding
- `1`: incorrect or unreliable

Suggested criteria:

- correctness
- completeness
- citation usefulness
- follow-up handling where applicable
- routing / stability behavior where applicable

## Evaluation Questions

| ID | Query | Type | Expected Focus |
|---|---|---|---|
| Q1 | What should I do after a traffic accident? | English procedure | Injuries vs damage-only workflow, emergency numbers, Najm handling |
| Q2 | What should a driver do when approaching a roundabout? | English procedure | Lane planning, giving way, signaling, exit behavior |
| Q3 | What about priority? | English follow-up | Correct continuation of roundabout context |
| Q4 | Can I let my friend drive my car without a license? | English legal rule | Article 77 style prohibition, responsible party, fine |
| Q5 | What happens if I do that? | English follow-up | Fine and legal liability for allowing unlicensed driving |
| Q6 | Is it allowed to use a phone while driving? | English legal rule | Prohibition, hands-free condition, violation framing |
| Q7 | هل يسمح باستخدام الهاتف أثناء القيادة؟ | Arabic legal rule | Same phone-use rule in Arabic with clear legal framing |
| Q8 | ماذا يجب علي فعله بعد وقوع حادث مروري؟ | Arabic procedure | Arabic accident workflow, injuries vs damage-only |
| Q9 | ولو كنت غلطان؟ | Arabic follow-up | Liability consequences after Arabic accident context |
| Q10 | Is modifying a car allowed? | English legal rule | Permission requirement, modification consequences |
| Q11 | Can the vehicles be moved before the accident is reported? | English multi-turn follow-up | Accident-specific follow-up continuity after Q1 |
| Q12 | Who is behind this project? | Meta question | Should remain stable across embeddings and bypass RAG |
| Q13 | وش العقوبة؟ | Clarification logic | Should trigger a clarification request when the penalty target remains unspecified |
| Q14 | What is the weather today? | Out-of-scope question | Should be handled outside the road-safety RAG path and remain stable across embeddings |
| Q15 | Who are you? | Meta / capability question | Should remain stable across embeddings and bypass RAG |

## Scoring Table

| ID | Gemini Correctness | Gemini Completeness | Gemini Citations | Gemini Notes | Multilingual Correctness | Multilingual Completeness | Multilingual Citations | Multilingual Notes |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Q1 | 5 | 5 | 5 | Covers injuries and damage-only workflow with strong detail | 4 | 4 | 4 | Correct but centered on damage-only case and less complete |
| Q2 | 5 | 5 | 5 | Clear steps, lane choice, and signaling | 5 | 5 | 5 | Same overall quality |
| Q3 | 5 | 5 | 5 | Strong roundabout follow-up continuity | 5 | 5 | 5 | Same |
| Q4 | 5 | 5 | 5 | Accurate rule, responsible party, and fine | 5 | 5 | 5 | Same |
| Q5 | 5 | 5 | 5 | Correct continuation of Article 77 consequences | 5 | 5 | 5 | Same |
| Q6 | 5 | 5 | 5 | Direct, grounded rule with hands-free condition and points | 5 | 5 | 5 | Same |
| Q7 | 5 | 5 | 5 | Strong Arabic answer with clear legal framing | 5 | 5 | 4 | Correct but supported by fewer visible references |
| Q8 | 5 | 5 | 4 | Full Arabic workflow for injuries and damage-only cases | 4 | 4 | 4 | Misses the injuries branch and is narrower |
| Q9 | 4 | 3 | 4 | Correct on liability basis, but too short and misses stronger consequence detail | 4 | 5 | 3 | More detailed legal explanation, but citation support is weak for the breadth of the answer |
| Q10 | 4 | 4 | 5 | Correct but includes extra repair-shop penalties beyond the core permission question | 4 | 5 | 4 | More complete, but includes a stray artifact and less focused citations |
| Q11 | 5 | 5 | 5 | Clear condition-based follow-up answer | 5 | 5 | 4 | Correct answer, but citation usefulness is slightly weaker |
| Q12 | 5 | 5 | - | Router-handled meta answer, not RAG-dependent | 5 | 5 | - | Same |
| Q13 | 5 | 5 | - | Correct clarification behavior for an underspecified penalty question | 2 | 2 | - | Answered a broad penalty summary instead of clarifying |
| Q14 | 5 | 5 | - | Correct out-of-scope handling | 5 | 5 | - | Same |
| Q15 | 5 | 5 | - | Correct identity / capability response | 5 | 5 | - | Same |

## Practical Notes

- For Q3, run it after Q2 in the same chat.
- For Q5, run it after Q4 in the same chat.
- For Q6 and Q7, both models should answer from retrieved sources rather than fallback after the phone-use topic fix.
- For Q11, run it after Q1 in the same chat.
- For Q13, keep the penalty target unspecified so it tests clarification behavior rather than direct answer generation.
- For Q9, run it after Q8 in the same chat.
- For Q9, use `ولو كنت غلطان؟` as the scored prompt and optionally test extra informal Arabic variants separately.
- For Q12 and Q15, the answer should not meaningfully depend on embedding choice because they are router-handled.
- For Q14, the answer should also remain essentially independent of embedding choice because it is outside the road-safety domain.
