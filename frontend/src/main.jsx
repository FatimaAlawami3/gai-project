import React, { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL =
  import.meta.env.VITE_DALIL_API_BASE_URL || "http://127.0.0.1:8011";
const API_STREAM_URL = `${API_BASE_URL}/ask/stream`;
const HISTORY_LIMIT = 6;
const EMBEDDING_PROFILES = [
  {
    id: "gemini-embedding-001",
    shortLabel: "Gemini",
    label: "Gemini Embedding 001",
  },
  {
    id: "text-multilingual-embedding-002",
    shortLabel: "Multilingual",
    label: "Text Multilingual Embedding 002",
  },
];

const starterQuestionsByLanguage = {
  en: [
    "What should I do after a traffic accident?",
    "Is it allowed to use a phone while driving?",
    "What happens if I drive without a license?",
    "Can I let someone else drive my car?",
    "Is modifying a car allowed?",
    "What should a driver do when approaching a roundabout?",
    "Who is behind this project?",
  ],
  ar: [
    "ماذا يجب علي فعله بعد وقوع حادث مروري؟",
    "هل يسمح باستخدام الهاتف أثناء القيادة؟",
    "ماذا يحدث إذا قدت بدون رخصة؟",
    "هل يمكنني السماح لشخص آخر بقيادة سيارتي؟",
    "هل يسمح بتعديل السيارة؟",
    "ما الذي يجب على السائق فعله عند الاقتراب من الدوار؟",
    "من وراء هذا المشروع؟",
  ],
};

function getStarterQuestions(language = "en") {
  return starterQuestionsByLanguage[language] || starterQuestionsByLanguage.en;
}

const uiCopyByLanguage = {
  en: {
    brandName: "DALIL",
    welcome:
      "Hello! I am DALIL, your Road Safety Guide AI Chatbot. Ask me about Saudi road safety, traffic rules, parking, accidents, roundabouts, or this project.",
    intro: "Your AI assistant for road safety and traffic regulations in Saudi Arabia",
    subtitle: "Road Safety Guide - AI Chatbot",
    placeholder: "Ask about roundabouts, parking, accidents, violations...",
    historyNote:
      "Ask about traffic rules, road signs, parking, accidents, or safe driving.",
    resetLabel: "Reset chat",
    sendLabel: "Send message",
    embeddingLabel: "Embedding model",
  },
  ar: {
    brandName: "دليل",
    welcome:
      "مرحباً! أنا دليل، روبوت دردشة ذكي للإرشاد في السلامة المرورية. اسألني عن السلامة المرورية في السعودية، وأنظمة المرور، والوقوف، والحوادث، والدوارات، أو هذا المشروع.",
    intro: "مساعدك الذكي للسلامة المرورية وأنظمة المرور في المملكة العربية السعودية",
    subtitle: "دليل - روبوت دردشة للسلامة المرورية",
    placeholder: "اسأل عن الدوارات، الوقوف، الحوادث، المخالفات...",
    historyNote:
      "اسأل عن أنظمة المرور، والإشارات، والوقوف، والحوادث، أو القيادة الآمنة.",
    resetLabel: "إعادة تعيين المحادثة",
    sendLabel: "إرسال الرسالة",
    embeddingLabel: "نموذج التضمين",
  },
};

function getUiCopy(language = "en") {
  return uiCopyByLanguage[language] || uiCopyByLanguage.en;
}

function detectArabic(text = "") {
  return /[\u0600-\u06FF]/.test(text);
}

function messageDirection(text = "") {
  return detectArabic(text) ? "rtl" : "ltr";
}

function stripInlineSources(text = "") {
  const sourceHeading =
    /(?:^|\n)\s*(?:[*_`#>\-\s]*)?(?:sources|references|المصادر|المراجع)(?:[*_`\s]*)?:?\s*(?:\n|$)/im;
  const match = sourceHeading.exec(text);
  if (!match) return text;
  return text.slice(0, match.index).trim();
}

function cleanAnswerText(text = "") {
  return stripInlineSources(text)
    .replace(/\r\n/g, "\n")
    .replace(
      /(?:^|\n)\s*(?:This answer is based on the uploaded Saudi traffic sources\.?|هذه الإجابة مبنية على مصادر المرور السعودية المرفوعة\.?|تعتمد هذه الإجابة على مصادر المرور السعودية المرفوعة\.?)\s*/gi,
      "\n"
    )
    .replace(/[ \t]+\*\s+(?=(?:\*\*)|[A-Za-z0-9\u0600-\u06FF])/g, "\n- ")
    .replace(/^\s*\*\s+/gm, "- ")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function toChatHistory(messages) {
  return messages
    .filter(
      (message) =>
        (message.role === "user" || message.role === "assistant") &&
        !message.excludeFromHistory
    )
    .slice(-HISTORY_LIMIT)
    .map((message) => ({
      role: message.role,
      content: message.content,
    }));
}

function formatError(error) {
  if (error?.message) return error.message;
  return "Something went wrong while contacting the API.";
}

function App() {
  const [preferredUiLanguage, setPreferredUiLanguage] = useState("en");
  const [preferredEmbeddingProfile, setPreferredEmbeddingProfile] = useState(
    EMBEDDING_PROFILES[0].id
  );
  const [messages, setMessages] = useState([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      content: getUiCopy("en").welcome,
      language: "en",
      suggestedQuestions: getStarterQuestions("en"),
      excludeFromHistory: true,
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  function handleLanguageChange(language) {
    setPreferredUiLanguage(language);
    setMessages((current) => {
      const hasUserMessages = current.some((message) => message.role === "user");
      if (hasUserMessages) return current;

      return current.map((message, index) =>
        index === 0 && message.role === "assistant" && message.excludeFromHistory
          ? {
              ...message,
              content: getUiCopy(language).welcome,
              language,
              suggestedQuestions: getStarterQuestions(language),
            }
          : message
      );
    });
    window.setTimeout(() => inputRef.current?.focus(), 50);
  }

  async function askQuestion(questionText) {
    const question = questionText.trim();
    if (!question || isLoading) return;

    const assistantMessageId = crypto.randomUUID();
    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      language: detectArabic(question) ? "ar" : "en",
    };

    const requestHistory = toChatHistory(messages);
    const assistantMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      rawAnswer: "",
      language: userMessage.language,
      sources: [],
      isStreaming: true,
    };

    const updateAssistantMessage = (updater) => {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessageId
            ? { ...message, ...updater(message) }
            : message
        )
      );
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput("");
    setError("");
    setIsLoading(true);

    try {
      const response = await fetch(API_STREAM_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          top_k: 3,
          chat_history: requestHistory,
          embedding_profile: preferredEmbeddingProfile,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with ${response.status}`);
      }
      if (!response.body) {
        throw new Error("This browser does not support streaming responses.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let rawAnswer = "";
      let receivedDone = false;

      const handleEvent = (event) => {
        if (event.type === "error") {
          throw new Error(event.detail || "Streaming request failed.");
        }

        if (event.type === "metadata") {
          updateAssistantMessage(() => ({
            language: event.language,
            intent: event.intent,
            intentDetail: event.intent_detail,
            usedRag: event.used_rag,
            isFallback: event.is_fallback,
            model: event.model,
            embeddingModel: event.embedding_model,
            embeddingDimensions: event.embedding_dimensions,
            embeddingProfile: event.embedding_profile,
            rewrittenQuery: event.rewritten_query,
            followupTopic: event.followup_topic,
            followupAspect: event.followup_aspect,
          }));
          return;
        }

        if (event.type === "chunk") {
          rawAnswer += event.content || "";
          updateAssistantMessage(() => ({
            content: cleanAnswerText(rawAnswer),
            rawAnswer,
          }));
          return;
        }

        if (event.type === "done") {
          receivedDone = true;
          rawAnswer = event.answer || rawAnswer;
          updateAssistantMessage(() => ({
            content: cleanAnswerText(rawAnswer),
            rawAnswer,
            language: event.language,
            sources: event.sources || [],
            intent: event.intent,
            intentDetail: event.intent_detail,
            usedRag: event.used_rag,
            isFallback: event.is_fallback,
            model: event.model,
            embeddingModel: event.embedding_model,
            embeddingDimensions: event.embedding_dimensions,
            embeddingProfile: event.embedding_profile,
            rewrittenQuery: event.rewritten_query,
            followupTopic: event.followup_topic,
            followupAspect: event.followup_aspect,
            isStreaming: false,
          }));
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          handleEvent(JSON.parse(line));
        }
      }

      buffer += decoder.decode();
      if (buffer.trim()) {
        handleEvent(JSON.parse(buffer));
      }

      if (!receivedDone) {
        updateAssistantMessage(() => ({ isStreaming: false }));
      }
    } catch (err) {
      const errorMessage = formatError(err);
      setError(errorMessage);
        updateAssistantMessage(() => ({
          content:
          `I could not reach the DALIL API. Please make sure the FastAPI server is running on ${API_BASE_URL}.`,
          language: "en",
          isFallback: true,
          isStreaming: false,
          excludeFromHistory: true,
      }));
    } finally {
      setIsLoading(false);
      window.setTimeout(() => inputRef.current?.focus(), 50);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    askQuestion(input);
  }

  function handleComposerKeyDown(event) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }

    event.preventDefault();
    askQuestion(input);
  }

  function resetChat() {
    setMessages([
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content: getUiCopy(preferredUiLanguage).welcome,
        language: preferredUiLanguage,
        suggestedQuestions: getStarterQuestions(preferredUiLanguage),
        excludeFromHistory: true,
      },
    ]);
    setError("");
    setInput("");
    inputRef.current?.focus();
  }

  const uiLanguage = preferredUiLanguage;
  const uiCopy = getUiCopy(uiLanguage);
  const latestSuggestions = getStarterQuestions(uiLanguage);

  return (
    <main className="app-shell">
      <section className="left-panel">
        <div className="brand-content">
          <div className="brand-logo-card">
            <img src="/dalil-logo.png" alt="DALIL Road Safety Guide AI Chatbot" />
            <h1 className="sr-only">DALIL Road Safety Guide AI Chatbot</h1>
          </div>

          <p className="intro">
            {uiCopy.intro}
          </p>
        </div>
      </section>

      <section className="chat-panel" aria-label="DALIL chat" dir={uiLanguage === "ar" ? "rtl" : "ltr"}>
        <header className="chat-header">
          <div className="chat-title">
            <span className="assistant-avatar" aria-hidden="true">
              <img src="/dalil-avatar.png" alt="" />
            </span>
            <div>
              <h2>{uiCopy.brandName}</h2>
              <p>{uiCopy.subtitle}</p>
            </div>
          </div>
          <div className="header-actions">
            <div className="embedding-toggle">
              <span className="control-label">{uiCopy.embeddingLabel}</span>
              <div className="embedding-toggle-buttons" role="group" aria-label={uiCopy.embeddingLabel}>
                {EMBEDDING_PROFILES.map((profile) => (
                  <button
                    key={profile.id}
                    className={`embedding-button ${
                      preferredEmbeddingProfile === profile.id ? "active" : ""
                    }`}
                    type="button"
                    onClick={() => setPreferredEmbeddingProfile(profile.id)}
                    aria-pressed={preferredEmbeddingProfile === profile.id}
                    disabled={isLoading}
                    title={profile.label}
                  >
                    {profile.shortLabel}
                  </button>
                ))}
              </div>
            </div>
            <div className="language-toggle" role="group" aria-label="Language switcher">
              <button
                className={`language-button ${uiLanguage === "en" ? "active" : ""}`}
                type="button"
                onClick={() => handleLanguageChange("en")}
                aria-pressed={uiLanguage === "en"}
              >
                English
              </button>
              <button
                className={`language-button ${uiLanguage === "ar" ? "active" : ""}`}
                type="button"
                onClick={() => handleLanguageChange("ar")}
                aria-pressed={uiLanguage === "ar"}
              >
                العربية
              </button>
            </div>
            <button className="icon-button" type="button" onClick={resetChat} aria-label={uiCopy.resetLabel} title={uiCopy.resetLabel}>
              ↻
            </button>
          </div>
        </header>

        <div className="messages" aria-live="polite">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          {isLoading && !messages.some((message) => message.isStreaming) && (
            <div className="message-row assistant-row">
              <div className="message-bubble assistant-bubble loading-bubble">
                <span className="pulse-dot" />
                <span className="pulse-dot" />
                <span className="pulse-dot" />
              </div>
            </div>
          )}
        </div>

        <div className="suggestions">
          {latestSuggestions.map((question) => (
            <button
              key={question}
              type="button"
              onClick={() => askQuestion(question)}
              disabled={isLoading}
              dir={messageDirection(question)}
            >
              {question}
            </button>
          ))}
        </div>

        {error && <p className="error-banner">{error}</p>}

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder={uiCopy.placeholder}
            rows={1}
            dir={input ? messageDirection(input) : uiLanguage === "ar" ? "rtl" : "ltr"}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()} aria-label={uiCopy.sendLabel}>
            <span aria-hidden="true">➜</span>
          </button>
        </form>

        <p className="history-note">
          {uiCopy.historyNote}
        </p>
      </section>
    </main>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const direction = messageDirection(message.content);
  const hasSources = message.sources?.length > 0;
  const isArabic = message.language === "ar" || direction === "rtl";
  const embeddingLabel = EMBEDDING_PROFILES.find(
    (profile) => profile.id === message.embeddingProfile
  )?.shortLabel;

  return (
    <article className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      {!isUser && (
        <div className="row-avatar assistant-row-avatar" aria-hidden="true">
          <img src="/dalil-avatar.png" alt="" />
        </div>
      )}
      <div
        className={`message-bubble ${isUser ? "user-bubble" : "assistant-bubble"} ${
          isArabic ? "arabic-text" : ""
        }`}
        dir={direction}
      >
        <div className="bubble-label">{isUser ? "You" : "DALIL"}</div>
        {!isUser && embeddingLabel ? (
          <div className="bubble-meta">{embeddingLabel}</div>
        ) : null}
        {message.content ? (
          <FormattedText text={message.content} />
        ) : message.isStreaming ? (
          <div className="streaming-dots" aria-label="DALIL is writing">
            <span className="pulse-dot" />
            <span className="pulse-dot" />
            <span className="pulse-dot" />
          </div>
        ) : null}

        {hasSources && <Sources sources={message.sources} />}
      </div>
      {isUser && (
        <div className="row-avatar user-row-avatar" aria-hidden="true">
          <span />
        </div>
      )}
    </article>
  );
}

function FormattedText({ text }) {
  const blocks = text.split(/\n{2,}/);

  return (
    <div className="message-text">
      {blocks.map((block, blockIndex) => {
        const lines = block.split("\n").filter((line) => line.trim().length > 0);
        const bulletLines = lines.filter((line) => /^\s*(?:\*|-|•)\s+/.test(line));
        const numberedLines = lines.filter((line) => /^\s*\d+[.)]\s+/.test(line));
        const headingIndexes = lines
          .map((line, index) =>
            /:\s*$/.test(line.trim()) && index < lines.length - 1 ? index : -1
          )
          .filter((index) => index >= 0);

        if (numberedLines.length === lines.length && lines.length > 0) {
          return (
            <ul className="answer-list" key={`block-${blockIndex}`}>
              {lines.map((line, lineIndex) => (
                <li key={`line-${lineIndex}`}>
                  <InlineMarkdown text={line.replace(/^\s*\d+[.)]\s+/, "")} />
                </li>
              ))}
            </ul>
          );
        }

        if (bulletLines.length === lines.length && lines.length > 0) {
          return (
            <ul className="answer-list" key={`block-${blockIndex}`}>
              {lines.map((line, lineIndex) => (
                <li key={`line-${lineIndex}`}>
                  <InlineMarkdown text={line.replace(/^\s*(?:\*|-|•)\s+/, "")} />
                </li>
              ))}
            </ul>
          );
        }

        if (
          lines.length >= 3 &&
          /:\s*$/.test(lines[0].trim()) &&
          lines.slice(1).every((line) => !/:\s*$/.test(line.trim()))
        ) {
          return (
            <div className="answer-sections" key={`block-${blockIndex}`}>
              <p>
                <InlineMarkdown text={lines[0]} />
              </p>
              <ul className="answer-list">
                {lines.slice(1).map((line, lineIndex) => (
                  <li key={`line-${lineIndex}`}>
                    <InlineMarkdown text={line.replace(/^\s*(?:\*|-|•|\d+[.)])\s+/, "")} />
                  </li>
                ))}
              </ul>
            </div>
          );
        }

        if (headingIndexes.length > 0) {
          const sections = [];
          let sectionStart = 0;

          for (const headingIndex of headingIndexes) {
            if (headingIndex > sectionStart) {
              sections.push({
                type: "paragraph",
                lines: lines.slice(sectionStart, headingIndex),
              });
            }

            let nextBoundary = lines.length;
            for (const candidate of headingIndexes) {
              if (candidate > headingIndex) {
                nextBoundary = candidate;
                break;
              }
            }

            sections.push({
              type: "section",
              heading: lines[headingIndex],
              lines: lines.slice(headingIndex + 1, nextBoundary),
            });
            sectionStart = nextBoundary;
          }

          if (sectionStart < lines.length) {
            sections.push({
              type: "paragraph",
              lines: lines.slice(sectionStart),
            });
          }

          return (
            <div className="answer-sections" key={`block-${blockIndex}`}>
              {sections.map((section, sectionIndex) => {
                if (section.type === "paragraph") {
                  return (
                    <p key={`section-${sectionIndex}`}>
                      {section.lines.map((line, lineIndex) => (
                        <React.Fragment key={`line-${lineIndex}`}>
                          {lineIndex > 0 && <br />}
                          <InlineMarkdown text={line} />
                        </React.Fragment>
                      ))}
                    </p>
                  );
                }

                return (
                  <section className="answer-section" key={`section-${sectionIndex}`}>
                    <p className="answer-section-title">
                      <InlineMarkdown text={section.heading} />
                    </p>
                    <ul className="answer-list answer-sublist">
                      {section.lines.map((line, lineIndex) => (
                        <li key={`line-${lineIndex}`}>
                          <InlineMarkdown text={line.replace(/^\s*(?:\*|-|•)\s+/, "")} />
                        </li>
                      ))}
                    </ul>
                  </section>
                );
              })}
            </div>
          );
        }

        return (
          <p key={`block-${blockIndex}`}>
            {lines.map((line, lineIndex) => (
              <React.Fragment key={`line-${lineIndex}`}>
                {lineIndex > 0 && <br />}
                <InlineMarkdown text={line} />
              </React.Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}

function InlineMarkdown({ text }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);

  return (
    <>
      {parts.map((part, index) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return <strong key={index}>{part.slice(2, -2)}</strong>;
        }
        return <React.Fragment key={index}>{part}</React.Fragment>;
      })}
    </>
  );
}

function uniqueSources(sources = []) {
  const seen = new Set();

  return sources.filter((source) => {
    const key = [
      source.display_citation || source.citation || "",
      source.section_title || "",
      source.source_file || "",
      source.page_reference || "",
      source.official_reference || "",
    ]
      .join("|")
      .toLowerCase();

    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function Sources({ sources }) {
  const references = uniqueSources(sources);

  return (
    <div className="sources-block">
      <p className="sources-title">References</p>
      <div className="source-list">
        {references.map((source) => (
          <article
            className="source-item"
            key={`${source.display_citation || source.citation}-${source.section_title}`}
          >
            <strong>{source.display_citation || source.citation}</strong>
            <span>{source.section_title}</span>
          </article>
        ))}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
