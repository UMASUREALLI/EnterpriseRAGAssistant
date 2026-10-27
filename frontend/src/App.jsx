
import { useRef, useState } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";
const ASK_URL = `${API_BASE_URL}/ask`;
const UPLOAD_URL = `${API_BASE_URL}/upload`;

function App() {
  // Reference to the file input
  const fileInputRef = useRef(null);

  // Chat state
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Document upload state
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");

  // Ask a question
  async function handleSubmit(event) {
    event.preventDefault();

    const text = question.trim();

    if (!text || loading) return;

    setMessages((previous) => [
      ...previous,
      { role: "user", text },
    ]);

    setQuestion("");
    setError("");
    setLoading(true);

    try {
      const response = await fetch(ASK_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: text }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `API request failed: ${response.status}`,
        );
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          text: data.answer || "No answer was returned.",
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setError(
        `${err.message}. Check that your FastAPI backend is running.`,
      );
    } finally {
      setLoading(false);
    }
  }

  // Upload and index a document
  async function handleUpload(event) {
    event.preventDefault();

    if (!selectedFile || uploading) return;

    const allowedExtensions = [".pdf", ".txt", ".docx"];
    const filename = selectedFile.name.toLowerCase();

    const isSupported = allowedExtensions.some((extension) =>
      filename.endsWith(extension),
    );

    if (!isSupported) {
      setUploadError("Choose a PDF, TXT, or DOCX file.");
      setUploadMessage("");
      return;
    }

    if (selectedFile.size === 0) {
      setUploadError("The selected file is empty.");
      setUploadMessage("");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setUploadError("The file must not exceed 10 MB.");
      setUploadMessage("");
      return;
    }

    setUploading(true);
    setUploadError("");
    setUploadMessage("");

    try {
      const formData = new FormData();

      formData.append("file", selectedFile);

      const response = await fetch(UPLOAD_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Upload failed: ${response.status}`,
        );
      }

      setUploadMessage(
        `${data.filename} uploaded and indexed successfully. ` +
          `${data.chunks_indexed} chunks indexed.`,
      );

      // Clear the selected file in React state.
      setSelectedFile(null);

      // Clear the file input safely using the ref.
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setUploadError(
        `${err.message}. Check that your FastAPI backend is running.`,
      );
    } finally {
      setUploading(false);
    }
  }

  return (
    <main className="app">
      <header className="header">
        <div className="logo">R</div>

        <div>
          <h1>Knowledge Assistant</h1>
          <p>Enterprise RAG · Company documents</p>
        </div>

        <span className="status">● Ready to help</span>
      </header>

      <section className="chat-panel">
        {/* Document upload */}
        <section className="upload-panel">
          <h2>Upload a document</h2>

          <p>
            Add a PDF, TXT, or DOCX file to the knowledge base.
          </p>

          <form
            className="upload-form"
            onSubmit={handleUpload}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt,.docx"
              aria-label="Choose a document to upload"
              disabled={uploading}
              onChange={(event) => {
                setSelectedFile(
                  event.target.files?.[0] || null,
                );
                setUploadMessage("");
                setUploadError("");
              }}
            />

            <button
              type="submit"
              disabled={uploading || !selectedFile}
            >
              {uploading
                ? "Uploading and indexing..."
                : "Upload document"}
            </button>
          </form>

          {uploading && (
            <p className="loading" role="status">
              Processing your document. Please wait...
            </p>
          )}

          {uploadMessage && (
            <p className="success" role="status">
              {uploadMessage}
            </p>
          )}

          {uploadError && (
            <p className="error" role="alert">
              {uploadError}
            </p>
          )}
        </section>

        {/* Welcome screen */}
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-icon">✦</div>

            <h2>How can I help you today?</h2>

            <p>
              Ask questions about company policies or documents
              that have been uploaded and indexed.
            </p>

            <div className="suggestions">
              {[
                "Who needs to approve my time off?",
                "How many annual leave days are available?",
                "How far in advance should I request leave?",
              ].map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setQuestion(item)}
                >
                  {item} <span>↗</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chat messages */}
        <div className="messages" aria-live="polite">
          {messages.map((message, index) => (
            <article
              className={`message ${message.role}`}
              key={`${message.role}-${index}`}
            >
              <strong>
                {message.role === "user"
                  ? "You"
                  : "Knowledge Assistant"}
              </strong>

              <p>{message.text}</p>

              {message.sources?.length > 0 && (
                <details className="sources">
                  <summary>
                    Sources ({message.sources.length})
                  </summary>

                  {message.sources.map((source, sourceIndex) => (
                    <div
                      className="source"
                      key={`${source.source}-${source.chunk_index}-${sourceIndex}`}
                    >
                      <span>{source.source}</span>
                      <small>
                        Chunk {source.chunk_index}
                      </small>
                    </div>
                  ))}
                </details>
              )}
            </article>
          ))}

          {loading && (
            <p className="loading" role="status">
              Searching documents and generating an answer...
            </p>
          )}

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </div>

        {/* Ask question */}
        <form
          className="chat-form"
          onSubmit={handleSubmit}
        >
          <input
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            placeholder="Ask about your documents..."
            aria-label="Your question"
            maxLength={2000}
            disabled={loading}
          />

          <button
            type="submit"
            disabled={loading || !question.trim()}
          >
            {loading ? "Thinking..." : "Send ↑"}
          </button>
        </form>

        <footer>
          Answers are based on available documents. Verify
          important decisions against official company policies.
        </footer>
      </section>
    </main>
  );
}

export default App;
