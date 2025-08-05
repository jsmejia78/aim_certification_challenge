import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Simple error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('React Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: "1rem", 
          background: "#fef2f2", 
          color: "#dc2626", 
          borderRadius: "8px",
          margin: "1rem"
        }}>
          <h3>Something went wrong</h3>
          <p>Please refresh the page and try again.</p>
          <button 
            onClick={() => window.location.reload()}
            style={{
              padding: "0.5rem 1rem",
              background: "#dc2626",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Refresh Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default function App() {
  // State for form fields
  const [userMessage, setUserMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  // Chat conversation history
  const [conversation, setConversation] = useState([]);
  
  // Health check state
  const [health, setHealth] = useState(null);
  
  // Context popup state
  const [showContextPopup, setShowContextPopup] = useState(false);
  const [selectedContext, setSelectedContext] = useState(null);
  
  // Ref for auto-scrolling to latest message
  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    if (chatEndRef.current && chatContainerRef.current) {
      // Use a more reliable scroll method
      const scrollToBottom = () => {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
      };
      
      // Use requestAnimationFrame to ensure DOM is updated
      requestAnimationFrame(scrollToBottom);
    }
  }, [conversation]);

  // Force scroll to bottom when loading state changes
  useEffect(() => {
    if (chatEndRef.current && chatContainerRef.current) {
      const scrollToBottom = () => {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
      };
      requestAnimationFrame(scrollToBottom);
    }
  }, [loading]);

  // Health check on mount
  React.useEffect(() => {
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => setHealth(data.status === "ok" ? "🟢" : "🔴"))
      .catch(() => setHealth("🔴"));
  }, []);

  // Handle form submit
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userMessage.trim()) return;
    
    setLoading(true);
    setError("");
    
    // Add user message to conversation
    const userMsg = { type: "user", content: userMessage, timestamp: new Date() };
    setConversation(prev => [...prev, userMsg]);
    
    const currentUserMessage = userMessage;
    setUserMessage(""); // Clear input immediately
    
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_message: currentUserMessage
        }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      // Add assistant message
      const assistantMsg = { 
        type: "assistant", 
        content: data.response || "No response received", 
        timestamp: new Date(),
        context: data.context || {}
      };
      setConversation(prev => [...prev, assistantMsg]);
      
    } catch (err) {
      console.error('Chat error:', err);
      setError(err.message || "Unknown error occurred");
      // Remove the user message if there was an error
      setConversation(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  // Clear conversation
  const clearConversation = () => {
    setConversation([]);
  };

  // Get context tool type
  const getContextToolType = (context) => {
    if (context && typeof context === 'object') {
      if (context.rag) return 'rag';
      if (context.search) return 'search';
    }
    return null;
  };

  return (
    <ErrorBoundary>
      <div className="chat-container" style={{ 
        fontFamily: "system-ui, sans-serif", 
        minHeight: "100vh", 
        background: "#f8fafc",
        padding: "1rem",
        display: "flex",
        justifyContent: "center"
      }}>
        <div className="chat-card" style={{
          width: "100%",
          maxWidth: "800px",
          background: "white",
          borderRadius: "12px",
          boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
          display: "flex",
          flexDirection: "column",
          height: "calc(100vh - 2rem)",
          overflow: "hidden"
        }}>
          {/* Header */}
          <div className="header-padding" style={{ 
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            color: "white",
            padding: "1rem 2rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderTopLeftRadius: "12px",
            borderTopRightRadius: "12px",
            flexWrap: "wrap",
            gap: "1rem"
          }}>
            <div style={{ 
              display: "flex", 
              alignItems: "center", 
              gap: "1rem",
              flex: 1,
              minWidth: "200px"
            }}>
              <h1 className="header-title" style={{ 
                margin: 0, 
                fontSize: "1.5rem", 
                fontWeight: "600" 
              }}>
                🤖 ParentALL
              </h1>
              <div style={{ 
                display: "flex", 
                alignItems: "center", 
                gap: "0.5rem"
              }}>
                <span style={{ fontSize: "0.875rem", opacity: 0.9 }}>API Status:</span>
                <span style={{ fontSize: "1rem" }}>{health || "🔶"}</span>
              </div>
            </div>
            
            <div style={{ 
              display: "flex", 
              alignItems: "center", 
              gap: "1rem",
              flexShrink: 0
            }}>
              <button
                onClick={clearConversation}
                style={{
                  padding: "0.5rem 1rem",
                  background: "rgba(255,255,255,0.2)",
                  border: "1px solid rgba(255,255,255,0.3)",
                  borderRadius: "8px",
                  color: "white",
                  cursor: "pointer",
                  fontSize: "0.875rem"
                }}
              >
                🗑️ <span className="button-text">Clear Chat</span>
              </button>
            </div>
          </div>

        {/* Context Popup Modal */}
        {showContextPopup && selectedContext && (
          <div style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
            padding: "1rem"
          }}>
            <div style={{
              background: "white",
              borderRadius: "12px",
              padding: window.innerWidth <= 768 ? "1.5rem" : "2rem",
              width: "100%",
              maxWidth: "700px",
              maxHeight: "80vh",
              overflow: "auto",
              boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)"
            }}>
              <div style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "1.5rem"
              }}>
                <h2 style={{ margin: 0, color: "#1f2937" }}>
                  🔧 Tool Context: {getContextToolType(selectedContext)}
                </h2>
                <button
                  onClick={() => setShowContextPopup(false)}
                  style={{
                    background: "none",
                    border: "none",
                    fontSize: "1.5rem",
                    cursor: "pointer",
                    color: "#6b7280"
                  }}
                >
                  ×
                </button>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                <div style={{
                  background: "#f8fafc",
                  border: "1px solid #e5e7eb",
                  borderRadius: "8px",
                  padding: "1rem"
                }}>
                  <div style={{ fontWeight: "600", marginBottom: "0.5rem", color: "#1f2937" }}>
                    Context Data
                  </div>
                  <pre style={{
                    background: "#f3f4f6",
                    padding: "1rem",
                    borderRadius: "4px",
                    fontSize: "0.75rem",
                    overflow: "auto",
                    whiteSpace: "pre-wrap",
                    border: "1px solid #e5e7eb",
                    maxHeight: "400px"
                  }}>
                    {JSON.stringify(selectedContext, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Chat Container */}
        <div style={{ 
          flex: 1, 
          display: "flex", 
          flexDirection: "column",
          overflow: "hidden"
        }}>
          {/* Chat Messages */}
          <div 
            ref={chatContainerRef}
            className="chat-messages-padding"
            style={{ 
              flex: 1, 
              overflowY: "auto", 
              padding: "1rem 2rem",
              background: "#ffffff"
            }}
          >
            {conversation.length === 0 ? (
              <div style={{ 
                textAlign: "center", 
                color: "#6b7280", 
                fontSize: "1.1rem",
                marginTop: "2rem"
              }}>
                <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🤖</div>
                <div>Start a conversation with ParentALL!</div>
                <div style={{ fontSize: "0.875rem", marginTop: "0.5rem" }}>
                  Ready to chat
                </div>
              </div>
            ) : (
              conversation.map((msg, idx) => (
                <div
                  key={idx}
                  style={{
                    display: "flex",
                    justifyContent: msg.type === "user" ? "flex-end" : "flex-start",
                    marginBottom: "1rem"
                  }}
                >
                  <div
                    className="message-bubble"
                    style={{
                      maxWidth: "70%",
                      padding: "1rem",
                      borderRadius: "12px",
                      background: msg.type === "user" 
                        ? "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                        : "#f8fafc",
                      color: msg.type === "user" ? "white" : "#1f2937",
                      border: msg.type === "assistant" ? "1px solid #e5e7eb" : "none",
                      boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
                    }}
                  >
                    <div style={{ 
                      fontSize: "0.75rem", 
                      opacity: 0.8, 
                      marginBottom: "0.5rem",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem"
                    }}>
                      {msg.type === "user" ? "👤 You" : "🤖 ParentALL"}
                      <span>{msg.timestamp.toLocaleTimeString()}</span>
                    </div>
                    <div style={{ lineHeight: "1.6" }}>
                      {msg.type === "assistant" ? (
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p: ({ children }) => (
                              <p style={{ margin: "0.5rem 0", lineHeight: "1.6" }}>
                                {children}
                              </p>
                            ),
                            ul: ({ children }) => (
                              <ul style={{ margin: "0.5rem 0", paddingLeft: "1.5rem" }}>
                                {children}
                              </ul>
                            ),
                            ol: ({ children }) => (
                              <ol style={{ margin: "0.5rem 0", paddingLeft: "1.5rem" }}>
                                {children}
                              </ol>
                            ),
                            li: ({ children }) => (
                              <li style={{ margin: "0.25rem 0" }}>
                                {children}
                              </li>
                            ),
                            code: ({ inline, children }) => (
                              <code style={{
                                background: inline ? "#f3f4f6" : "transparent",
                                padding: inline ? "0.2rem 0.4rem" : "0",
                                borderRadius: inline ? "4px" : "0",
                                fontSize: "0.875rem",
                                fontFamily: "monospace"
                              }}>
                                {children}
                              </code>
                            ),
                            pre: ({ children }) => (
                              <pre style={{
                                background: "#f8fafc",
                                padding: "1rem",
                                borderRadius: "8px",
                                overflow: "auto",
                                border: "1px solid #e5e7eb",
                                fontSize: "0.875rem"
                              }}>
                                {children}
                              </pre>
                            )
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      ) : (
                        <div style={{ whiteSpace: "pre-wrap" }}>
                          {msg.content}
                        </div>
                      )}
                    </div>
                    
                    {/* Show context tool link for assistant messages */}
                    {msg.type === "assistant" && msg.context && getContextToolType(msg.context) && (
                      <div style={{
                        marginTop: "0.5rem",
                        fontSize: "0.75rem"
                      }}>
                        <span 
                          onClick={() => {
                            setSelectedContext(msg.context);
                            setShowContextPopup(true);
                          }}
                          style={{
                            cursor: "pointer",
                            textDecoration: "underline",
                            color: "#3b82f6",
                            opacity: 0.8
                          }}
                        >
                          🔧 tool:{getContextToolType(msg.context)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            
            {/* Loading indicator */}
            {loading && (
              <div style={{
                display: "flex",
                justifyContent: "flex-start",
                marginBottom: "1rem"
              }}>
                <div style={{
                  background: "#f8fafc",
                  border: "1px solid #e5e7eb",
                  borderRadius: "12px",
                  padding: "1rem",
                  color: "#6b7280",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem"
                }}>
                  <div style={{
                    width: "20px",
                    height: "20px",
                    border: "2px solid #e5e7eb",
                    borderTop: "2px solid #3b82f6",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite"
                  }}></div>
                  ParentALL is thinking...
                </div>
              </div>
            )}
            
            <div ref={chatEndRef} />
          </div>

          {/* Error message */}
          {error && (
            <div className="error-margin error-padding" style={{
              background: "#fef2f2",
              border: "1px solid #fecaca",
              color: "#dc2626",
              padding: "1rem",
              margin: "0 2rem",
              borderRadius: "8px",
              fontSize: "1rem"
            }}>
              ❌ {error}
            </div>
          )}
        </div>

        {/* Input Form */}
        <div className="input-form-padding" style={{ 
          background: "#ffffff",
          borderTop: "1px solid #e5e7eb",
          padding: "1rem 2rem",
          borderBottomLeftRadius: "12px",
          borderBottomRightRadius: "12px"
        }}>
          <form onSubmit={handleSubmit} className="input-form" style={{ 
            display: "flex", 
            gap: "1rem",
            flexDirection: "row"
          }}>
            <div style={{ flex: 1 }}>
              <textarea
                value={userMessage}
                onChange={(e) => setUserMessage(e.target.value)}
                placeholder="Type your message..."
                rows="3"
                className="input-textarea"
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  border: "1px solid #d1d5db",
                  borderRadius: "8px",
                  fontSize: "0.875rem",
                  resize: "vertical",
                  minHeight: "60px",
                  boxSizing: "border-box"
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />
            </div>
            <button
              type="submit"
              disabled={loading || !userMessage.trim()}
              className="send-button"
              style={{
                padding: "0.75rem 1.5rem",
                background: (loading || !userMessage.trim()) ? "#94a3b8" : "#3b82f6",
                color: "white",
                border: "none",
                borderRadius: "8px",
                cursor: (loading || !userMessage.trim()) ? "not-allowed" : "pointer",
                fontSize: "0.875rem",
                fontWeight: "500",
                whiteSpace: "nowrap",
                opacity: (loading || !userMessage.trim()) ? 0.5 : 1,
                minHeight: "auto"
              }}
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </form>
        </div>

        {/* Add CSS animation for loading spinner and responsive styles */}
        <style>
          {`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
            
            /* Responsive styles */
            @media (max-width: 768px) {
              .chat-container {
                height: calc(100vh - 1rem) !important;
                margin: 0.5rem !important;
              }
              
              .chat-card {
                border-radius: 8px !important;
              }
              
              .header-title {
                font-size: 1.25rem !important;
              }
              
              .header-padding {
                padding: 1rem !important;
              }
              
              .chat-messages-padding {
                padding: 1rem !important;
              }
              
              .message-bubble {
                max-width: 85% !important;
                padding: 0.75rem !important;
              }
              
              .input-form-padding {
                padding: 1rem !important;
              }
              
              .input-form {
                gap: 0.5rem !important;
              }
              
              .input-textarea {
                padding: 0.5rem !important;
                min-height: 50px !important;
              }
              
              .send-button {
                padding: 0.5rem 1rem !important;
              }
              
              .error-margin {
                margin: 0 1rem !important;
              }
              
              .error-padding {
                padding: 0.75rem !important;
                font-size: 0.875rem !important;
              }
              
              .button-text {
                display: none !important;
              }
            }
            
            @media (max-width: 480px) {
              .input-form {
                flex-direction: column !important;
              }
              
              .send-button {
                min-height: 40px !important;
              }
            }
          `}
        </style>
        </div>
      </div>
    </ErrorBoundary>
  );
} 