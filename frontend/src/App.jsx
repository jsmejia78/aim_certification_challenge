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
  const [clearing, setClearing] = useState(false);
  
  // Chat conversation history
  const [conversation, setConversation] = useState([]);
  
  // Health check state
  const [health, setHealth] = useState(null);
  
  // Context popup state
  const [showContextPopup, setShowContextPopup] = useState(false);
  const [selectedContext, setSelectedContext] = useState(null);
  
  // Simple scroll control
  const [isStreaming, setIsStreaming] = useState(false);
  const [questionSubmitted, setQuestionSubmitted] = useState(false);
  
  // Ref for auto-scrolling to latest message
  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Smart scroll logic: bring the new question to the top of viewing area
  useEffect(() => {
    if (questionSubmitted && chatContainerRef.current) {
      // Find the last user message (the question we just added)
      const userMessages = chatContainerRef.current.querySelectorAll('[data-message-type="user"]');
      const lastUserMessage = userMessages[userMessages.length - 1];
      
      if (lastUserMessage) {
        // Calculate scroll position to bring question to top of viewing area
        const scrollToQuestion = () => {
          const container = chatContainerRef.current;
          const messageRect = lastUserMessage.getBoundingClientRect();
          const containerRect = container.getBoundingClientRect();
          
          // Calculate how much to scroll to bring the message to the top
          const scrollAmount = container.scrollTop + (messageRect.top - containerRect.top);
          
          // Scroll to position the question at the top
          container.scrollTo({
            top: scrollAmount,
            behavior: 'smooth'
          });
        };
        requestAnimationFrame(scrollToQuestion);
      }
      setQuestionSubmitted(false); // Only scroll once per question
    }
  }, [questionSubmitted]);

  // Disable scrolling during streaming by adding CSS
  useEffect(() => {
    const chatContainer = chatContainerRef.current;
    if (!chatContainer) return;

    if (isStreaming) {
      // Disable scrolling during streaming
      chatContainer.style.overflowY = 'hidden';
    } else {
      // Re-enable scrolling when streaming finishes
      chatContainer.style.overflowY = 'auto';
    }

    return () => {
      // Cleanup
      if (chatContainer) {
        chatContainer.style.overflowY = 'auto';
      }
    };
  }, [isStreaming]);

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
    
    // Trigger scroll to question and start streaming mode
    setQuestionSubmitted(true);
    setIsStreaming(true);
    
    // Add user message to conversation
    const userMsg = { type: "user", content: userMessage, timestamp: new Date() };
    setConversation(prev => [...prev, userMsg]);
    
    const currentUserMessage = userMessage;
    setUserMessage(""); // Clear input immediately
    
    // Don't add assistant message yet - wait for first streaming content
    
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_message: currentUserMessage,
          thread_id: "1" // Default thread_id as requested
        }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      // Handle streaming response
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                // Update conversation based on streaming data type
                setConversation(prev => {
                  const newConversation = [...prev];
                  let lastMessage = newConversation[newConversation.length - 1];
                  
                  // Check if we need to create the assistant message for the first time
                  const needsAssistantMessage = !lastMessage || lastMessage.type !== "assistant";
                  
                  if (needsAssistantMessage && (data.type === "message" || data.type === "response")) {
                    // Create assistant message on first streaming content
                    const assistantMsg = { 
                      type: "assistant", 
                      content: "", 
                      timestamp: new Date(),
                      context: {},
                      isStreaming: true,
                      tool_calls: [],
                      metadata: null
                    };
                    newConversation.push(assistantMsg);
                    lastMessage = assistantMsg;
                  }
                  
                  if (data.type === "message" && lastMessage && lastMessage.type === "assistant") {
                    // Append message content
                    lastMessage.content += data.content;
                  } else if (data.type === "response" && lastMessage && lastMessage.type === "assistant") {
                    // Update response content
                    lastMessage.content = data.content;
                  } else if (data.type === "tool_call" && lastMessage && lastMessage.type === "assistant") {
                    // Update tool calls
                    lastMessage.tool_calls = data.content.tool_calls;
                    lastMessage.context = { ...lastMessage.context, tool_calls: data.content.tool_calls };
                  } else if (data.type === "final" && lastMessage && lastMessage.type === "assistant") {
                    // Final update with metadata
                    lastMessage.isStreaming = false;
                    lastMessage.metadata = data.content.metadata;
                    lastMessage.tool_calls = data.content.tool_calls || [];
                    // End streaming mode - allow user to scroll
                    setIsStreaming(false);
                  }
                  
                  return newConversation;
                });
              } catch (parseError) {
                console.error('Error parsing streaming data:', parseError);
              }
            }
          }
        }
      } catch (streamError) {
        console.error('Streaming error:', streamError);
        // Mark the message as no longer streaming and show error
        setConversation(prev => {
          const newConversation = [...prev];
          const lastMessage = newConversation[newConversation.length - 1];
          if (lastMessage.type === "assistant") {
            lastMessage.isStreaming = false;
            lastMessage.content = "Error: Streaming response was interrupted.";
          }
          return newConversation;
        });
      } finally {
        // Ensure the message is marked as not streaming
        setConversation(prev => {
          const newConversation = [...prev];
          const lastMessage = newConversation[newConversation.length - 1];
          if (lastMessage && lastMessage.type === "assistant") {
            lastMessage.isStreaming = false;
          }
          return newConversation;
        });
      }
      
    } catch (err) {
      console.error('Chat error:', err);
      setError(err.message || "Unknown error occurred");
      // Remove the user message if there was an error
      setConversation(prev => {
        const newConv = [...prev];
        // Remove user message (always the last one if no assistant message was added)
        if (newConv.length > 0 && newConv[newConv.length - 1].type === "user") {
          newConv.pop();
        }
        // If assistant message was added, remove it too
        else if (newConv.length > 1 && newConv[newConv.length - 1].type === "assistant") {
          newConv.pop(); // Remove assistant
          newConv.pop(); // Remove user
        }
        return newConv;
      });
    } finally {
      setLoading(false);
      setIsStreaming(false); // End streaming mode on error
    }
  };

  // Clear conversation
  const clearConversation = async () => {
    setClearing(true);
    setConversation([]);
    
    // Also clear the agent's memory
    try {
      await fetch("/api/clear-memory", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      });
    } catch (err) {
      console.error('Failed to clear agent memory:', err);
      // Don't show error to user - clearing chat still works locally
    } finally {
      setClearing(false);
    }
  };

  // Get context tool type
  const getContextToolType = (context) => {
    if (context && typeof context === 'object') {
      const hasRag = context.rag;
      const hasSearch = context.search;
      
      if (hasRag && hasSearch) return 'search+rag';
      if (hasRag) return 'rag';
      if (hasSearch) return 'search';
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
                🤖 ParentALL.ai
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
                disabled={clearing}
                style={{
                  padding: "0.5rem 1rem",
                  background: "rgba(255,255,255,0.2)",
                  border: "1px solid rgba(255,255,255,0.3)",
                  borderRadius: "8px",
                  color: "white",
                  cursor: clearing ? "not-allowed" : "pointer",
                  fontSize: "0.875rem",
                  opacity: clearing ? 0.7 : 1
                }}
              >
                {clearing ? (
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <div style={{
                      width: "20px",
                      height: "20px",
                      border: "2px solid #e5e7eb",
                      borderTop: "2px solid #3b82f6",
                      borderRadius: "50%",
                      animation: "spin 1s linear infinite"
                    }}></div>
                    Clearing...
                  </div>
                ) : (
                  <>
                    🗑️ <span className="button-text">Clear Chat</span>
                  </>
                )}
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
                <div>Start a conversation with ParentALL.ai!</div>
                <div style={{ fontSize: "0.875rem", marginTop: "0.5rem" }}>
                  Ready to chat
                </div>
              </div>
            ) : (
              conversation.map((msg, idx) => (
                <div
                  key={idx}
                  data-message-type={msg.type}
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
                      {msg.type === "user" ? "👤 You" : "🤖 ParentALL.ai"}
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
                          {msg.content || (msg.isStreaming ? "..." : "No response received")}
                        </ReactMarkdown>
                      ) : (
                        <div style={{ whiteSpace: "pre-wrap" }}>
                          {msg.content}
                        </div>
                      )}
                    </div>
                    
                    {/* Show tool calls and metadata for assistant messages */}
                    {msg.type === "assistant" && (
                      <div style={{ marginTop: "0.5rem", fontSize: "0.75rem" }}>
                        {/* Tool calls indicator */}
                        {msg.tool_calls && msg.tool_calls.length > 0 && (
                          <div style={{
                            marginBottom: "0.5rem",
                            padding: "0.5rem",
                            background: "#f0f9ff",
                            border: "1px solid #bae6fd",
                            borderRadius: "6px",
                            color: "#0369a1"
                          }}>
                            🔧 Used {msg.tool_calls.length} tool(s)
                          </div>
                        )}
                        

                        
                        {/* Legacy context tool link for backward compatibility */}
                        {msg.context && getContextToolType(msg.context) && (
                          <div style={{ marginTop: "0.5rem" }}>
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
                  ParentALL.ai is thinking...
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