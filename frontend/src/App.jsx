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
  
  // Session and mood management
  const [sessionStarted, setSessionStarted] = useState(false);
  const [showMoodSelection, setShowMoodSelection] = useState(false);
  const [selectedMood, setSelectedMood] = useState(null);
  
  // Family data management
  const [showFamilyForm, setShowFamilyForm] = useState(false);
  const [familyData, setFamilyData] = useState({
    mother_name: "",
    father_name: "",
    number_of_kids: 1,
    kids_names: [""],
    kids_ages: [0],
    children: [{
      name: "",
      age: 0,
      strengths: ["", "", ""],
      growth_areas: ["", "", ""]
    }]
  });
  
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

  // Load family data on mount
  React.useEffect(() => {
    loadFamilyData();
  }, []);

  // Load family data from backend
  const loadFamilyData = async () => {
    try {
      const res = await fetch("/api/get-family");
      if (res.ok) {
        const data = await res.json();
        if (data.status === "success" && data.data) {
          setFamilyData(data.data);
        }
      }
    } catch (err) {
      console.error('Failed to load family data:', err);
    }
  };

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
                 console.log('Received streaming data:', data); // Debug log
                 console.log('Data type:', data.type); // Debug log
                 console.log('Data content:', data.content); // Debug log
                
                                 // Update conversation based on streaming data type
                 setConversation(prev => {
                   const newConversation = [...prev];
                   let lastMessageIndex = newConversation.length - 1;
                   
                   // Check if we need to create the assistant message for the first time
                   const needsAssistantMessage = lastMessageIndex < 0 || newConversation[lastMessageIndex].type !== "assistant";
                   
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
                     lastMessageIndex = newConversation.length - 1;
                     console.log('Created new assistant message'); // Debug log
                   }
                   
                   if (lastMessageIndex >= 0 && newConversation[lastMessageIndex].type === "assistant") {
                     const lastMessage = newConversation[lastMessageIndex];
                     
                     if (data.type === "message") {
                       // Append message content
                       newConversation[lastMessageIndex] = {
                         ...lastMessage,
                         content: lastMessage.content + data.content
                       };
                       console.log('Appended content:', data.content); // Debug log
                     } else if (data.type === "response") {
                       // Update response content
                       newConversation[lastMessageIndex] = {
                         ...lastMessage,
                         content: data.content
                       };
                       console.log('Updated response content:', data.content); // Debug log
                     } else if (data.type === "tool_call") {
                       // Update tool calls
                       newConversation[lastMessageIndex] = {
                         ...lastMessage,
                         tool_calls: data.content.tool_calls,
                         context: { ...lastMessage.context, tool_calls: data.content.tool_calls }
                       };
                     } else if (data.type === "final") {
                       // Final update with metadata
                       newConversation[lastMessageIndex] = {
                         ...lastMessage,
                         isStreaming: false,
                         metadata: data.content.metadata,
                         tool_calls: data.content.tool_calls || []
                       };
                       // End streaming mode - allow user to scroll
                       setIsStreaming(false);
                       console.log('Final message received, streaming ended'); // Debug log
                     }
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
    setSessionStarted(false);
    setShowMoodSelection(false);
    setSelectedMood(null);
    
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

  // Start new session
  const startNewSession = () => {
    setSessionStarted(true);
    setShowMoodSelection(true);
    setConversation([]);
    setSelectedMood(null);
  };

  // Handle mood selection
  const handleMoodSelection = async (mood) => {
    setSelectedMood(mood);
    setShowMoodSelection(false);
    
    try {
      // Send mood to backend
      const res = await fetch("/api/set-mood", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mood: mood,
          thread_id: "1"
        }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      // Add welcome message and mood selection to conversation
      const welcomeMsg = { 
        type: "assistant", 
        content: "How can I help you today?", 
        timestamp: new Date() 
      };
      setConversation([welcomeMsg]);
      
    } catch (err) {
      console.error('Failed to set mood:', err);
      setError("Failed to set mood. Please try again.");
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

  // Family form handling functions
  const handleFamilyDataChange = (field, value, childIndex = null, subField = null) => {
    setFamilyData(prev => {
      try {
        const newData = { ...prev };
        
        if (childIndex !== null) {
          // Create a new children array
          newData.children = [...prev.children];
          
          if (subField !== null) {
            // Handle nested fields like strengths and growth_areas
            if (field === "strengths" || field === "growth_areas") {
              // Ensure the child exists and has the required arrays
              if (!newData.children[childIndex]) {
                newData.children[childIndex] = {
                  name: "",
                  age: 0,
                  strengths: ["", "", ""],
                  growth_areas: ["", "", ""]
                };
              }
              
              // Create a new array for the specific field
              const newArray = [...newData.children[childIndex][field]];
              newArray[subField] = value;
              
              newData.children[childIndex] = {
                ...newData.children[childIndex],
                [field]: newArray
              };
            } else {
              // Handle other child fields
              newData.children[childIndex] = {
                ...newData.children[childIndex],
                [field]: value
              };
            }
          } else {
            // Handle direct child field updates
            newData.children[childIndex] = {
              ...newData.children[childIndex],
              [field]: value
            };
          }
        } else {
          // Handle parent-level fields
          newData[field] = value;
        }
        
        return newData;
      } catch (error) {
        console.error('Error in handleFamilyDataChange:', error);
        // Return the previous state if there's an error
        return prev;
      }
    });
  };

  const handleNumberOfKidsChange = (newNumber) => {
    const newNumberInt = Math.max(1, Math.min(10, newNumber)); // Limit between 1-10
    
    setFamilyData(prev => {
      try {
        const newData = { ...prev };
        newData.number_of_kids = newNumberInt;
        
        // Create new children array
        newData.children = [...prev.children];
        
        // Ensure all children have the required structure
        for (let i = 0; i < newData.children.length; i++) {
          if (!newData.children[i]) {
            newData.children[i] = {
              name: "",
              age: 0,
              strengths: ["", "", ""],
              growth_areas: ["", "", ""]
            };
          } else {
            // Ensure arrays exist
            if (!newData.children[i].strengths || newData.children[i].strengths.length !== 3) {
              newData.children[i].strengths = ["", "", ""];
            }
            if (!newData.children[i].growth_areas || newData.children[i].growth_areas.length !== 3) {
              newData.children[i].growth_areas = ["", "", ""];
            }
          }
        }
        
        // Adjust arrays to match new number of kids
        while (newData.children.length < newNumberInt) {
          newData.children.push({
            name: "",
            age: 0,
            strengths: ["", "", ""],
            growth_areas: ["", "", ""]
          });
        }
        while (newData.children.length > newNumberInt) {
          newData.children.pop();
        }
        
        return newData;
      } catch (error) {
        console.error('Error in handleNumberOfKidsChange:', error);
        return prev;
      }
    });
  };

  const saveFamilyData = async () => {
    try {
      const res = await fetch("/api/save-family", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(familyData),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const result = await res.json();
      if (result.status === "success") {
        setShowFamilyForm(false);
        // Show success message or toast
        console.log("Family data saved successfully");
      }
    } catch (err) {
      console.error('Failed to save family data:', err);
      setError("Failed to save family data. Please try again.");
    }
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
                 <span style={{ fontSize: "2rem" }}>🌻</span> ParentALL.ai
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
                               {!sessionStarted ? (
                  <button
                    onClick={() => setShowFamilyForm(true)}
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
                                         <span style={{ fontSize: "1.2rem" }}>👨‍👩‍👧‍👦</span> <span className="button-text">Family</span>
                  </button>
                ) : sessionStarted && conversation.length > 0 && (
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
                        🔒 <span className="button-text">Close Session</span>
                      </>
                    )}
                  </button>
                )}
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

         {/* Family Intake Modal */}
         {showFamilyForm && (
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
               maxWidth: "900px",
               maxHeight: "90vh",
               overflow: "auto",
               boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)"
             }}>
               <div style={{
                 display: "flex",
                 justifyContent: "space-between",
                 alignItems: "center",
                 marginBottom: "2rem"
               }}>
                 <h2 style={{ margin: 0, color: "#1f2937", fontSize: "1.75rem" }}>
                   👨‍👩‍👧‍👦 Family Intake
                 </h2>
                 <button
                   onClick={() => setShowFamilyForm(false)}
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

               <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
                 {/* Parent Information */}
                 <div style={{
                   background: "#f8fafc",
                   border: "1px solid #e5e7eb",
                   borderRadius: "8px",
                   padding: "1.5rem"
                 }}>
                   <h3 style={{ margin: "0 0 1rem 0", color: "#1f2937", fontSize: "1.25rem" }}>
                     👨‍👩‍👧‍👦 Parent Information
                   </h3>
                                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }} className="family-form-grid">
                       <div>
                         <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500", color: "#374151" }}>
                           Mother's Name
                         </label>
                         <input
                           type="text"
                           value={familyData.mother_name}
                           onChange={(e) => handleFamilyDataChange("mother_name", e.target.value)}
                           style={{
                             width: "100%",
                             padding: "0.75rem",
                             border: "1px solid #d1d5db",
                             borderRadius: "6px",
                             fontSize: "0.875rem"
                           }}
                           placeholder="Enter mother's name"
                         />
                       </div>
                       <div>
                         <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500", color: "#374151" }}>
                           Father's Name
                         </label>
                         <input
                           type="text"
                           value={familyData.father_name}
                           onChange={(e) => handleFamilyDataChange("father_name", e.target.value)}
                           style={{
                             width: "100%",
                             padding: "0.75rem",
                             border: "1px solid #d1d5db",
                             borderRadius: "6px",
                             fontSize: "0.875rem"
                           }}
                           placeholder="Enter father's name"
                         />
                       </div>
                     </div>
                 </div>

                 {/* Number of Kids */}
                 <div style={{
                   background: "#f8fafc",
                   border: "1px solid #e5e7eb",
                   borderRadius: "8px",
                   padding: "1.5rem"
                 }}>
                   <h3 style={{ margin: "0 0 1rem 0", color: "#1f2937", fontSize: "1.25rem" }}>
                     👶 Number of Children
                   </h3>
                   <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                     <label style={{ fontWeight: "500", color: "#374151" }}>
                       Number of kids:
                     </label>
                                            <input
                         type="number"
                         min="1"
                         max="10"
                         value={familyData.number_of_kids}
                         onChange={(e) => handleNumberOfKidsChange(parseInt(e.target.value) || 1)}
                         style={{
                           width: "80px",
                           padding: "0.5rem",
                           border: "1px solid #d1d5db",
                           borderRadius: "6px",
                           fontSize: "0.875rem",
                           textAlign: "center"
                         }}
                         className="family-form-number-input"
                       />
                   </div>
                 </div>

                 {/* Children Information */}
                 {familyData.children.map((child, index) => (
                   <div key={index} style={{
                     background: "#f8fafc",
                     border: "1px solid #e5e7eb",
                     borderRadius: "8px",
                     padding: "1.5rem"
                   }}>
                     <h3 style={{ margin: "0 0 1rem 0", color: "#1f2937", fontSize: "1.25rem" }}>
                       👶 Child {index + 1}
                     </h3>
                     <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }} className="family-form-grid">
                       <div>
                         <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500", color: "#374151" }}>
                           Name
                         </label>
                         <input
                           type="text"
                           value={child.name}
                           onChange={(e) => handleFamilyDataChange("name", e.target.value, index)}
                           style={{
                             width: "100%",
                             padding: "0.75rem",
                             border: "1px solid #d1d5db",
                             borderRadius: "6px",
                             fontSize: "0.875rem"
                           }}
                           placeholder="Enter child's name"
                         />
                       </div>
                       <div>
                         <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500", color: "#374151" }}>
                           Age
                         </label>
                         <input
                           type="number"
                           min="0"
                           max="18"
                           value={child.age}
                           onChange={(e) => handleFamilyDataChange("age", parseInt(e.target.value) || 0, index)}
                           style={{
                             width: "100%",
                             padding: "0.75rem",
                             border: "1px solid #d1d5db",
                             borderRadius: "6px",
                             fontSize: "0.875rem"
                           }}
                           placeholder="Enter age"
                         />
                       </div>
                     </div>

                     {/* Strengths */}
                     <div style={{ marginBottom: "1rem" }}>
                       <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500", color: "#374151" }}>
                         🌟 3 Strengths
                       </label>
                       <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.5rem" }} className="family-form-strengths-grid">
                         {child.strengths.map((strength, strengthIndex) => (
                           <input
                             key={strengthIndex}
                             type="text"
                             value={strength}
                             onChange={(e) => handleFamilyDataChange("strengths", e.target.value, index, strengthIndex)}
                             style={{
                               padding: "0.5rem",
                               border: "1px solid #d1d5db",
                               borderRadius: "6px",
                               fontSize: "0.875rem"
                             }}
                             placeholder={`Strength ${strengthIndex + 1}`}
                           />
                         ))}
                       </div>
                     </div>

                     {/* Growth Areas */}
                     <div>
                       <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500", color: "#374151" }}>
                         📈 3 Growth Areas
                       </label>
                       <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.5rem" }} className="family-form-growth-grid">
                         {child.growth_areas.map((area, areaIndex) => (
                           <input
                             key={areaIndex}
                             type="text"
                             value={area}
                             onChange={(e) => handleFamilyDataChange("growth_areas", e.target.value, index, areaIndex)}
                             style={{
                               padding: "0.5rem",
                               border: "1px solid #d1d5db",
                               borderRadius: "6px",
                               fontSize: "0.875rem"
                             }}
                             placeholder={`Growth area ${areaIndex + 1}`}
                           />
                         ))}
                       </div>
                     </div>
                   </div>
                 ))}

                 {/* Action Buttons */}
                 <div style={{ display: "flex", gap: "1rem", justifyContent: "flex-end" }}>
                   <button
                     onClick={() => setShowFamilyForm(false)}
                     style={{
                       padding: "0.75rem 1.5rem",
                       background: "#6b7280",
                       color: "white",
                       border: "none",
                       borderRadius: "8px",
                       cursor: "pointer",
                       fontSize: "0.875rem",
                       fontWeight: "500"
                     }}
                   >
                     Cancel
                   </button>
                   <button
                     onClick={saveFamilyData}
                     style={{
                       padding: "0.75rem 1.5rem",
                       background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                       color: "white",
                       border: "none",
                       borderRadius: "8px",
                       cursor: "pointer",
                       fontSize: "0.875rem",
                       fontWeight: "500"
                     }}
                   >
                     💾 Save Family Data
                   </button>
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
            {!sessionStarted ? (
              <div style={{ 
                textAlign: "center", 
                color: "#6b7280", 
                fontSize: "1.1rem",
                marginTop: "2rem"
              }}>
                <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🌻</div>
                <div>Welcome to ParentALL.ai!</div>
                <div style={{ fontSize: "0.875rem", marginTop: "0.5rem", marginBottom: "2rem" }}>
                  Click the button below to start a new session
                </div>
                <button
                  onClick={startNewSession}
                  style={{
                    padding: "1rem 2rem",
                    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "12px",
                    fontSize: "1.1rem",
                    fontWeight: "600",
                    cursor: "pointer",
                    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
                  }}
                >
                  New Session
                </button>
              </div>
            ) : showMoodSelection ? (
              <div style={{ 
                textAlign: "center", 
                color: "#6b7280", 
                fontSize: "1.1rem",
                marginTop: "2rem"
              }}>
                <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>😊</div>
                <div style={{ fontSize: "1.2rem", marginBottom: "2rem", color: "#1f2937" }}>
                  Hello there, how are you feeling today?
                </div>
                <div style={{ 
                  display: "grid", 
                  gridTemplateColumns: "repeat(auto-fit, minmax(80px, 1fr))",
                  gap: "1rem",
                  maxWidth: "500px",
                  margin: "0 auto"
                }} className="mood-grid">
                  {[
                    { emoji: "😡", mood: "very upset", label: "Very Upset" },
                    { emoji: "😭", mood: "very sad", label: "Very Sad" },
                    { emoji: "😢", mood: "sad", label: "Sad" },
                    { emoji: "😐", mood: "neutral", label: "Neutral" },
                    { emoji: "🙂", mood: "happy", label: "Happy" },
                    { emoji: "😃", mood: "very happy", label: "Very Happy" },
                    { emoji: "🤩", mood: "elated", label: "Elated" }
                  ].map((moodOption) => (
                    <button
                      key={moodOption.mood}
                      onClick={() => handleMoodSelection(moodOption.mood)}
                      className="mood-button"
                      style={{
                        padding: "1rem",
                        background: "white",
                        border: "2px solid #e5e7eb",
                        borderRadius: "12px",
                        fontSize: "2rem",
                        cursor: "pointer",
                        transition: "all 0.2s ease",
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: "0.5rem"
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.borderColor = "#667eea";
                        e.target.style.transform = "translateY(-2px)";
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.borderColor = "#e5e7eb";
                        e.target.style.transform = "translateY(0)";
                      }}
                    >
                      <span>{moodOption.emoji}</span>
                      <span style={{ 
                        fontSize: "0.75rem", 
                        color: "#6b7280",
                        fontWeight: "500"
                      }} className="mood-label">
                        {moodOption.label}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
                         ) : conversation.length === 0 ? (
               <div style={{ 
                 textAlign: "center", 
                 color: "#6b7280", 
                 fontSize: "1.1rem",
                 marginTop: "2rem"
               }}>
                                   <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🌻</div>
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
                      {msg.type === "user" ? "👤 You" : "🌻 ParentALL.ai"}
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
        {sessionStarted && !showMoodSelection && (
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
        )}

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
              
              /* Mood selection responsive styles */
              .mood-grid {
                grid-template-columns: repeat(3, 1fr) !important;
                gap: 0.5rem !important;
                max-width: 300px !important;
              }
              
              .mood-button {
                padding: 0.75rem !important;
                font-size: 1.5rem !important;
              }
              
                             .mood-label {
                 font-size: 0.625rem !important;
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
             
             /* Family form responsive styles */
             @media (max-width: 768px) {
               .family-form-grid {
                 grid-template-columns: 1fr !important;
               }
               
               .family-form-strengths-grid,
               .family-form-growth-grid {
                 grid-template-columns: 1fr !important;
               }
               
               .family-form-number-input {
                 width: 100% !important;
               }
             }
          `}
        </style>
        </div>
      </div>
    </ErrorBoundary>
  );
} 