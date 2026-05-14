import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import "./styles.css";

interface Candidate {
  id: string;
  name: string;
  match_score: number;
  rerank_score: number;
  skill_score: number;
  final_score: number;
  technical_skills?: string;
  category?: string;
}

interface Analysis {
  category: string;
  experience_level: string;
  technical_skills: string[];
  summary: string;
}

const API_BASE_URL = "http://localhost:8000";

export default function App() {
  const [jdText, setJdText]     = useState<string>("");
  const [loading, setLoading]   = useState<boolean>(false);
  const [results, setResults]   = useState<Candidate[]>([]);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError]       = useState<string>("");

  const [chatInput, setChatInput]       = useState("");
  const [chatMessages, setChatMessages] = useState<{ role: string; text: string }[]>([]);
  const [chatHistory, setChatHistory]   = useState<any[]>([]);
  const [chatLoading, setChatLoading]   = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const exportCSV = async () => {
    if (!results.length || !analysis) return;
    try {
      const response = await axios.post(
        `${API_BASE_URL}/export-csv`,
        { candidates: results, analysis },
        { responseType: "blob" }
      );
      const url  = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href  = url;
      link.setAttribute("download", "ITG_SmartMatcher_Results.csv");
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export failed", err);
    }
  };

  const startMatching = async () => {
    if (!jdText.trim()) return;
    setError("");
    setLoading(true);
    setResults([]);
    setAnalysis(null);
    setChatMessages([]);
    setChatHistory([]);
    try {
      const response = await axios.post(`${API_BASE_URL}/search`, {
        job_description: jdText,
        top_k: 10,
      });
      setResults(response.data.candidates);
      setAnalysis(response.data.analysis);
    } catch (err) {
      setError("Connection failed! Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async () => {
    if (!chatInput.trim() || results.length === 0) return;
    const userMsg = chatInput;
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", text: userMsg }]);
    setChatLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        question: userMsg,
        history: chatHistory,
        candidate_ids: results.map((c) => c.id),
      });
      setChatHistory(response.data.history);
      setChatMessages((prev) => [...prev, { role: "ai", text: response.data.answer }]);
    } catch {
      setChatMessages((prev) => [...prev, { role: "ai", text: "Connection error." }]);
    } finally {
      setChatLoading(false);
    }
  };

  const scoreColor = (s: number) => s >= 0.7 ? "#22c55e" : s >= 0.4 ? "#fbbf24" : "#ef4444";

  return (
    <div className="app">

      {/* NAV */}
      <header className="nav">
        <div className="nav-left">
          <div className="logo-box">
            <img
              src="https://media.licdn.com/dms/image/v2/C560BAQGCgWJKIp1RPA/company-logo_200_200/company-logo_200_200/0/1630598874498/integrated_technology_group_itg__logo?e=2147483647&v=beta&t=0hjwr1CC_-OXxWvcxofFJ1nkYPoG1p6bujsYDHl50a8"
              alt="ITG"
            />
          </div>
          <div className="logo-name">
            <span className="logo-smart">SMART</span>
            <span className="logo-matcher">MATCHER</span>
          </div>
          <div className="nav-divider" />
          <span className="nav-sub">AI Talent Intelligence</span>
        </div>
        <div className="nav-right">
          {analysis && (
            <>
              <span className="pill-cat">{analysis.category}</span>
              <span className="pill-lvl">{analysis.experience_level}</span>
            </>
          )}
        </div>
      </header>

      <div className="workspace">

        {/* LEFT PANEL */}
        <aside className="left-panel">
          <p className="panel-label">Job Description</p>
          <textarea
            className="jd-input"
            value={jdText}
            onChange={(e) => setJdText(e.target.value)}
            placeholder="Paste the full job description here..."
          />
          {error && <p className="err">{error}</p>}
          <button
            className={`match-btn ${loading ? "busy" : ""}`}
            onClick={startMatching}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Find Best Candidates"}
          </button>

          {/* CHAT BOX */}
          <div className="chat-box">
            <p className="panel-label" style={{ marginBottom: "12px" }}>AI Talent Assistant</p>
            <div className="chat-display">
              {chatMessages.length === 0 && (
                <p className="chat-hint">
                  {results.length > 0
                    ? "Ask me to compare candidates or analyze results..."
                    : "Find candidates first, then ask me anything."}
                </p>
              )}
              {chatMessages.map((msg, i) => (
                <div key={i} className={`chat-msg ${msg.role}`}>{msg.text}</div>
              ))}
              {chatLoading && <div className="chat-msg ai thinking">Processing...</div>}
              <div ref={chatEndRef} />
            </div>
            <div className="chat-input-row">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleChat()}
                placeholder={results.length > 0 ? "Ask about candidates..." : "Find candidates first..."}
                disabled={results.length === 0}
              />
              <button onClick={handleChat} disabled={chatLoading || results.length === 0}>→</button>
            </div>
          </div>

          {analysis?.summary && (
            <div className="summary-box">
              <p className="panel-label" style={{ color: "#fbbf24", marginBottom: "8px" }}>Role Summary</p>
              <p className="summary-text">{analysis.summary}</p>
            </div>
          )}
        </aside>

        {/* CENTER PANEL */}
        <main className="center-panel">
          <div className="list-header">
            <h2 className="list-title">Matched Talent Pool</h2>
            {results.length > 0 && (
              <button 
                 className="new-search-btn"
                 onClick={() => {
                    setResults([]);
                    setAnalysis(null);
                    setJdText("");
                    setChatMessages([]);
                    setChatHistory([]);
                 }}
              >
              ↩ New Search
            </button>
)}
            <div className="list-actions">
              {results.length > 0 && <span className="list-count">{results.length} candidates</span>}
              {results.length > 0 && <button className="export-btn" onClick={exportCSV}>⬇ Export CSV</button>}
            </div>
          </div>

          <div className="candidates-list">
            <AnimatePresence>
              {results.map((cand, idx) => (
                <motion.div
                  key={cand.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="cand-card"
                >
                  <div className="cand-rank">#{idx + 1}</div>

                  <div className="cand-info">
                    <h4 className="cand-name">{cand.name}</h4>
                    <span className="cand-id">REF: {cand.id}</span>
                    <div className="cand-skills">
                      {cand.technical_skills?.split(",").slice(0, 4).map((s, i) => (
                        <span key={i} className="cand-skill-tag">{s.trim()}</span>
                      ))}
                    </div>
                  </div>

                  {/* Score Bars */}
                  <div className="cand-bars">
                    <div className="bar-row">
                      <span className="bar-lbl">Semantic</span>
                      <div className="bar-bg"><div className="bar-fill b1" style={{ width: `${cand.match_score * 100}%` }} /></div>
                      <span className="bar-pct">{Math.round(cand.match_score * 100)}%</span>
                    </div>
                    <div className="bar-row">
                      <span className="bar-lbl">Rerank</span>
                      <div className="bar-bg"><div className="bar-fill b2" style={{ width: `${cand.rerank_score * 100}%` }} /></div>
                      <span className="bar-pct">{Math.round(cand.rerank_score * 100)}%</span>
                    </div>
                    <div className="bar-row">
                      <span className="bar-lbl">Skills</span>
                      <div className="bar-bg"><div className="bar-fill b3" style={{ width: `${cand.skill_score * 100}%` }} /></div>
                      <span className="bar-pct">{Math.round(cand.skill_score * 100)}%</span>
                    </div>
                  </div>

                  {/* Circle */}
                  <div className="cand-score">
                    <svg viewBox="0 0 36 36" className="score-svg">
                      <path className="score-bg-path" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                      <path
                        className="score-fill-path"
                        stroke={scoreColor(cand.final_score)}
                        strokeDasharray={`${Math.round(cand.final_score * 100)}, 100`}
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                    </svg>
                    <span className="score-pct" style={{ color: scoreColor(cand.final_score) }}>
                      {Math.round(cand.final_score * 100)}%
                    </span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {!loading && results.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">⬡</div>
                <p>Paste a job description and click<br /><strong>Find Best Candidates</strong></p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
