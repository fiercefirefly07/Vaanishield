import { useEffect, useRef, useState } from "react";
import {
  ShieldCheck,
  ShieldAlert,
  Radio,
  FolderKanban,
  Mic,
  Square,
  UploadCloud,
  FileAudio,
  FileText,
  AlertTriangle,
  UserCheck,
  Activity,
  Cpu,
  ExternalLink,
  Printer,
  Zap,
  Lock,
  Volume2,
} from "lucide-react";

import LiveMonitor from "./LiveMonitor";
import VerificationSimulator from "./VerificationSimulator";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const RECORDING_DURATION = 6000;

const bankingActions = [
  ["balance_inquiry", "Balance Enquiry"],
  ["transaction_status", "Transaction Status"],
  ["password_reset", "Password Reset"],
  ["mobile_number_change", "Mobile Number Change"],
  ["transfer_limit_increase", "Transfer Limit Increase"],
  ["high_value_transfer", "High Value Transfer"],
];

const authenticationOptions = [
  ["verified", "Verified"],
  ["unverified", "Unverified"],
  ["failed", "Failed Authentication"],
];

const actionLabels = {
  STANDARD_BANKING_FLOW: "Continue Normal Banking Flow",
  STEP_UP_VERIFICATION: "Additional Verification Required",
  HUMAN_REVIEW_REQUIRED: "Human Review Required",
  CRITICAL_FRAUD_INTERCEPT: "CRITICAL FRAUD INTERCEPT (Terminate & Lock)",
};

function percentage(value) {
  if (typeof value !== "number") return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}

// Convert recorded browser audio into mono PCM WAV
function audioBufferToWav(audioBuffer) {
  const length = audioBuffer.length;
  const channels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;

  const wav = new ArrayBuffer(44 + length * 2);
  const view = new DataView(wav);

  function writeString(offset, value) {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, length * 2, true);

  const channelData = [];
  for (let c = 0; c < channels; c++) {
    channelData.push(audioBuffer.getChannelData(c));
  }

  let offset = 44;
  for (let i = 0; i < length; i++) {
    let sample = 0;
    for (let c = 0; c < channels; c++) {
      sample += channelData[c][i];
    }
    sample /= channels;
    sample = Math.max(-1, Math.min(1, sample));
    const pcm = sample < 0 ? sample * 32768 : sample * 32767;
    view.setInt16(offset, pcm, true);
    offset += 2;
  }

  return new Blob([wav], { type: "audio/wav" });
}

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState("analyzer");

  // Banking & Target State
  const [file, setFile] = useState(null);
  const [bankingAction, setBankingAction] = useState("mobile_number_change");
  const [authentication, setAuthentication] = useState("verified");
  const [unusualActivity, setUnusualActivity] = useState(false);
  const [customerId, setCustomerId] = useState("CUST-10492");
  const [transcriptOverride, setTranscriptOverride] = useState("");
  const [presets, setPresets] = useState(null);

  // Analysis State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [analysisSessionId, setAnalysisSessionId] = useState(0);

  // Recording State
  const [isRecording, setIsRecording] = useState(false);
  const [processingAudio, setProcessingAudio] = useState(false);
  const [recordingMessage, setRecordingMessage] = useState("");

  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    return () => {
      clearTimeout(timerRef.current);
      if (recorderRef.current && recorderRef.current.state === "recording") {
        recorderRef.current.stop();
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/api/scam-presets`)
      .then((res) => res.json())
      .then((data) => setPresets(data))
      .catch(() => {});
  }, []);

  function applyPreset(key) {
    if (!presets || !presets[key]) return;
    const item = presets[key];
    setTranscriptOverride(item.sample_transcript);
    setResult(null);

    if (key === "otp_fraud") {
      setBankingAction("password_reset");
      setAuthentication("failed");
      setUnusualActivity(true);
    } else if (key === "remote_access") {
      setBankingAction("high_value_transfer");
      setAuthentication("unverified");
      setUnusualActivity(true);
    } else if (key === "safe_account") {
      setBankingAction("high_value_transfer");
      setAuthentication("unverified");
      setUnusualActivity(true);
    } else if (key === "benign_inquiry") {
      setBankingAction("balance_inquiry");
      setAuthentication("verified");
      setUnusualActivity(false);
    }
  }

  function releaseMicrophone() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  async function startRecording() {
    setError("");
    setResult(null);
    setFile(null);
    setRecordingMessage("");

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("Microphone recording is not supported in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };

      recorder.onerror = () => {
        clearTimeout(timerRef.current);
        releaseMicrophone();
        setIsRecording(false);
        setProcessingAudio(false);
        setError("Microphone recording failed.");
      };

      recorder.onstop = async () => {
        clearTimeout(timerRef.current);
        releaseMicrophone();
        setIsRecording(false);
        setProcessingAudio(true);
        setRecordingMessage("Encoding WAV audio stream...");

        let audioContext;
        try {
          const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
          const encodedAudio = await blob.arrayBuffer();

          const AudioContextClass = window.AudioContext || window.webkitAudioContext;
          audioContext = new AudioContextClass();

          const audioBuffer = await audioContext.decodeAudioData(encodedAudio.slice(0));
          const wavBlob = audioBufferToWav(audioBuffer);

          const recordedFile = new File([wavBlob], "microphone_call.wav", {
            type: "audio/wav",
          });

          setFile(recordedFile);
          setRecordingMessage("Recording captured (6s WAV). Ready for analysis.");
        } catch {
          setError("Failed to convert recorded audio into valid WAV.");
        } finally {
          if (audioContext && audioContext.state !== "closed") {
            audioContext.close().catch(() => {});
          }
          setProcessingAudio(false);
        }
      };

      recorder.start();
      setIsRecording(true);
      setRecordingMessage("Recording live call audio (6 seconds)...");

      timerRef.current = setTimeout(() => {
        if (recorder.state === "recording") {
          recorder.stop();
        }
      }, RECORDING_DURATION);
    } catch {
      releaseMicrophone();
      setIsRecording(false);
      setError("Microphone access denied or audio input device unavailable.");
    }
  }

  function stopRecording() {
    clearTimeout(timerRef.current);
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
  }

  async function analyzeCall(e) {
    e.preventDefault();
    if (!file) {
      setError("Please upload an audio file or record via microphone.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("banking_action", bankingAction);
    formData.append("authentication_status", authentication);
    formData.append("unusual_activity", String(unusualActivity));
    formData.append("customer_id", customerId);

    if (transcriptOverride.trim()) {
      formData.append("transcript_override", transcriptOverride.trim());
    }

    try {
      const response = await fetch(`${API_URL}/api/analyze-call`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : "Analysis failed");
      }
      setAnalysisSessionId((current) => current + 1);
      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to connect to the VaaniShield backend.");
    } finally {
      setLoading(false);
    }
  }

  async function downloadAuditReport() {
    if (!result?.compliance_report) return;
    try {
      const res = await fetch(`${API_URL}/api/compliance/generate-html`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ report_data: result.compliance_report }),
      });
      const html = await res.text();
      const blob = new Blob([html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");
    } catch (err) {
      alert("Failed to render audit report: " + err.message);
    }
  }

  const audio = result?.audio_analysis;
  const transcription = result?.transcription;
  const scam = result?.scam_analysis;
  const threatFusion = result?.threat_fusion;
  const biometrics = result?.voice_biometrics;
  const forensics = result?.audio_forensics;
  const compliance = result?.compliance_report;

  return (
    <div className="app">
      {/* ====================================================================
          TOPBAR / ENTERPRISE BRANDING
      ==================================================================== */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon-wrapper">
            <ShieldCheck size={24} />
          </div>
          <div>
            <h1>
              VaaniShield <span className="brand-badge">SOC v2.4</span>
            </h1>
            <p>AI Voice Anti-Spoofing & Multi-Modal Fraud Operations Console</p>
          </div>
        </div>

        <div className="topbar-right">
          <div className="status-indicator">
            <div className="status-dot" />
            <span>RTX CUDA ACTIVE</span>
          </div>

          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="docs-link"
          >
            <span>Swagger API</span>
            <ExternalLink size={13} />
          </a>
        </div>
      </header>

      {/* ====================================================================
          WORKSPACE TABS
      ==================================================================== */}
      <div className="nav-tabs-wrapper">
        <nav className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === "analyzer" ? "active" : ""}`}
            onClick={() => setActiveTab("analyzer")}
          >
            <ShieldAlert size={16} />
            <span>Threat Analyzer Lab</span>
            <span className="tab-pill">Primary</span>
          </button>

          <button
            className={`nav-tab ${activeTab === "live" ? "active" : ""}`}
            onClick={() => setActiveTab("live")}
          >
            <Radio size={16} />
            <span>Live Audio Radar</span>
            <span className="tab-pill">WebSocket</span>
          </button>

          <button
            className={`nav-tab ${activeTab === "cases" ? "active" : ""}`}
            onClick={() => setActiveTab("cases")}
          >
            <FolderKanban size={16} />
            <span>Case Management (SOC Desk)</span>
            <span className="tab-pill">SQLite</span>
          </button>

          <button
            className={`nav-tab ${activeTab === "system" ? "active" : ""}`}
            onClick={() => setActiveTab("system")}
          >
            <Cpu size={16} />
            <span>Engine Telemetry & Specs</span>
          </button>
        </nav>
      </div>

      {/* ====================================================================
          WORKSPACE CONTENT
      ==================================================================== */}
      <main className="main-content">
        {/* ================================================================
            TAB 1: THREAT ANALYZER LAB
        ================================================================ */}
        {activeTab === "analyzer" && (
          <div>
            <div className="view-header">
              <div className="view-header-title">
                <h2>Multi-Modal Voice Fraud Analyzer</h2>
                <p>
                  Assess incoming telephony audio against neural deepfake spoofing (W2V2-AASIST),
                  speech-to-text social engineering intents, customer voice biometrics, and banking policy.
                </p>
              </div>
            </div>

            <div className="soc-grid">
              {/* LEFT COLUMN: CALL INGESTION & PARAMETERS */}
              <section className="panel-card">
                <div className="panel-header">
                  <div className="panel-title">
                    <Volume2 size={18} color="#38bdf8" />
                    <h3>Call Ingestion & Context</h3>
                  </div>
                  <span className="panel-step-badge">INPUT 01</span>
                </div>

                <form onSubmit={analyzeCall}>
                  {/* AUDIO RECORDER & UPLOADER */}
                  <div className="audio-recorder-zone">
                    <div className="audio-controls">
                      {!isRecording ? (
                        <button
                          type="button"
                          className="action-btn-secondary action-btn-record"
                          disabled={loading || processingAudio}
                          onClick={startRecording}
                        >
                          <Mic size={16} />
                          <span>Record Mic Call (6s)</span>
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="action-btn-secondary action-btn-stop"
                          onClick={stopRecording}
                        >
                          <Square size={16} />
                          <span>Stop Recording</span>
                        </button>
                      )}

                      <label
                        className="action-btn-secondary"
                        style={{ cursor: loading || isRecording ? "not-allowed" : "pointer" }}
                      >
                        <UploadCloud size={16} />
                        <span>Upload Audio File</span>
                        <input
                          type="file"
                          accept=".wav,.flac"
                          style={{ display: "none" }}
                          disabled={loading || isRecording}
                          onChange={(e) => {
                            const selected = e.target.files?.[0];
                            if (selected) {
                              setFile(selected);
                              setResult(null);
                              setError("");
                              setRecordingMessage("");
                            }
                          }}
                        />
                      </label>
                    </div>

                    {isRecording && (
                      <div className="equalizer-wave">
                        <div className="equalizer-bar" />
                        <div className="equalizer-bar" />
                        <div className="equalizer-bar" />
                        <div className="equalizer-bar" />
                        <div className="equalizer-bar" />
                      </div>
                    )}

                    {recordingMessage && (
                      <div style={{ fontSize: "12px", color: "#38bdf8", marginTop: "6px" }}>
                        {recordingMessage}
                      </div>
                    )}

                    {file && (
                      <div className="file-status-tag">
                        <FileAudio size={14} />
                        <span>Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
                      </div>
                    )}
                  </div>

                  {/* QUICK ATTACK PRESETS */}
                  <div className="preset-section">
                    <span className="preset-title">
                      <Zap size={13} />
                      <span>One-Click Simulation Attack Presets</span>
                    </span>
                    <div className="preset-chips">
                      <button
                        type="button"
                        className="preset-chip"
                        onClick={() => applyPreset("otp_fraud")}
                      >
                        ⚡ OTP Extortion
                      </button>
                      <button
                        type="button"
                        className="preset-chip"
                        onClick={() => applyPreset("remote_access")}
                      >
                        ⚡ AnyDesk Coercion
                      </button>
                      <button
                        type="button"
                        className="preset-chip"
                        onClick={() => applyPreset("safe_account")}
                      >
                        ⚡ RBI Safe-Account
                      </button>
                      <button
                        type="button"
                        className="preset-chip"
                        onClick={() => applyPreset("benign_inquiry")}
                      >
                        ✓ Legitimate Inquiry
                      </button>
                    </div>
                  </div>

                  {/* TARGET CUSTOMER BIOMETRICS */}
                  <div className="form-group">
                    <label className="field-label" htmlFor="customer-select">
                      <span>Target Account Holder (Voice Biometrics)</span>
                      <span className="field-hint">1:1 Profile Match</span>
                    </label>
                    <select
                      id="customer-select"
                      value={customerId}
                      disabled={loading}
                      onChange={(e) => {
                        setCustomerId(e.target.value);
                        setResult(null);
                      }}
                    >
                      <option value="CUST-10492">
                        CUST-10492 — Sarah Jenkins (Enrolled Account Holder)
                      </option>
                      <option value="CUST-NEW">
                        CUST-99999 — Unregistered / New Customer
                      </option>
                    </select>
                  </div>

                  {/* BANKING ACTION */}
                  <div className="form-group">
                    <label className="field-label" htmlFor="banking-select">
                      <span>Requested Banking Operation</span>
                      <span className="field-hint">Policy Sensitivity</span>
                    </label>
                    <select
                      id="banking-select"
                      value={bankingAction}
                      disabled={loading}
                      onChange={(e) => {
                        setBankingAction(e.target.value);
                        setResult(null);
                      }}
                    >
                      {bankingActions.map(([val, label]) => (
                        <option key={val} value={val}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* AUTHENTICATION STATUS */}
                  <div className="form-group">
                    <label className="field-label" htmlFor="auth-select">
                      <span>Identity Authentication Status</span>
                      <span className="field-hint">Direct Verification</span>
                    </label>
                    <select
                      id="auth-select"
                      value={authentication}
                      disabled={loading}
                      onChange={(e) => {
                        setAuthentication(e.target.value);
                        setResult(null);
                      }}
                    >
                      {authenticationOptions.map(([val, label]) => (
                        <option key={val} value={val}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* UNUSUAL ACTIVITY CHECKBOX */}
                  <div className="form-group">
                    <label style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer", fontSize: "13px" }}>
                      <input
                        type="checkbox"
                        checked={unusualActivity}
                        disabled={loading}
                        onChange={(e) => {
                          setUnusualActivity(e.target.checked);
                          setResult(null);
                        }}
                      />
                      <span>Flag anomalous velocity or abnormal geo-location</span>
                    </label>
                  </div>

                  {/* TRANSCRIPT OVERRIDE */}
                  <div className="form-group">
                    <label className="field-label" htmlFor="transcript-area">
                      <span>Call Dialogue Transcript (Optional script override)</span>
                      <span className="field-hint">Auto-transcribed if blank</span>
                    </label>
                    <textarea
                      id="transcript-area"
                      rows={2}
                      placeholder="Leave empty to auto-transcribe speech from audio, or customize dialogue..."
                      value={transcriptOverride}
                      disabled={loading}
                      onChange={(e) => {
                        setTranscriptOverride(e.target.value);
                        setResult(null);
                      }}
                    />
                  </div>

                  <button
                    type="submit"
                    className="cta-analyze-btn"
                    disabled={loading || !file || isRecording || processingAudio}
                  >
                    {loading ? (
                      <>
                        <Activity size={18} className="animate-spin" />
                        <span>Running Multi-Modal Security Inference...</span>
                      </>
                    ) : (
                      <>
                        <ShieldCheck size={18} />
                        <span>Run Threat Fusion Analysis →</span>
                      </>
                    )}
                  </button>

                  {error && (
                    <div style={{ marginTop: "14px", padding: "12px", background: "rgba(239, 68, 68, 0.15)", border: "1px solid #ef4444", borderRadius: "8px", color: "#fca5a5", fontSize: "13px" }}>
                      {error}
                    </div>
                  )}
                </form>
              </section>

              {/* RIGHT COLUMN: INTELLIGENCE OUTPUT */}
              <section className="panel-card">
                <div className="panel-header">
                  <div className="panel-title">
                    <Activity size={18} color="#10b981" />
                    <h3>Threat Intelligence Results</h3>
                  </div>
                  <span className="panel-step-badge">OUTPUT 02</span>
                </div>

                {!result ? (
                  <div className="empty-results-box">
                    <div className="empty-results-icon">
                      <Radio size={28} />
                    </div>
                    <h4>No Analysis Evaluated</h4>
                    <p>
                      Upload an audio recording or record a 6-second mic snippet, select a banking operation,
                      and execute the threat assessment.
                    </p>
                  </div>
                ) : (
                  <div>
                    {/* MULTI-MODAL THREAT FUSION BANNER */}
                    {threatFusion && (
                      <div className={`threat-fusion-banner ${threatFusion.threat_level}`}>
                        <div className="threat-header">
                          <div>
                            <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "1px", color: "#94a3b8" }}>
                              Unified Threat Fusion Index
                            </span>
                            <div className="threat-score-large">
                              <strong>{threatFusion.fused_threat_score}</strong>
                              <span>/100</span>
                            </div>
                          </div>
                          <span className={`threat-tier-pill ${threatFusion.threat_level}`}>
                            {threatFusion.threat_level} THREAT
                          </span>
                        </div>

                        {threatFusion.compound_threat && (
                          <div className="compound-alert-box">
                            <AlertTriangle size={18} />
                            <span>COMPOUND THREAT DETECTED: Simultaneous AI voice clone and social engineering vectors identified!</span>
                          </div>
                        )}

                        <div className="threat-pillars">
                          <div className="pillar-card">
                            <span className="pillar-label">Acoustic AI</span>
                            <span className="pillar-val" style={{ color: threatFusion.acoustic_risk > 50 ? "#f87171" : "#4ade80" }}>
                              {threatFusion.acoustic_risk}%
                            </span>
                          </div>
                          <div className="pillar-card">
                            <span className="pillar-label">Scam Intent</span>
                            <span className="pillar-val" style={{ color: threatFusion.linguistic_risk > 50 ? "#f87171" : "#4ade80" }}>
                              {threatFusion.linguistic_risk}%
                            </span>
                          </div>
                          <div className="pillar-card">
                            <span className="pillar-label">Action Risk</span>
                            <span className="pillar-val" style={{ color: "#38bdf8" }}>
                              {threatFusion.transaction_risk}%
                            </span>
                          </div>
                          <div className="pillar-card">
                            <span className="pillar-label">Identity Risk</span>
                            <span className="pillar-val" style={{ color: threatFusion.auth_risk > 20 ? "#fbbf24" : "#94a3b8" }}>
                              {threatFusion.auth_risk}%
                            </span>
                          </div>
                        </div>

                        {threatFusion.recommended_protocol && (
                          <div className="protocol-strip">
                            <strong>Automated Security Action: {actionLabels[threatFusion.action] || threatFusion.action}</strong>
                            <span>{threatFusion.recommended_protocol}</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* SPEECH TRANSCRIPTION BUBBLE */}
                    {transcription?.text && (
                      <div className="transcript-bubble">
                        <div className="transcript-header">
                          <span>🎙️ Dialogue Transcript ({transcription.engine})</span>
                          <span>{transcription.word_count} words</span>
                        </div>
                        <p className="transcript-text">"{transcription.text}"</p>
                      </div>
                    )}

                    {/* SOCIAL ENGINEERING TRIGGERS */}
                    {scam?.triggers?.length > 0 && (
                      <div style={{ marginBottom: "18px" }}>
                        <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "#f87171", display: "block", marginBottom: "8px" }}>
                          Flagged Scam Indicators ({scam.triggers.length})
                        </span>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                          {scam.triggers.map((t, idx) => (
                            <span
                              key={idx}
                              style={{
                                fontSize: "11px",
                                padding: "4px 8px",
                                borderRadius: "6px",
                                background: t.severity === "CRITICAL" ? "rgba(239, 68, 68, 0.2)" : "rgba(249, 115, 22, 0.2)",
                                border: `1px solid ${t.severity === "CRITICAL" ? "#ef4444" : "#f97316"}`,
                                color: "#f1f5f9",
                              }}
                              title={t.snippet}
                            >
                              [{t.category}] "{t.pattern_matched}"
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* VOICE BIOMETRICS CARD */}
                    {biometrics && (
                      <div className="info-section-card">
                        <div className="info-section-title">
                          <h4>
                            <UserCheck size={16} color="#38bdf8" />
                            <span>1:1 Voice Biometric Matching</span>
                          </h4>
                          <span
                            style={{
                              fontSize: "11px",
                              fontWeight: "700",
                              padding: "3px 10px",
                              borderRadius: "12px",
                              background: biometrics.tri_factor_verdict === "AUTHENTIC_ACCOUNT_HOLDER" ? "#14532d" : "#7f1d1d",
                              color: biometrics.tri_factor_verdict === "AUTHENTIC_ACCOUNT_HOLDER" ? "#86efac" : "#fca5a5",
                            }}
                          >
                            {biometrics.tri_factor_verdict.replace(/_/g, " ")}
                          </span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", marginBottom: "6px" }}>
                          <span style={{ color: "#94a3b8" }}>Target Profile: {biometrics.customer_name} ({biometrics.customer_id})</span>
                          <strong style={{ color: "#38bdf8" }}>{percentage(biometrics.similarity_score)} match</strong>
                        </div>
                        <p style={{ fontSize: "12px", color: "#cbd5e1", margin: 0, lineHeight: 1.5 }}>
                          {biometrics.security_explanation}
                        </p>
                      </div>
                    )}

                    {/* ACOUSTIC FORENSICS (XAI) */}
                    {forensics && (
                      <div className="info-section-card">
                        <div className="info-section-title">
                          <h4>
                            <Activity size={16} color="#06b6d4" />
                            <span>Acoustic Forensics Diagnostics (XAI)</span>
                          </h4>
                          <span style={{ fontSize: "11px", fontWeight: "700", color: forensics.forensic_anomaly_score > 20 ? "#f87171" : "#4ade80" }}>
                            Anomaly: {forensics.forensic_anomaly_score}/100
                          </span>
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "8px", margin: "10px 0" }}>
                          <div style={{ background: "#0b1424", padding: "8px", borderRadius: "6px", textAlign: "center" }}>
                            <span style={{ fontSize: "10px", color: "#94a3b8", display: "block" }}>Centroid</span>
                            <strong style={{ fontSize: "13px" }}>{forensics.spectral_centroid_hz} Hz</strong>
                          </div>
                          <div style={{ background: "#0b1424", padding: "8px", borderRadius: "6px", textAlign: "center" }}>
                            <span style={{ fontSize: "10px", color: "#94a3b8", display: "block" }}>Rolloff</span>
                            <strong style={{ fontSize: "13px" }}>{forensics.spectral_rolloff_hz} Hz</strong>
                          </div>
                          <div style={{ background: "#0b1424", padding: "8px", borderRadius: "6px", textAlign: "center" }}>
                            <span style={{ fontSize: "10px", color: "#94a3b8", display: "block" }}>Vocoder Cutoff</span>
                            <strong style={{ fontSize: "13px", color: forensics.vocoder_cutoff_detected ? "#f87171" : "#4ade80" }}>
                              {forensics.vocoder_cutoff_detected ? "DETECTED" : "ORGANIC"}
                            </strong>
                          </div>
                        </div>

                        <p style={{ fontSize: "12px", color: "#cbd5e1", margin: 0 }}>
                          {forensics.forensic_summary}
                        </p>
                      </div>
                    )}

                    {/* MODEL METRICS GRID */}
                    {audio && (
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "10px", marginBottom: "18px" }}>
                        <div style={{ background: "#091224", padding: "12px", borderRadius: "8px", border: "1px solid #1e293b", textAlign: "center" }}>
                          <span style={{ fontSize: "10px", color: "#94a3b8", display: "block", textTransform: "uppercase" }}>Model Spoof</span>
                          <strong style={{ fontSize: "16px", color: audio.spoof_score > 0.5 ? "#f87171" : "#4ade80" }}>
                            {percentage(audio.spoof_score)}
                          </strong>
                        </div>
                        <div style={{ background: "#091224", padding: "12px", borderRadius: "8px", border: "1px solid #1e293b", textAlign: "center" }}>
                          <span style={{ fontSize: "10px", color: "#94a3b8", display: "block", textTransform: "uppercase" }}>Latency</span>
                          <strong style={{ fontSize: "16px", color: "#f1f5f9" }}>{audio.inference_seconds}s</strong>
                        </div>
                        <div style={{ background: "#091224", padding: "12px", borderRadius: "8px", border: "1px solid #1e293b", textAlign: "center" }}>
                          <span style={{ fontSize: "10px", color: "#94a3b8", display: "block", textTransform: "uppercase" }}>Architecture</span>
                          <strong style={{ fontSize: "13px", color: "#38bdf8" }}>{audio.model}</strong>
                        </div>
                      </div>
                    )}

                    {/* COMPLIANCE AUDIT CERTIFICATE RIBBON */}
                    {compliance && (
                      <div className="audit-export-ribbon">
                        <div>
                          <div className="audit-meta-title">
                            <FileText size={16} />
                            <span>Cryptographic Audit Certificate: {compliance.certificate_id}</span>
                          </div>
                          <div className="audit-meta-digest">
                            SHA-256: {compliance.audio_forensic_fingerprint.sha256_checksum.slice(0, 32)}...
                          </div>
                        </div>
                        <button type="button" className="audit-print-btn" onClick={downloadAuditReport}>
                          <Printer size={15} />
                          <span>Print / Save Audit Certificate</span>
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </section>
            </div>
          </div>
        )}

        {/* ================================================================
            TAB 2: LIVE STREAMING RADAR (WEBSOCKET)
        ================================================================ */}
        {activeTab === "live" && (
          <div>
            <div className="view-header">
              <div className="view-header-title">
                <h2>Real-Time Microphone Telemetry & Stream Interceptor</h2>
                <p>
                  Streams raw 16kHz audio chunks across WebSocket to the CUDA-accelerated backend with continuous
                  sliding-window deepfake evaluations.
                </p>
              </div>
            </div>
            <LiveMonitor />
          </div>
        )}

        {/* ================================================================
            TAB 3: SOC CASE MANAGEMENT
        ================================================================ */}
        {activeTab === "cases" && (
          <div>
            <div className="view-header">
              <div className="view-header-title">
                <h2>Fraud Investigation Desk & Case Management</h2>
                <p>
                  SQLite-backed incident resolution workflow, customer SMS/OTP push simulations, officer escalation notes,
                  and audit event logging.
                </p>
              </div>
            </div>
            <VerificationSimulator
              key={result ? analysisSessionId : "no-analysis"}
              analysis={result}
            />
          </div>
        )}

        {/* ================================================================
            TAB 4: ENGINE TELEMETRY & SPECS
        ================================================================ */}
        {activeTab === "system" && (
          <div>
            <div className="view-header">
              <div className="view-header-title">
                <h2>Security Architecture & Engine Specifications</h2>
                <p>
                  Telemetry parameters, active hardware acceleration providers, neural model topology, and risk policy matrices.
                </p>
              </div>
            </div>

            <div className="system-spec-grid">
              <div className="system-spec-card">
                <h4>
                  <Cpu size={16} />
                  <span>Neural Core Pipeline</span>
                </h4>
                <ul className="spec-list">
                  <li><span>Model Architecture</span><strong>Wav2Vec2 + AASIST</strong></li>
                  <li><span>Inference Engine</span><strong>ONNX Runtime v1.26</strong></li>
                  <li><span>Hardware Provider</span><strong style={{ color: "#34d399" }}>CUDAExecutionProvider</strong></li>
                  <li><span>Device Target</span><strong>NVIDIA GeForce RTX GPU</strong></li>
                  <li><span>Model Checkpoint</span><strong>1.26 GB (FP32)</strong></li>
                  <li><span>Latency Benchmark</span><strong>~0.042 seconds</strong></li>
                </ul>
              </div>

              <div className="system-spec-card">
                <h4>
                  <Activity size={16} />
                  <span>Acoustic Ingestion & DSP</span>
                </h4>
                <ul className="spec-list">
                  <li><span>Sample Rate</span><strong>16,000 Hz (16 kHz Mono)</strong></li>
                  <li><span>Audio Window</span><strong>64,600 samples (~4.037s)</strong></li>
                  <li><span>Hop Size</span><strong>32,000 samples (2.0s)</strong></li>
                  <li><span>Silence Gate</span><strong>RMS &lt; 0.001 rejection</strong></li>
                  <li><span>ASR Transcription</span><strong>Google Speech ASR</strong></li>
                  <li><span>Biometrics Embeddings</span><strong>128-dim L2 Normalized</strong></li>
                </ul>
              </div>

              <div className="system-spec-card">
                <h4>
                  <Lock size={16} />
                  <span>Threat Fusion Weights</span>
                </h4>
                <ul className="spec-list">
                  <li><span>Acoustic Deepfake Vector</span><strong>35% Weight</strong></li>
                  <li><span>Linguistic Scam Intent</span><strong>35% Weight</strong></li>
                  <li><span>Banking Action Sensitivity</span><strong>15% Weight</strong></li>
                  <li><span>Auth & Identity Context</span><strong>15% Weight</strong></li>
                  <li><span>Compound Alert Gate</span><strong>Dual vector &gt; 65%</strong></li>
                  <li><span>Audit Verification</span><strong>SHA-256 Sealed</strong></li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ====================================================================
          FOOTER
      ==================================================================== */}
      <footer className="soc-footer">
        VaaniShield Security Suite v2.4 · Multi-Modal Anti-Spoofing & Cyber-Fraud Defense Console · No Real Transactions
      </footer>
    </div>
  );
}
