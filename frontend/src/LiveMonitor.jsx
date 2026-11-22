
import { useEffect, useRef, useState } from "react";
import "./LiveMonitor.css";

const WS_URL = "ws://127.0.0.1:8000/ws/live-audio";
const MAX_DURATION_MS = 20000;
const MAX_BUFFERED_BYTES = 128000;

export default function LiveMonitor() {
  const [status, setStatus] = useState("idle");
  const [elapsed, setElapsed] = useState(0);
  const [predictions, setPredictions] = useState([]);
  const [qualityWarning, setQualityWarning] = useState("");
  const [error, setError] = useState("");

  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const contextRef = useRef(null);
  const sourceRef = useRef(null);
  const processorRef = useRef(null);

  const timerRef = useRef(null);
  const clockRef = useRef(null);
  const startTimeRef = useRef(0);
  const stoppingRef = useRef(false);

  function cleanupAudio() {
    clearTimeout(timerRef.current);
    clearInterval(clockRef.current);

    timerRef.current = null;
    clockRef.current = null;

    if (processorRef.current) {
      processorRef.current.port.onmessage = null;
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => track.stop());

      streamRef.current = null;
    }

    if (contextRef.current) {
      const context = contextRef.current;
      contextRef.current = null;

      if (context.state !== "closed") {
        context.close().catch(() => {});
      }
    }
  }

  function stopMonitoring() {
    if (stoppingRef.current) return;

    stoppingRef.current = true;
    cleanupAudio();

    const socket = socketRef.current;

    if (socket?.readyState === WebSocket.OPEN) {
      setStatus("stopping");
      socket.send("stop");
    } else {
      socket?.close();
      setStatus("done");
    }
  }

  async function startMicrophone(socket) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });

      if (
        stoppingRef.current ||
        socket.readyState !== WebSocket.OPEN
      ) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;

      const context = new AudioContext();
      contextRef.current = context;

      await context.audioWorklet.addModule(
        "/pcm16-processor.js"
      );

      if (
        stoppingRef.current ||
        socket.readyState !== WebSocket.OPEN
      ) {
        cleanupAudio();
        return;
      }

      const source = context.createMediaStreamSource(stream);

      const processor = new AudioWorkletNode(
        context,
        "pcm16-processor",
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [1],
        }
      );

      sourceRef.current = source;
      processorRef.current = processor;

      processor.port.onmessage = (event) => {
        if (
          stoppingRef.current ||
          socket.readyState !== WebSocket.OPEN
        ) {
          return;
        }

        if (!(event.data instanceof ArrayBuffer)) {
          return;
        }

        // Stop rather than silently dropping audio samples.
        if (socket.bufferedAmount > MAX_BUFFERED_BYTES) {
          setError(
            "Streaming is falling behind. Monitoring stopped."
          );
          stopMonitoring();
          return;
        }

        socket.send(event.data);
      };

      source.connect(processor);

      // The processor outputs silence, avoiding feedback.
      processor.connect(context.destination);

      await context.resume();

      if (stoppingRef.current) {
        cleanupAudio();
        return;
      }

      startTimeRef.current = Date.now();

      setStatus("recording");

      clockRef.current = setInterval(() => {
        setElapsed(
          (Date.now() - startTimeRef.current) / 1000
        );
      }, 250);

      timerRef.current = setTimeout(
        stopMonitoring,
        MAX_DURATION_MS
      );
    } catch (err) {
      console.error("Microphone error:", err);

      setError(
        "Unable to start microphone. Check browser permissions."
      );

      stoppingRef.current = true;
      cleanupAudio();
      socket.close();
      setStatus("error");
    }
  }

  function startMonitoring() {
    if (
      !navigator.mediaDevices?.getUserMedia ||
      !window.AudioWorkletNode ||
      !window.WebSocket
    ) {
      setError(
        "Live monitoring requires a compatible browser."
      );
      return;
    }

    stoppingRef.current = false;

    setError("");
    setQualityWarning("");
    setPredictions([]);
    setElapsed(0);
    setStatus("connecting");

    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      let message;

      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }

      switch (message.type) {
        case "ready":
          if (!stoppingRef.current) {
            startMicrophone(socket);
          }
          break;

        case "prediction":
          setPredictions((previous) =>
            [...previous, message].slice(-12)
          );
          break;

        case "audio_quality":
          setQualityWarning(
            `Audio window ending at ${
              message.window_end_seconds
            }s: ${message.status}`
          );
          break;

        case "finished":
          stoppingRef.current = true;
          cleanupAudio();
          setElapsed(message.duration_seconds);
          setStatus("done");
          socket.close();
          break;

        case "error":
          setError(
            message.message || "Live analysis failed."
          );
          stoppingRef.current = true;
          cleanupAudio();
          setStatus("error");
          socket.close();
          break;

        default:
          break;
      }
    };

    socket.onerror = () => {
      setError(
        "WebSocket connection failed. Check that the " +
        "streaming FastAPI server is running."
      );

      stoppingRef.current = true;
      cleanupAudio();
      setStatus("error");
    };

    socket.onclose = () => {
      cleanupAudio();

      if (socketRef.current === socket) {
        socketRef.current = null;
      }

      if (!stoppingRef.current) {
        setError(
          "The live connection closed unexpectedly."
        );
        setStatus("error");
      }
    };
  }

  useEffect(() => {
    return () => {
      stoppingRef.current = true;
      cleanupAudio();

      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, []);

  const latest = predictions.at(-1);

  const active = [
    "connecting",
    "recording",
    "stopping",
  ].includes(status);

  return (
    <section className="panel live-monitor">
      <div className="panel-heading">
        <h3>Live Voice Monitoring</h3>
        <span className="step">03 / LIVE AUDIO</span>
      </div>

      <div className="live-header">
        <div>
          <p className="live-caption">MICROPHONE STATUS</p>
          <h4>{status.toUpperCase()}</h4>
        </div>

        <div className="live-timer">
          {elapsed.toFixed(1)}s
          <small> / 20s maximum</small>
        </div>
      </div>

      <p className="live-description">
        Continuously stream microphone audio to
        W2V2-AASIST and view experimental detection
        scores while speaking.
      </p>

      <div className="live-controls">
        <button
          type="button"
          className="live-start"
          disabled={active}
          onClick={startMonitoring}
        >
          Start Live Monitoring
        </button>

        <button
          type="button"
          className="live-stop"
          disabled={!active || status === "stopping"}
          onClick={stopMonitoring}
        >
          Stop Monitoring
        </button>
      </div>

      {error && (
        <div className="live-error" role="alert">
          {error}
        </div>
      )}

      {qualityWarning && (
        <div className="live-warning">
          {qualityWarning}
        </div>
      )}

      <div className="live-data">
        <div className="live-stat">
          <span>Latest Spoof Score</span>
          <strong>
            {latest
              ? `${(latest.spoof_score * 100).toFixed(2)}%`
              : "—"}
          </strong>
        </div>

        <div className="live-stat">
          <span>Latest Genuine Score</span>
          <strong>
            {latest
              ? `${(latest.genuine_score * 100).toFixed(2)}%`
              : "—"}
          </strong>
        </div>

        <div className="live-stat">
          <span>Analysis Windows</span>
          <strong>{predictions.length}</strong>
        </div>

        <div className="live-stat">
          <span>Last Inference Time</span>
          <strong>
            {latest
              ? `${latest.inference_seconds}s`
              : "—"}
          </strong>
        </div>
      </div>

      <h4 className="live-section-title">
        Live Prediction History
      </h4>

      {predictions.length === 0 ? (
        <p className="live-placeholder">
          No predictions yet. Start monitoring and speak
          naturally. The first result should arrive after
          roughly four seconds of audio plus processing
          and buffering time.
        </p>
      ) : (
        <div className="live-history">
          {[...predictions].reverse().map((item) => (
            <div
              className="live-history-row"
              key={item.window_start_seconds}
            >
              <span>
                {item.window_start_seconds}s –
                {" "}
                {item.window_end_seconds}s
              </span>

              <strong>
                {(item.spoof_score * 100).toFixed(2)}%
              </strong>

              <small>Experimental spoof score</small>
            </div>
          ))}
        </div>
      )}

      <div className="live-disclaimer">
        <strong>Monitor-only research prototype</strong>
        <p>
          The voice model is not validated for banking
          decisions. High spoof scores may be false
          positives. This panel does not approve, block,
          or freeze any banking activity.
        </p>
      </div>
    </section>
  );
}
