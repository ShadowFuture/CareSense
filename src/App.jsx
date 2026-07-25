import { useEffect, useRef, useState } from 'react';

const backendBase = '/api';

const pages = [
  { key: 'home', label: 'Overview' },
  { key: 'pulse', label: 'Pulse Online' },
  { key: 'ml', label: 'CareSense ML' },
];

const moduleCards = [
  {
    key: 'pulse',
    title: 'Pulse Online',
    description: 'Keep the webcam pulse capture active and see the latest cardiovascular status from the backend.',
    highlight: 'Live pulse monitoring',
  },
  {
    key: 'ml',
    title: 'CareSense ML',
    description: 'Run the ML inference pipeline, review alerts, and watch prediction confidence in real time.',
    highlight: 'Real-time model insights',
  },
];

const featureCards = [
  {
    title: 'Unified status',
    description: 'One app to control both Pulse Online and CareSense ML from a single interface.',
  },
  {
    title: 'Camera controls',
    description: 'Start or stop the backend camera service without leaving the app.',
  },
  {
    title: 'Live backend data',
    description: 'Poll backend state and surface the latest inference output automatically.',
  },
  {
    title: 'Voice assistant',
    description: 'Hands-free voice commands make it easy to start, stop, and monitor systems without clicking.',
  },
];

function App() {
  const [view, setView] = useState('home');
  const [pulseStatus, setPulseStatus] = useState(null);
  const [mlStatus, setMlStatus] = useState(null);
  const [loadingAction, setLoadingAction] = useState(false);
  const [error, setError] = useState(null);
  const [voiceListening, setVoiceListening] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(true);
  const [voiceTranscript, setVoiceTranscript] = useState('Use voice commands like “Start Pulse Online” or “Stop CareSense ML”.');
  const [voiceStatus, setVoiceStatus] = useState('Ready');
  const recognitionRef = useRef(null);

  const currentModule = view === 'pulse' ? 'pulse' : view === 'ml' ? 'ml' : null;
  const currentStatus = currentModule === 'pulse' ? pulseStatus : currentModule === 'ml' ? mlStatus : null;
  const currentOutput = currentStatus?.prediction ?? currentStatus;

  const fetchStatus = async () => {
    setError(null);
    try {
      const [pulseRes, mlRes] = await Promise.all([
        fetch(`${backendBase}/pulse`),
        fetch(`${backendBase}/ml`),
      ]);

      if (!pulseRes.ok || !mlRes.ok) {
        throw new Error('Unable to connect to backend API');
      }

      const [pulseData, mlData] = await Promise.all([pulseRes.json(), mlRes.json()]);
      setPulseStatus(pulseData);
      setMlStatus(mlData);
    } catch (err) {
      setError(err.message || 'Backend is unavailable');
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setVoiceSupported(false);
      setVoiceStatus('Voice unavailable');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join(' ')
        .trim();
      setVoiceTranscript(transcript);

      const command = transcript.toLowerCase();
      if (/pulse/.test(command)) {
        if (/stop|pause|disable|off/.test(command)) {
          handleAction('pulse', 'stop');
        } else if (/start|run|activate|on/.test(command)) {
          handleAction('pulse', 'start');
        }
      }
      if (/ml/.test(command) || /care sense|caresense/.test(command)) {
        if (/stop|pause|disable|off/.test(command)) {
          handleAction('ml', 'stop');
        } else if (/start|run|activate|on/.test(command)) {
          handleAction('ml', 'start');
        }
      }
    };

    recognition.onerror = () => {
      setVoiceStatus('Voice error');
      setVoiceListening(false);
    };

    recognition.onend = () => {
      if (voiceListening) {
        recognition.start();
      } else {
        setVoiceStatus('Paused');
      }
    };

    recognitionRef.current = recognition;
    return () => {
      recognition.stop();
    };
  }, [voiceListening]);

  const handleAction = async (module, action) => {
    setLoadingAction(true);
    setError(null);
    try {
      const response = await fetch(`${backendBase}/${module}/${action}`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Action request failed');
      }
      await fetchStatus();
    } catch (err) {
      setError(err.message || 'Action failed');
    } finally {
      setLoadingAction(false);
    }
  };

  const toggleVoiceAssistant = () => {
    if (!voiceSupported) {
      setError('Voice assistant is not supported by this browser.');
      return;
    }

    if (!recognitionRef.current) {
      setError('Voice assistant is not available yet.');
      return;
    }

    if (voiceListening) {
      recognitionRef.current.stop();
      setVoiceListening(false);
      setVoiceStatus('Paused');
    } else {
      recognitionRef.current.start();
      setVoiceListening(true);
      setVoiceStatus('Listening for commands');
    }
  };

  const summaryCards = [
    {
      title: 'Pulse status',
      value: pulseStatus?.running ? 'Online' : 'Offline',
      detail: 'Pulse Online camera',
    },
    {
      title: 'ML status',
      value: mlStatus?.running ? 'Online' : 'Offline',
      detail: 'CareSense ML pipeline',
    },
    {
      title: 'Backend health',
      value: error ? 'Unavailable' : 'Connected',
      detail: error || 'API polling every 4 seconds',
    },
  ];

  const statusLabel = currentStatus?.running ? 'Online' : 'Offline';
  const lastChecked = currentOutput?.timestamp ?? currentStatus?.last_checked ?? 'N/A';
  const eventType = currentOutput?.event_type ?? currentOutput?.prediction?.event_type ?? 'N/A';
  const confidence = currentOutput?.confidence ?? currentOutput?.prediction?.confidence ?? 'N/A';
  const riskScore = currentOutput?.risk_score ?? currentOutput?.prediction?.risk_score ?? 'N/A';
  const bpmValue = currentOutput?.estimated_bpm ?? currentOutput?.prediction?.estimated_bpm ?? 'N/A';
  const warnings = currentOutput?.warnings ?? currentOutput?.prediction?.warnings ?? 'No warnings';
  const recommendation =
    currentOutput?.recommendation ?? currentOutput?.prediction?.recommendation ?? currentOutput?.summary_short ?? 'No recommendation available.';

  return (
    <div className="app-shell app-shell--app">
      <div className="app-header">
        <div>
          <span className="eyebrow">CareSense</span>
          <h1 className="app-title">CareSense Control Hub</h1>
          <p className="app-subtitle">A modern operations center for Pulse Online and CareSense ML.</p>
        </div>

        <nav className="top-nav" aria-label="Primary navigation">
          {pages.map((page) => (
            <button
              key={page.key}
              className={`nav-button ${view === page.key ? 'active' : ''}`}
              onClick={() => setView(page.key)}
            >
              {page.label}
            </button>
          ))}
        </nav>
      </div>

      <main className="app-content">
        <section className="hero-section">
          <div className="hero-copy">
            <span className="eyebrow">Smart Operations</span>
            <h2>Control your monitoring workflow with clarity.</h2>
            <p>
              Access live status, manage camera services, and review backend inference results from a polished app interface.
            </p>
          </div>

          <div className="hero-aside">
            <div className="status-grid status-grid--top">
              {summaryCards.map((card) => (
                <div key={card.title} className="status-tile">
                  <span>{card.title}</span>
                  <strong>{card.value}</strong>
                  <p>{card.detail}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="content-panel">
          {view === 'home' && (
            <>
              <div className="feature-grid">
                {featureCards.map((feature) => (
                  <article key={feature.title} className="feature-card">
                    <h3>{feature.title}</h3>
                    <p>{feature.description}</p>
                  </article>
                ))}
              </div>

              <div className="workbench-grid">
                {moduleCards.map((module) => {
                  const status = module.key === 'pulse' ? pulseStatus : mlStatus;
                  return (
                    <div key={module.key} className="workbench-card">
                      <div>
                        <span className="eyebrow">{module.title}</span>
                        <h3>{module.highlight}</h3>
                        <p>{module.description}</p>
                      </div>
                      <div className="workbench-details">
                        <span>Status</span>
                        <strong>{status?.running ? 'Live' : 'Stopped'}</strong>
                      </div>
                      <button className="action-button start" onClick={() => setView(module.key)}>
                        Open {module.title}
                      </button>
                    </div>
                  );
                })}
              </div>

              <div className="assistant-card">
                <div>
                  <span className="eyebrow">Voice assistant</span>
                  <h3>Hands-free commands</h3>
                  <p>Keep the assistant active and use voice commands to start or stop Pulse Online and CareSense ML.</p>
                </div>
                <div className="assistant-controls">
                  <button
                    className={`action-button ${voiceListening ? 'stop' : 'start'}`}
                    onClick={toggleVoiceAssistant}
                  >
                    {voiceListening ? 'Stop voice assistant' : 'Start voice assistant'}
                  </button>
                  <span>{voiceStatus}</span>
                  <p className="assistant-note">{voiceTranscript}</p>
                </div>
              </div>
            </>
          )}

          {view !== 'home' && (
            <div className="module-panel">
              <div className="module-overview">
                <div>
                  <span className="eyebrow">{view === 'pulse' ? 'Pulse Online' : 'CareSense ML'}</span>
                  <h2>{view === 'pulse' ? 'Live pulse capture' : 'Live ML inference'}</h2>
                  <p>
                    {view === 'pulse'
                      ? 'Manage the webcam pulse service and inspect the latest cardiovascular inference output.'
                      : 'Manage the ML pipeline and inspect the latest alert confidence and risk summary.'}
                  </p>
                </div>
                <div className="module-actions">
                  <button
                    className="action-button start"
                    onClick={() => handleAction(currentModule, 'start')}
                    disabled={loadingAction}
                  >
                    Start
                  </button>
                  <button
                    className="action-button stop"
                    onClick={() => handleAction(currentModule, 'stop')}
                    disabled={loadingAction}
                  >
                    Stop
                  </button>
                </div>
              </div>

              <div className="status-grid">
                <div className="status-tile status-tile--large">
                  <span>Session state</span>
                  <strong>{statusLabel}</strong>
                  <p>Last checked: {lastChecked}</p>
                </div>
                <div className="status-tile">
                  <span>Event type</span>
                  <strong>{eventType}</strong>
                </div>
                <div className="status-tile">
                  <span>Confidence</span>
                  <strong>{confidence}</strong>
                </div>
                <div className="status-tile">
                  <span>Risk score</span>
                  <strong>{riskScore}</strong>
                </div>
              </div>

              <div className="module-details">
                <div className="output-card">
                  <h3>Key readings</h3>
                  <div className="output-grid">
                    <div>
                      <span>BPM</span>
                      <strong>{bpmValue}</strong>
                    </div>
                    <div>
                      <span>Warnings</span>
                      <strong>{Array.isArray(warnings) ? warnings.join(', ') : warnings}</strong>
                    </div>
                  </div>
                </div>
                <div className="assistant-card">
                  <h3>Voice assistant</h3>
                  <p>Speak commands while this module is open to manage it faster. Try “Start Pulse Online” or “Stop CareSense ML”.</p>
                  <button
                    className={`action-button ${voiceListening ? 'stop' : 'start'}`}
                    onClick={toggleVoiceAssistant}
                  >
                    {voiceListening ? 'Pause voice assistant' : 'Listen for voice commands'}
                  </button>
                  <span>{voiceStatus}</span>
                </div>
                <div className="recommendation-card">
                  <h3>Recommended action</h3>
                  <p>{recommendation}</p>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
