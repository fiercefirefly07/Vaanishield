# VaaniShield 🛡️
### AI-Powered Voice Cloning Fraud Detection for Banking and Digital Payments

VaaniShield is an AI-powered voice fraud detection prototype designed to help identify potentially cloned or synthetically generated voices during voice-based interactions.

It explores how deep learning and audio analysis can help detect voice-cloning attempts in banking and digital payment scenarios, supporting safer voice-based communication and fraud investigation.

> **Project status:** Research and prototype. VaaniShield is not a certified banking security system and should not be used as the sole basis for financial or identity-verification decisions.

---

## 🚀 Key Features

- **AI-Based Voice Analysis:** Uses deep-learning-based audio classification to assess whether a voice recording may be genuine or spoofed.
- **Multiple Model Support:** Includes AASIST and W2V2-AASIST ONNX model integrations.
- **Audio Preprocessing:** Processes audio for model-compatible inference.
- **Live Audio Monitoring:** Includes components for live microphone-based audio capture and analysis.
- **Fraud Risk Analysis:** Provides supporting risk-analysis functionality for suspicious voice interactions.
- **Voice Verification Workflow:** Includes a verification simulator and case-management interface.
- **Case Management:** Supports verification-case handling and investigation-related functionality.
- **Audio Forensics:** Includes audio-forensics functionality for additional analysis.
- **Web Dashboard:** Provides a React-based interface for interacting with the system.

## 🧠 Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, JavaScript, Vite |
| Backend | Python, FastAPI |
| AI / ML | PyTorch, ONNX Runtime |
| Audio Processing | Librosa, SoundFile, SpeechRecognition |
| Machine Learning | Scikit-learn |
| Communication | REST APIs, WebSockets |
| Development | VS Code, Git, Node.js |

## 🏗️ Project Architecture

VaaniShield is organized into three main application layers.

### 1. Frontend
The frontend provides the user interface for interacting with VaaniShield.

Technologies:
- React
- Vite
- CSS
- JavaScript

Location: `frontend/`

### 2. Backend
The backend provides the application API and supporting fraud-analysis workflows.

Technologies:
- Python
- FastAPI
- Pydantic
- Uvicorn

Location: `backend/app/`

### 3. AI/ML Engine
The AI/ML component contains audio preprocessing, model inference, model evaluation, and diagnostic utilities.

Technologies:
- PyTorch
- ONNX Runtime
- Librosa
- NumPy
- Scikit-learn

Location: `ai_ml/`

### Architecture Overview

```text
                USER
                  |
                  v
       +----------------------+
       |   React Frontend     |
       |   Vite Dashboard     |
       +----------------------+
                  |
                  v
       +----------------------+
       |    FastAPI Backend   |
       |                      |
       | Audio Analysis       |
       | Fraud Risk Engine    |
       | Verification         |
       | Case Management      |
       +----------------------+
                  |
                  v
       +----------------------+
       |     AI/ML Engine     |
       |                      |
       | Audio Preprocessing  |
       | AASIST               |
       | W2V2-AASIST          |
       | ONNX Inference       |
       +----------------------+
                  |
                  v
       +----------------------+
       | Analysis Results    |
       +----------------------+
                  |
                  v
            DASHBOARD
```

*This diagram represents the high-level project organization, not a complete specification of every runtime interaction.*

## 📁 Repository Structure

```text
VaaniShield/
│
├── ai_ml/
│   ├── audio_preprocessing.py
│   ├── voice_detector.py
│   ├── compare_models.py
│   ├── compare_cpu_cuda.py
│   ├── evaluate_models.py
│   ├── diagnose_voice.py
│   ├── collect_genuine_samples.py
│   ├── download_eval_samples.py
│   ├── phase10_diagnostics.py
│   ├── phase10_domain_analysis.py
│   ├── record_sample.py
│   └── test_live_microphone.py
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── live_app.py
│       ├── audio_forensics.py
│       ├── compliance_report.py
│       ├── risk_engine.py
│       ├── scam_detector.py
│       ├── transcription.py
│       ├── verification_cases.py
│       └── voice_biometrics.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── requirements.txt
├── .gitignore
└── README.md
```

Model files, datasets, local caches, and generated application data are intentionally excluded from Git.

## ⚙️ Prerequisites

Install the following before setting up the project:

- Python 3.11
- Node.js and npm
- Git
- VS Code (recommended)

For GPU-based model inference:
- NVIDIA GPU
- Compatible NVIDIA drivers
- Compatible CUDA-enabled PyTorch and ONNX Runtime installations

CPU/GPU compatibility depends on the installed runtime and model configuration.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_PRIVATE_REPOSITORY_URL>
cd VaaniShield
```

### 2. Create a Python virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

For an NVIDIA CUDA 12.8 PyTorch setup:

```powershell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the Python environment:

```powershell
python -m pip check
```

### 4. Install frontend dependencies

```powershell
cd frontend
npm ci
```

Return to the project root when needed:

```powershell
cd ..
```

## 🤖 Model Setup

VaaniShield expects the following ONNX model files:

```text
ai_ml/
└── models/
    ├── aasist.onnx
    └── w2v2-aasist.onnx
```

**Important:**

The model files are not included in this Git repository because of their size.

Obtain the model files from their appropriate original or authorized distribution sources and place them at the expected paths.

The exact original model download locations have not yet been verified for this repository. Do not download replacement models from untrusted sources.

## ▶️ Running the Application

### 1. Start the backend

From the project root, activate the Python virtual environment and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.live_app:app --host 127.0.0.1 --port 8000
```

The backend should start at:

```text
http://127.0.0.1:8000
```

### 2. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm run dev
```

Use the local development URL printed by Vite in the terminal.

The frontend and backend must both be running for workflows that depend on backend services.

## 🧪 Development and Validation

### Frontend lint

```powershell
cd frontend
npm run lint
```

### Frontend production build

```powershell
npm run build
```

### Frontend dependency security audit

```powershell
npm audit
```

The frontend has successfully passed its lint and production-build checks in the development environment. The latest npm audit reported zero vulnerabilities.

These checks do not constitute a complete security or functional validation of the application.

## 📊 Model Evaluation

The project includes scripts for model comparison, audio diagnostics, and evaluation.

Previously recorded development evaluation results include:

| Evaluation | Result |
|---|---:|
| ASV2019 evaluation samples | 37 |
| AASIST ROC-AUC | 0.9464 |
| Genuine samples incorrectly flagged as spoof | 6 |
| Spoof samples incorrectly accepted as genuine | 0 |

These are preliminary results from a limited evaluation set, not a guarantee of real-world performance.

A genuine voice recording was also incorrectly classified as spoof during local testing. This demonstrates the need for additional testing, threshold calibration, and evaluation across different speakers, microphones, recording environments, and audio conditions.

## 🔐 Privacy and Security

Voice recordings and voice-related information can be highly sensitive.

VaaniShield follows a development approach that aims to avoid accidentally publishing sensitive project data.

- Local datasets and generated data are excluded from Git.
- Model files are excluded from Git.
- Python virtual environments are excluded.
- Frontend build outputs and dependency directories are excluded.
- Environment files and common secret-file patterns are excluded through `.gitignore`.

**Recommended before real-world deployment:**

- Implement appropriate consent and retention policies.
- Protect audio recordings and biometric information.
- Use secure transport and authentication.
- Restrict access to verification records.
- Avoid storing sensitive voice data longer than necessary.
- Perform independent security and privacy assessments.

## ⚠️ Limitations

VaaniShield is currently a research and development prototype.

Important limitations include:

- Model performance may vary across speakers and recording environments.
- Genuine speech can produce false-positive spoof predictions.
- Limited evaluation data cannot establish production-level reliability.
- Microphone quality, background noise, compression, and audio preprocessing can influence results.
- Model predictions should not be treated as definitive proof of fraud.
- Further testing is required before deployment in banking or payment systems.

## 🔮 Future Improvements

Potential future development areas include:

- Expanding evaluation datasets across different speakers and environments.
- Reducing false-positive and false-negative predictions.
- Improving model calibration and confidence handling.
- Strengthening real-time inference performance.
- Improving privacy-preserving voice verification.
- Adding more comprehensive security testing.
- Evaluating performance under realistic banking fraud scenarios.

## 👨‍💻 Development

VaaniShield is being developed as a project exploring the application of AI/ML to voice-cloning fraud detection.

Contributions, testing feedback, and suggestions are welcome.

## 📜 Disclaimer

VaaniShield is an experimental software project intended for research, education, and demonstration purposes.

It is not a certified fraud-detection solution, banking product, or replacement for established identity-verification systems. Its outputs should be reviewed alongside independent verification methods.

---

**VaaniShield — Making voice-based interactions safer through AI.** 🛡️