# Watch Once, Teach Forever (WOTF AI) — Kushal.ai 🏺

An advanced AI-powered physical skill transmission and real-time co-piloting ecosystem. WOTF AI is designed to digitize local trade skills (e.g., heritage crafts, electrical repair, knot tying, and basic sciences) and guide vocational learners in their native regional Indian languages.

---

## 🚀 Key Features

### 🎬 Phase 1: Expert Video Digitization & Synthesis
- **Webcam & File Upload**: Ingests expert performance footage or live demonstrations.
- **Dynamic Step Segmentation**: Chronologically segments continuous actions into logical milestones.
- **Validation Checklist Synthesis**: Generates strict visual checks with associated confidence thresholds.
- **Safety Hazard Identification**: Pinpoints potential errors, mapping out mitigation warnings and vocal guidance.
- **Serialized Protocol Schema**: Outputs validated profiles adhering strictly to robust [Pydantic](file:///Users/nirranjannaarayanmr/Downloads/Compiler%20part1/schema.py) structures.

### 🎯 Phase 2: Live Hands-Free Visual & Audio Coaching
- **Continuous Camera Stream**: Evaluates student performance frame-by-frame.
- **Dual-Model Arbitration**: 
  - **Fast-Loop (Gemma E2B)**: Executes instant visual confirmations and safety checks.
  - **Slow-Loop (Gemma E4B)**: Escalates near-threshold or ambiguous actions to deep reasoning layers to avoid false positives.
- **Interactive Step Visual Guide**: A lightweight, real-time client-side SVG drawing engine that illustrates the target task on-the-fly (e.g., mathematical binomial expansion boards, marine knots, or hot soldering joints).
- **Animated Companion (Nano Banana)**: Fully responsive animated SVG companion exhibiting real-time state changes (bouncing, worrying, and dance-spinning celebrations).

### 🇮🇳 Multi-lingual Vernacular Interactions (`antigravity-preview-05-2026`)
- Prioritizes Indian regional languages over foreign dialects. 
- Supported locales: **Hindi (हिंदी)** `hi-IN`, **Tamil (தமிழ்)** `ta-IN`, **Telugu (తెలుగు)** `te-IN`, **Kannada (ಕನ್ನಡ)** `kn-IN`, **Malayalam (മലയാളം)** `ml-IN`, **Marathi (मराठी)** `mr-IN`, **Bengali (বাংলা)** `bn-IN`, and **Gujarati (ગુજરાતી)** `gu-IN` (alongside Spanish and French).
- **Hands-Free Speech Commands**: Recognizes vocal directions in native accents to switch languages seamlessly on-the-fly (e.g., saying *"speak in Kannada"* or *"हिंदी"* translates the entire UI, checklist, and audio loop instantly).

---

## 🛠 Model Registry Architecture

All intelligence routing maps to next-generation Google models to deliver fast visual processing and conversational responses:

| Feature Task | Target Model | Purpose |
| :--- | :--- | :--- |
| **Vision Evaluation** | `gemini-3.5-flash` | Continuous frame evaluation against safety & checklist regulations. |
| **Live Translation** | `gemini-3.5-live-translate-preview` | Real-time translation of instruction sets and coaching alerts. |
| **Multimodal Flow** | `gemini-omni-flash-preview` | Interactive video and audio streaming analysis. |
| **Voice Coaching** | `gemini-3.1-flash-tts-preview` | High-quality text-to-speech feedback synthesis. |
| **Interactive Live** | `gemini-3.1-flash-live-preview` | Low-latency audio-to-audio dialogue cycles. |
| **Banana Generator** | `gemini-3.1-flash-lite-image` | On-the-fly custom illustration assets. |

---

## 📦 System Installation & Execution

### Prerequisites
- Python 3.10+
- A Google GenAI API Key configured in your environment.

### 1. Setup Environment
Clone the repository and install dependencies:
```bash
git clone https://github.com/nirranjan121/WOTF_AI.git
cd WOTF_AI
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
PORT=5001
```

### 2. Launch the Application
Run the local Flask server:
```bash
python3 app.py
```
Open your browser and navigate to `http://127.0.0.1:5001` to begin training!

---

## 🗃 File & Workspace Directory Structure

- [`app.py`](file:///Users/nirranjannaarayanmr/Downloads/Compiler%20part1/app.py) — Core Flask application routing model evaluations, live translation pipelines, and frame-captures.
- [`schema.py`](file:///Users/nirranjannaarayanmr/Downloads/Compiler%20part1/schema.py) — Enriched Pydantic schemas validating step structures, checklist verifications, and voice guidelines.
- [`person_d_coaching.py`](file:///Users/nirranjannaarayanmr/Downloads/Compiler%20part1/person_d_coaching.py) — State machine tracking training milestones, managing dual-model arbitration, and queueing audio feedback.
- [`phase1_synthesis.py`](file:///Users/nirranjannaarayanmr/Downloads/Compiler%20part1/phase1_synthesis.py) — Video segmenter mapping demonstration recordings to structural JSON modules.
- [`templates/index.html`](file:///Users/nirranjannaarayanmr/Downloads/Compiler%20part1/templates/index.html) — Sleek, glassmorphic student-facing dashboard with hands-free STT commands, high-fidelity Web Speech voice configurations, and dynamic SVG illustrations.
- `*.json` — Serilized physical skill database files (e.g., `algebra1.json`, `bowline_knot.json`, `soldering_join.json`, etc.).
